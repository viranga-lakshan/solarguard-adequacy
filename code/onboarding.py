"""Household onboarding and operational alert logic (Algorithm 1, productionised).

Turns the offline benchmark into something a solar operator can run per new
household without editing scripts:

  1. validate_schema / daily completeness  -- required columns, >=80% valid
     hourly readings per day (tolerates 2G/GSM packet loss)
  2. baseline gatekeeper                  -- >=30 valid unconstrained days, and
     rejection of batteries that are already degraded at commissioning
  3. train_household                      -- Algorithm 1: q10/q50/q90/q_tau
     QuantileRegressor (highs), data-driven tau, monotone rearrangement of the
     descriptive interval, pickles + registry entry
  4. detect_vacancy                       -- load flat-lined near 0 W for 24+ h at
     float charge is "VACANT / INACTIVE", not suppression
  5. compute_alerts                       -- Normal / Advisory / Critical tiers

Telemetry contract (any native sampling interval; resampled to hourly means):
  timestamp (column or DatetimeIndex), load_w, voltage_v and/or soc_pct,
  optional irradiance_w_m2 and temperature_c.

Design choices the spec left open (all overridable by constant/argument):
  * Thresholds are written for a 48 V system. For other voltages they scale by
    system_voltage_v / 48 (the registry also stores the resolved per-unit values).
  * SOC-metered units use SOC > 20% as "unconstrained" and SOC < 10% as the
    low-cutoff analogue of 46.0 V.
  * "Before 10:00 PM" is read as hours 06:00-21:59, i.e. the PV-recharge and
    evening window. An overnight dip is the normal tail of the previous
    evening's use and is not by itself evidence of a defect.
  * A commissioning defect needs cutoffs on >= DEFECT_MIN_DAYS distinct days,
    so one sensor glitch cannot reject a good installation.
  * The tau tolerance is max(5 W, 10% of mean fit-set load), since the
    benchmark used 50 W for Mafate and 5 W for appliance-scale sites.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from paths import CODE_DIR, DATA_DIR, MODELS_DIR
from train_model import rearrange_quantile_pair

SCHEMA_VERSION = 1
REGISTRY_PATH = DATA_DIR / "household_registry.json"

# --- data quality ---------------------------------------------------------
MIN_DAILY_COMPLETENESS = 0.80      # fraction of the 24 hourly slots that must hold data
MIN_VALID_DAYS = 30
UNCONSTRAINED_DAY_FRACTION = 0.80  # share of a day's valid hours that must be unconstrained
NOMINAL_VOLTAGE_V = 48.0
VOLTAGE_OK_V = 47.0
VOLTAGE_CUTOFF_V = 46.0
SOC_OK_PCT = 20.0
SOC_CUTOFF_PCT = 10.0
DEFECT_WINDOW_HOURS = range(6, 22)  # 06:00-21:59
DEFECT_MIN_DAYS = 3
WEATHER_MIN_COVERAGE = 0.80

# --- training (same recipe as train_model.py) -------------------------------
LEAN_FEATURES = ["Hour", "DayOfWeek", "Month_sin", "Month_cos", "Trend_days"]
WEATHER_FEATURES = LEAN_FEATURES + ["Gincl", "Ta"]
CATEGORICAL_COLS = ["Hour", "DayOfWeek"]
QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}
TAU_GRID = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
FIT_FRACTION, CALIBRATION_FRACTION = 0.60, 0.15
TRAIN_SAMPLE_N = 20_000
ALPHA = 0.001
MIN_TOLERANCE_W = 5.0
TOLERANCE_LOAD_FRACTION = 0.10

# --- vacancy guard and alert tiers -------------------------------------------
DEFAULT_VACANCY_LOAD_W = 5.0
VACANCY_MIN_HOURS = 24
FLOAT_VOLTAGE_QUANTILE = 0.90
FLOAT_VOLTAGE_MARGIN = 0.99
FLOAT_SOC_PCT = 95.0
DEFAULT_ADVISORY_W = 50.0
DEFAULT_CRITICAL_W = 100.0
ROLLING_HOURS = 24
SUSTAIN_DAYS = 3

STATUS_ONBOARDED = "ONBOARDED"
STATUS_DEFECT = "COMMISSIONING_DEFECT"
STATUS_DEFECT_MSG = "COMMISSIONING_DEFECT (Battery already degraded upon installation)"
STATUS_INSUFFICIENT = "INSUFFICIENT_BASELINE"
STATUS_BAD_DATA = "REJECTED_DATA_QUALITY"

TIER_NORMAL, TIER_ADVISORY, TIER_CRITICAL = "NORMAL", "ADVISORY", "CRITICAL"
TIER_VACANT, TIER_VOLTAGE_ALARM, TIER_UNKNOWN = "VACANT/INACTIVE", "VOLTAGE_ALARM", "UNKNOWN"

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


# --- registry ---------------------------------------------------------------
def load_registry(path: Path | None = None) -> dict:
    path = Path(path or REGISTRY_PATH)
    with open(path, encoding="utf-8") as fh:
        registry = json.load(fh)
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported registry schema_version {registry.get('schema_version')!r}")
    return registry


def save_registry(registry: dict, path: Path | None = None) -> None:
    """Atomic write: a crash mid-save can never leave a truncated registry."""
    path = Path(path or REGISTRY_PATH)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def get_household(household_id: str, registry_path: Path | None = None) -> dict:
    households = load_registry(registry_path)["households"]
    if household_id not in households:
        raise KeyError(f"household {household_id!r} is not in the registry")
    return households[household_id]


def resolve_model_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else CODE_DIR / p


# --- telemetry validation -----------------------------------------------------
def standardize_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the schema and return an hourly frame indexed by hour start.

    Columns: load_w, battery (the channel used for health: voltage_v else
    soc_pct), battery_metric is returned in .attrs, plus optional
    irradiance_w_m2 / temperature_c and n_valid (valid sub-readings per hour).
    Raises ValueError on a schema problem."""
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except (ValueError, TypeError) as exc:
            raise ValueError("telemetry needs a 'timestamp' column or a DatetimeIndex") from exc
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df[~df.index.isna()].sort_index()

    if "load_w" not in df.columns:
        raise ValueError("telemetry is missing required column 'load_w'")
    if "voltage_v" not in df.columns and "soc_pct" not in df.columns:
        raise ValueError("telemetry needs 'voltage_v' or 'soc_pct'")

    metric = "voltage_v" if "voltage_v" in df.columns else "soc_pct"
    cols = ["load_w", metric] + [c for c in ("irradiance_w_m2", "temperature_c") if c in df.columns]
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    num["load_w"] = num["load_w"].clip(lower=0.0)  # sign convention glitches must not fake demand

    hourly = num.resample("1h").mean()
    hourly = hourly.rename(columns={metric: "battery"})
    hourly["valid"] = hourly["load_w"].notna() & hourly["battery"].notna()
    hourly.attrs["battery_metric"] = metric
    return hourly


def day_completeness(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per calendar day: share of valid hourly readings and whether it clears
    MIN_DAILY_COMPLETENESS."""
    frac = hourly["valid"].groupby(hourly.index.normalize()).sum() / 24.0
    return pd.DataFrame({"valid_fraction": frac, "complete": frac >= MIN_DAILY_COMPLETENESS})


def battery_thresholds(metric: str, system_voltage_v: float = NOMINAL_VOLTAGE_V) -> tuple[float, float]:
    """(ok, cutoff) for the battery channel. Voltage thresholds scale with system voltage."""
    if metric == "soc_pct":
        return SOC_OK_PCT, SOC_CUTOFF_PCT
    scale = system_voltage_v / NOMINAL_VOLTAGE_V
    return VOLTAGE_OK_V * scale, VOLTAGE_CUTOFF_V * scale


# --- gatekeeper ---------------------------------------------------------------
@dataclass
class OnboardingResult:
    household_id: str
    status: str
    message: str
    onboarded: bool = False
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"STATUS: {self.message}" if not self.onboarded else f"STATUS: {self.status} - {self.message}"


def _reject(hid: str, status: str, message: str, **details) -> OnboardingResult:
    return OnboardingResult(hid, status, message, False, details)


def check_commissioning_defect(
    hourly: pd.DataFrame, ok: float, cutoff: float, complete_days: pd.DatetimeIndex
) -> list[pd.Timestamp]:
    """Days on which the battery hit its low cutoff during 06:00-21:59."""
    in_window = hourly.index.hour.isin(list(DEFECT_WINDOW_HOURS))
    low = hourly["valid"] & (hourly["battery"] < cutoff) & in_window
    on_complete = hourly.index.normalize().isin(complete_days)
    return sorted(hourly.index[low & on_complete].normalize().unique())


def float_threshold(hourly: pd.DataFrame, metric: str) -> float:
    if metric == "soc_pct":
        return FLOAT_SOC_PCT
    return round(float(hourly.loc[hourly["valid"], "battery"].quantile(FLOAT_VOLTAGE_QUANTILE)) * FLOAT_VOLTAGE_MARGIN, 2)


# --- features / training --------------------------------------------------------
def build_features(hourly: pd.DataFrame, trend_origin: pd.Timestamp, weather: bool) -> pd.DataFrame:
    idx = hourly.index
    month = idx.month
    feats = pd.DataFrame(
        {
            "Hour": idx.hour,
            "DayOfWeek": idx.dayofweek,
            "Month_sin": np.sin(2 * np.pi * month / 12.0),
            "Month_cos": np.cos(2 * np.pi * month / 12.0),
            "Trend_days": (idx - pd.Timestamp(trend_origin)).total_seconds() / 86400.0,
        },
        index=idx,
    )
    if weather:
        feats["Gincl"] = hourly["irradiance_w_m2"]
        feats["Ta"] = hourly["temperature_c"]
    return feats


def _make_pipeline(quantile: float) -> Pipeline:
    ct = ColumnTransformer(
        [("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_COLS)],
        remainder="passthrough",
    )
    return Pipeline([("features", ct), ("model", QuantileRegressor(quantile=quantile, alpha=ALPHA, solver="highs"))])


def select_tau(X_fit, y_fit, X_cal, y_cal, tolerance_w: float) -> tuple[float, pd.DataFrame]:
    """Smallest tau whose mean suppression on the unconstrained calibration
    split is under tolerance (same rule and fallback as train_model.select_tau)."""
    rows, chosen = [], None
    for tau in TAU_GRID:
        pipe = _make_pipeline(tau).fit(X_fit, y_fit)
        mean_supp = float(np.mean(np.clip(pipe.predict(X_cal) - y_cal, 0.0, None)))
        rows.append({"tau": tau, "calibration_mean_suppression_w": round(mean_supp, 2)})
        if chosen is None and mean_supp < tolerance_w:
            chosen = tau
    return (chosen if chosen is not None else TAU_GRID[-1]), pd.DataFrame(rows)


def train_household(
    train: pd.DataFrame, feature_cols: list[str]
) -> tuple[dict[str, Pipeline], float, dict]:
    """train: chronologically ordered frame of feature_cols + 'load_w' (unconstrained rows only)."""
    n = len(train)
    fit_end = int(n * FIT_FRACTION)
    cal_end = fit_end + int(n * CALIBRATION_FRACTION)
    fit, cal, test = train.iloc[:fit_end], train.iloc[fit_end:cal_end], train.iloc[cal_end:]

    tolerance = max(MIN_TOLERANCE_W, TOLERANCE_LOAD_FRACTION * float(fit["load_w"].mean()))
    tau, tau_table = select_tau(fit[feature_cols], fit["load_w"], cal[feature_cols], cal["load_w"], tolerance)

    fit_sample = fit.sample(n=min(TRAIN_SAMPLE_N, len(fit)), random_state=0)
    models, test_preds = {}, {}
    for name, q in {**QUANTILES, "qtau": tau}.items():
        models[name] = _make_pipeline(q).fit(fit_sample[feature_cols], fit_sample["load_w"])
        test_preds[name] = models[name].predict(test[feature_cols])
    low, high = rearrange_quantile_pair(test_preds["q10"], test_preds["q90"])  # descriptive interval only

    y_test = test["load_w"].to_numpy()
    summary = {
        "tau": tau,
        "tau_tolerance_w": round(tolerance, 2),
        "tau_table": tau_table.to_dict("records"),
        "n_train_rows": n,
        "n_test_rows": len(test),
        "test_MAE_W_qtau": round(float(np.mean(np.abs(test_preds["qtau"] - y_test))), 2),
        "q10_q90_coverage_pct": round(float(np.mean((y_test >= low) & (y_test <= high)) * 100.0), 2),
    }
    return models, tau, summary


def onboard_household(
    household_id: str,
    telemetry: pd.DataFrame,
    site: str = "unspecified",
    system_voltage_v: float = NOMINAL_VOLTAGE_V,
    registry_path: Path | None = None,
    models_dir: Path | None = None,
    dry_run: bool = False,
    telemetry_dir: Path | None = None,
) -> OnboardingResult:
    """Validate commissioning telemetry, train Algorithm 1 and register the household.

    `telemetry` is the baseline period (>= MIN_VALID_DAYS valid unconstrained days).
    Never touches an existing registry entry or model file: a duplicate id is
    rejected. dry_run runs every check and the fit but writes nothing.
    telemetry_dir (opt-in; the CLI passes data/telemetry) also stores the
    baseline telemetry as <id>.parquet so the dashboard has data to show."""
    registry_path = Path(registry_path or REGISTRY_PATH)
    models_dir = Path(models_dir or MODELS_DIR)

    if not _ID_RE.match(household_id or ""):
        return _reject(household_id, STATUS_BAD_DATA, "household_id must match [A-Za-z0-9_-]{1,64}")
    registry = load_registry(registry_path) if registry_path.exists() else {"schema_version": SCHEMA_VERSION, "households": {}}
    if household_id in registry["households"]:
        return _reject(household_id, STATUS_BAD_DATA, f"household {household_id!r} is already registered")
    model_files = {q: models_dir / f"{household_id}_{q}.pkl" for q in [*QUANTILES, "qtau"]}
    if any(p.exists() for p in model_files.values()):
        return _reject(household_id, STATUS_BAD_DATA, f"model files for {household_id!r} already exist; refusing to overwrite")

    # 1. schema + completeness
    try:
        hourly = standardize_telemetry(telemetry)
    except ValueError as exc:
        return _reject(household_id, STATUS_BAD_DATA, f"REJECTED_SCHEMA: {exc}")
    metric = hourly.attrs["battery_metric"]
    ok, cutoff = battery_thresholds(metric, system_voltage_v)

    days = day_completeness(hourly)
    complete_days = days.index[days["complete"]]
    if len(complete_days) == 0:
        return _reject(household_id, STATUS_BAD_DATA, "no day reaches 80% valid hourly readings")

    # 2a. commissioning-defect check comes first: a degraded battery also
    # thins the unconstrained-day count, and that must not mask the real cause.
    defect_days = check_commissioning_defect(hourly, ok, cutoff, complete_days)
    if len(defect_days) >= DEFECT_MIN_DAYS:
        return _reject(
            household_id, STATUS_DEFECT, STATUS_DEFECT_MSG,
            defect_days=[d.date().isoformat() for d in defect_days],
            cutoff=cutoff, metric=metric,
        )

    # 2b. >= 30 valid unconstrained days
    in_complete = hourly.index.normalize().isin(complete_days)
    unconstrained = hourly["valid"] & (hourly["battery"] > ok)
    day_key = hourly.index.normalize()
    valid_per_day = hourly["valid"].groupby(day_key).sum()
    unc_per_day = unconstrained.groupby(day_key).sum()
    good_days = days["complete"] & (unc_per_day.reindex(days.index) / valid_per_day.reindex(days.index).clip(lower=1) >= UNCONSTRAINED_DAY_FRACTION)
    if int(good_days.sum()) < MIN_VALID_DAYS:
        return _reject(
            household_id, STATUS_INSUFFICIENT,
            f"INSUFFICIENT_BASELINE: {int(good_days.sum())} valid unconstrained days, need {MIN_VALID_DAYS}",
            valid_days=int(good_days.sum()), complete_days=int(len(complete_days)),
        )

    # 3. feature tier + training
    weather_cols = {"irradiance_w_m2", "temperature_c"}
    weather = weather_cols <= set(hourly.columns) and bool(
        hourly.loc[hourly["valid"], list(weather_cols)].notna().all(axis=1).mean() >= WEATHER_MIN_COVERAGE
    )
    feature_cols = WEATHER_FEATURES if weather else LEAN_FEATURES
    trend_origin = hourly.index.min()
    feats = build_features(hourly, trend_origin, weather)
    rows = (unconstrained & in_complete).to_numpy()
    train = feats.loc[rows].assign(load_w=hourly.loc[rows, "load_w"]).dropna()

    models, tau, summary = train_household(train, feature_cols)

    entry = {
        "household_id": household_id,
        "site": site,
        "unit": household_id,
        "origin": "onboarded",
        "status": "active",
        "feature_tier": "weather_aware" if weather else "telemetry_only",
        "feature_cols": feature_cols,
        "model_paths": {q: _stored_path(p) for q, p in model_files.items()},
        "tau": tau,
        "trend_origin": trend_origin.isoformat(),
        "battery_metric": metric,
        "battery_ok_threshold": ok,
        "battery_cutoff_threshold": cutoff,
        "nominal_system_voltage_v": system_voltage_v if metric == "voltage_v" else None,
        "vacancy_float_threshold": float_threshold(hourly, metric),
        "vacancy_load_w": DEFAULT_VACANCY_LOAD_W,
        "advisory_suppression_w": DEFAULT_ADVISORY_W,
        "critical_suppression_w": DEFAULT_CRITICAL_W,
        "onboarded_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        "baseline_days": int(good_days.sum()),
        "training_summary": {k: v for k, v in summary.items() if k != "tau_table"},
    }

    if not dry_run:
        written = []
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
            for q, path in model_files.items():
                with open(path, "wb") as fh:
                    pickle.dump(models[q], fh)
                written.append(path)
            if telemetry_dir is not None:
                Path(telemetry_dir).mkdir(parents=True, exist_ok=True)
                saved = Path(telemetry_dir) / f"{household_id}.parquet"
                raw = telemetry if "timestamp" in telemetry.columns else telemetry.rename_axis("timestamp").reset_index()
                raw.to_parquet(saved)
                written.append(saved)
            registry["households"][household_id] = entry
            save_registry(registry, registry_path)
        except Exception:
            for path in written:  # no half-onboarded household
                path.unlink(missing_ok=True)
            raise

    return OnboardingResult(
        household_id, STATUS_ONBOARDED,
        f"{household_id} onboarded ({entry['feature_tier']}, tau={tau}, {entry['baseline_days']} baseline days)",
        True, {**summary, "feature_tier": entry["feature_tier"], "dry_run": dry_run},
    )


def _stored_path(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(CODE_DIR).as_posix()
    except ValueError:
        return str(Path(path).resolve())


# --- vacancy guard ------------------------------------------------------------
def detect_vacancy(
    load_w: pd.Series,
    battery: pd.Series,
    float_threshold_value: float,
    load_max_w: float = DEFAULT_VACANCY_LOAD_W,
    min_hours: int = VACANCY_MIN_HOURS,
) -> pd.Series:
    """True for every hour inside a run of >= min_hours consecutive hours with
    load <= load_max_w while the battery sits at/above its float level. A
    missing reading breaks the run, so a data outage is never read as vacancy."""
    cond = ((load_w <= load_max_w) & (battery >= float_threshold_value)).fillna(False).to_numpy()
    out = np.zeros(len(cond), dtype=bool)
    start = None
    for i, flag in enumerate([*cond, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_hours:
                out[start:i] = True
            start = None
    return pd.Series(out, index=load_w.index, name="vacant")


# --- multi-tier alerting ------------------------------------------------------
def _consecutive(flag: pd.Series) -> pd.Series:
    """Length of the current run of True values."""
    grp = (~flag).cumsum()
    return flag.astype(int).groupby(grp).cumsum()


def compute_alerts(
    suppression_w: pd.Series,
    battery: pd.Series,
    vacant: pd.Series,
    battery_ok: float,
    advisory_w: float = DEFAULT_ADVISORY_W,
    critical_w: float = DEFAULT_CRITICAL_W,
    window_hours: int = ROLLING_HOURS,
    sustain_days: int = SUSTAIN_DAYS,
) -> pd.DataFrame:
    """Rolling-window tiers on the hourly series.

    instant_tier is causal (trailing window only). alert_tier is the day-level
    verdict, broadcast over the day's hours, so it is final only once the day
    is complete; the per-day table is in result.attrs['daily']. Vacant hours
    are removed from both rolling means, vacant days are reported as
    VACANT/INACTIVE, and they reset the persistence counters (so an alert needs
    sustain_days fresh days after the family returns).
    Tiers: NORMAL < advisory_w; ADVISORY advisory_w..critical_w; CRITICAL >
    critical_w with rolling battery above battery_ok. Suppression above
    critical_w with the battery NOT ok is a VOLTAGE_ALARM -- conventional
    monitoring already covers it, so it is not counted as deprivation."""
    min_periods = math.ceil(window_hours * 0.8)
    supp = suppression_w.where(~vacant)
    batt = battery.where(~vacant)
    roll_supp = supp.rolling(window_hours, min_periods=min_periods).mean()
    roll_batt = batt.rolling(window_hours, min_periods=min_periods).mean()

    tier = pd.Series(TIER_NORMAL, index=supp.index, dtype=object)
    tier[roll_supp >= advisory_w] = TIER_ADVISORY
    crit = roll_supp > critical_w
    tier[crit & (roll_batt > battery_ok)] = TIER_CRITICAL
    tier[crit & (roll_batt <= battery_ok)] = TIER_VOLTAGE_ALARM
    tier[roll_supp.isna() | roll_batt.isna()] = TIER_UNKNOWN
    tier[vacant] = TIER_VACANT

    day = supp.index.normalize()
    frac = lambda mask: mask.groupby(day).mean()  # noqa: E731
    vacant_day = frac(vacant) >= 0.5
    critical_day = frac(tier == TIER_CRITICAL) > 0.5
    advisory_day = frac(tier.isin([TIER_ADVISORY, TIER_CRITICAL])) > 0.5
    full_days = pd.date_range(day.min(), day.max(), freq="D")  # a missing day breaks persistence
    vacant_day, critical_day, advisory_day = (
        s.reindex(full_days, fill_value=False) for s in (vacant_day, critical_day, advisory_day)
    )
    critical_day &= ~vacant_day
    advisory_day &= ~vacant_day

    crit_run, adv_run = _consecutive(critical_day), _consecutive(advisory_day)
    daily_tier = pd.Series(TIER_NORMAL, index=full_days, dtype=object)
    daily_tier[adv_run >= sustain_days] = TIER_ADVISORY
    daily_tier[crit_run >= sustain_days] = TIER_CRITICAL
    daily_tier[vacant_day] = TIER_VACANT

    daily = pd.DataFrame(
        {"critical_run_days": crit_run, "advisory_run_days": adv_run, "vacant": vacant_day, "alert_tier": daily_tier}
    )
    out = pd.DataFrame(
        {
            "rolling_suppression_w": roll_supp,
            "rolling_battery": roll_batt,
            "vacant": vacant,
            "instant_tier": tier,
            "alert_tier": daily_tier.reindex(day).to_numpy(),
        },
        index=supp.index,
    )
    out.attrs["daily"] = daily.to_dict("index")  # plain dict: DataFrames in attrs break pandas concat
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Onboard a household from a commissioning telemetry CSV")
    ap.add_argument("household_id")
    ap.add_argument("csv", help="columns: timestamp, load_w, voltage_v|soc_pct[, irradiance_w_m2, temperature_c]")
    ap.add_argument("--site", default="unspecified")
    ap.add_argument("--system-voltage", type=float, default=NOMINAL_VOLTAGE_V)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    result = onboard_household(
        args.household_id, pd.read_csv(args.csv, parse_dates=["timestamp"]),
        site=args.site, system_voltage_v=args.system_voltage, dry_run=args.dry_run,
        telemetry_dir=DATA_DIR / "telemetry",
    )
    print(result)
    raise SystemExit(0 if result.onboarded else 1)


if __name__ == "__main__":
    main()
