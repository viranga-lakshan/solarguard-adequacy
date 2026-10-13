"""Task 12 (E3 continuation) -- Jharkhand counterfactual model, houses 2-4.

Same Algorithm 1 procedure as train_model.py/train_heed.py/train_star.py --
no methodology change. Jharkhand has no appliance-level load meter at all
(verified against the supplied MATLAB calibration script -- see
jharkhand_check.py's docstring), so "load" here is DERIVED from an energy
balance over the two current channels that do exist:

    load_w = PV_power_w + Battery_Current_A * Battery_V

PV_power_w = -PV_Current_A * PV_V (note the negative sign: verified
empirically that PV_Current_A is recorded with the OPPOSITE sign convention
from what "generation is positive" would suggest -- raw PV_Current_A reads
strongly NEGATIVE at solar noon, strongly positive at night; negating it
produces a properly non-negative daytime-peaked generation curve). Battery
power is added with its own native sign, where the supplied MATLAB script
documents "+ battery current means discharging; - means charging" -- so a
discharging battery (positive) ADDS to what PV alone could supply, and a
charging battery (negative) SUBTRACTS (some generation is going into the
battery rather than to load). Verified against House 2: produces a sensible
diurnal load shape (quiet overnight baseload ~3.5-6 W, evening peak ~15-17 W
around 19:00-20:00, consistent with a lights-and-fan system) once averaged
by hour, though the per-minute signal is noisy (negative-load fraction ~40%
before flooring) -- a direct consequence of the sensor's low ADC resolution
(the dissertation's own Table 3.2 records only a 354-509 raw-count span for
these channels). Negative instantaneous values are floored to 0 (physical
load cannot be negative; a negative reading is measurement noise around a
near-zero true load), which is a cleaning step, not a methodology change.

This derivation makes Jharkhand's "load" the least direct signal of any
site in this study -- weaker evidence than Mafate/HEED/STAR's appliance
meters -- and that is reported as a limitation, not smoothed over.
"""

from __future__ import annotations

import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")

from jharkhand_check import HOUSES, UNCONSTRAINED_BATTERY_V, load_house
from paths import DATA_DIR, MODELS_DIR, PLOTS_DIR
from train_model import rearrange_quantile_pair

FEATURE_COLS = ["Hour", "DayOfWeek", "Month_sin", "Month_cos", "Trend_days"]
CATEGORICAL_COLS = ["Hour", "DayOfWeek"]
QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}
TAU_GRID = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
TAU_SELECTION_SAMPLE_N = 3_000
TRAIN_SAMPLE_N = 10_000
FIT_FRACTION = 0.60
CALIBRATION_FRACTION = 0.15
ALPHA = 0.001
TARGET_COL = "load_w"
MIN_USABLE_ROWS = 200


def _make_pipeline(quantile: float) -> Pipeline:
    ct = ColumnTransformer(
        [("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_COLS)],
        remainder="passthrough",
    )
    return Pipeline(
        [("features", ct), ("model", QuantileRegressor(quantile=quantile, alpha=ALPHA, solver="highs"))]
    )


def _crps_from_quantiles(y_true: np.ndarray, quantile_preds: dict[float, np.ndarray]) -> float:
    losses = []
    for tau in sorted(quantile_preds):
        diff = y_true - quantile_preds[tau]
        pinball = np.maximum(tau * diff, (tau - 1) * diff)
        losses.append(pinball.mean())
    return float(2.0 * np.mean(losses))


def prepare_house(filename: str) -> pd.DataFrame:
    df = load_house(filename)
    df["PV_power_w"] = -df["PV_Current_A"] * df["PV_V"]
    df["load_w"] = (df["PV_power_w"] + df["Battery_Current_A"] * df["Battery_V"]).clip(lower=0.0)
    df["unconstrained"] = df["Battery_V"] > UNCONSTRAINED_BATTERY_V

    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.dayofweek
    month = df.index.month
    df["Month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["Month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["Trend_days"] = (df.index - df.index.min()).total_seconds() / 86400.0
    return df


def select_tau(fit: pd.DataFrame, calibration: pd.DataFrame, placebo_tolerance_w: float) -> tuple[float, pd.DataFrame]:
    fit_sample = fit.sample(n=min(TAU_SELECTION_SAMPLE_N, len(fit)), random_state=0)
    X_fit, y_fit = fit_sample[FEATURE_COLS], fit_sample[TARGET_COL]
    X_cal, y_cal = calibration[FEATURE_COLS], calibration[TARGET_COL]

    rows, chosen = [], None
    for tau in TAU_GRID:
        pipe = _make_pipeline(tau)
        pipe.fit(X_fit, y_fit)
        mean_supp = float(np.mean(np.clip(pipe.predict(X_cal) - y_cal, 0.0, None)))
        rows.append({"tau": tau, "calibration_mean_suppression_w": round(mean_supp, 2)})
        if chosen is None and mean_supp < placebo_tolerance_w:
            chosen = tau
    return chosen if chosen is not None else TAU_GRID[-1], pd.DataFrame(rows)


def run_house(filename: str) -> dict | None:
    house_tag = filename.split("_January")[0]
    df = prepare_house(filename)
    frame = df.loc[df["unconstrained"], FEATURE_COLS + [TARGET_COL]].dropna()

    if len(frame) < MIN_USABLE_ROWS:
        print(f"  {house_tag}: only {len(frame)} usable unconstrained rows -- declining to fit")
        return {"house": house_tag, "status": "low_confidence_insufficient_data", "n_rows": len(frame)}

    n = len(frame)
    fit = frame.iloc[: int(n * FIT_FRACTION)]
    calibration = frame.iloc[int(n * FIT_FRACTION): int(n * (FIT_FRACTION + CALIBRATION_FRACTION))]
    test = frame.iloc[int(n * (FIT_FRACTION + CALIBRATION_FRACTION)):]

    if len(fit) < 30 or len(calibration) < 10 or len(test) < 10:
        print(f"  {house_tag}: split too thin -- declining")
        return {"house": house_tag, "status": "low_confidence_split_too_thin", "n_rows": n}

    placebo_tolerance_w = max(2.0, 0.1 * fit[TARGET_COL].mean())
    tau, tau_table = select_tau(fit, calibration, placebo_tolerance_w)
    print(f"  {house_tag} tau selection (tolerance={placebo_tolerance_w:.1f} W):")
    print("    " + tau_table.to_string(index=False).replace("\n", "\n    "))
    print(f"  selected tau = {tau}")

    fit_sample = fit.sample(n=min(TRAIN_SAMPLE_N, len(fit)), random_state=0)
    X_fit, y_fit = fit_sample[FEATURE_COLS], fit_sample[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    preds_test = {}
    for name, q in {**QUANTILES, "qtau": tau}.items():
        pipe = _make_pipeline(q)
        pipe.fit(X_fit, y_fit)
        preds_test[q] = pipe.predict(X_test)
        with open(MODELS_DIR / f"jharkhand_{house_tag.lower()}_{name}.pkl", "wb") as fh:
            pickle.dump(pipe, fh)

    preds_test[0.10], preds_test[0.90] = rearrange_quantile_pair(preds_test[0.10], preds_test[0.90])

    mae = float(np.mean(np.abs(preds_test[tau] - y_test)))
    crps = _crps_from_quantiles(y_test.to_numpy(), preds_test)
    coverage = float(np.mean((y_test >= preds_test[0.10]) & (y_test <= preds_test[0.90])) * 100.0)

    with open(MODELS_DIR / f"jharkhand_{house_tag.lower()}_qtau.pkl", "rb") as fh:
        model_qtau = pickle.load(fh)
    full_pred = model_qtau.predict(frame[FEATURE_COLS])
    suppression = np.clip(full_pred - frame[TARGET_COL], 0.0, None)
    mean_suppression = float(suppression.mean())

    plot_path = PLOTS_DIR / f"e3_jharkhand_{house_tag.lower()}.png"
    fig, ax = plt.subplots(figsize=(11, 4.5))
    window = test.iloc[: min(len(test), 2000)]
    ax.plot(window.index, window[TARGET_COL], color="#4b5563", linewidth=1.0, label="Derived load (test)")
    ax.plot(window.index, preds_test[tau][: len(window)], color="#1f6feb", linewidth=1.2, label=f"Q{tau:.2f} counterfactual")
    ax.set_title(f"Jharkhand {house_tag} -- E3 cross-site, held-out test window")
    ax.set_xlabel("Time")
    ax.set_ylabel("Watts (derived)")
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return {
        "house": house_tag,
        "status": "fitted",
        "tau": tau,
        "n_rows": n,
        "n_fit": len(fit),
        "n_calibration": len(calibration),
        "n_test": len(test),
        "test_MAE_W": round(mae, 2),
        "test_CRPS_W": round(crps, 2),
        "Q10_Q90_coverage_pct": round(coverage, 2),
        "mean_suppression_w": round(mean_suppression, 2),
        "mean_load_w": round(float(frame[TARGET_COL].mean()), 2),
    }


def main() -> None:
    results = []
    for filename in HOUSES:
        print(f"{filename} ...")
        result = run_house(filename)
        if result is not None:
            results.append(result)
            print(f"  {result}")

    table = pd.DataFrame(results)
    print()
    print(table.to_string(index=False))
    table.to_csv(DATA_DIR / "jharkhand_e3_results.csv", index=False)
    print()
    print(f"saved -> {DATA_DIR / 'jharkhand_e3_results.csv'}")


if __name__ == "__main__":
    main()
