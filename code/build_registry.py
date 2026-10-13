"""Build code/data/household_registry.json for the 15 pre-trained benchmark units.

Read-only with respect to every existing artefact: it opens the merged
parquet frames (for the Trend_days origin and the float-voltage reference) and
the pickled qtau models (for the selected tau), and writes only the registry.
Entries for households added later by onboarding.py are preserved, so
re-running this script is safe.

Per-unit battery channel/thresholds mirror what each site's own training
script used to define "unconstrained": Mafate by pack voltage (48 V system,
47.0 V), Jharkhand by battery voltage (12 V system, 11.5 V), HEED/STAR/SONG by
SOC (20%).
"""

from __future__ import annotations

import pickle

import pandas as pd

from onboarding import (
    DEFAULT_ADVISORY_W,
    DEFAULT_CRITICAL_W,
    DEFAULT_VACANCY_LOAD_W,
    REGISTRY_PATH,
    SCHEMA_VERSION,
    load_registry,
    save_registry,
)
from paths import CODE_DIR, DATA_DIR, MODELS_DIR

QUANTILE_NAMES = ["q10", "q50", "q90", "qtau"]
WEATHER_FEATURES = ["Hour", "DayOfWeek", "Month_sin", "Month_cos", "Trend_days", "Gincl", "Ta"]
LEAN_FEATURES = ["Hour", "DayOfWeek", "Month_sin", "Month_cos", "Trend_days"]

# (household_id, site, unit label, model stem, merged parquet, battery metric,
#  ok threshold, cutoff threshold, nominal system volts or None, battery column in parquet)
SOC = ("soc_pct", 20.0, 10.0, None)
UNITS = [
    ("mafate_h1", "Mafate", "h1", "mafate_h1", "mafate_merged.parquet", ("voltage_v", 47.0, 46.0, 48.0), "Ubat"),
    ("mafate_h2", "Mafate", "h2", "mafate_h2", "mafate_merged.parquet", ("voltage_v", 47.0, 46.0, 48.0), "Ubat"),
    ("mafate_h3", "Mafate", "h3", "mafate_h3", "mafate_merged.parquet", ("voltage_v", 47.0, 46.0, 48.0), "Ubat"),
    ("heed_cpe1", "HEED Rwanda", "CPE1", "heed_cpe1", "heed_cpe1_merged.parquet", SOC, "soc_pct"),
    ("star_a", "STAR Sierra Leone", "A", "star_a", "star_a_merged.parquet", SOC, "soc_pct"),
    ("star_b", "STAR Sierra Leone", "B", "star_b", "star_b_merged.parquet", SOC, "soc_pct"),
    ("star_c", "STAR Sierra Leone", "C", "star_c", "star_c_merged.parquet", SOC, "soc_pct"),
    ("star_d", "STAR Sierra Leone", "D", "star_d", "star_d_merged.parquet", SOC, "soc_pct"),
    ("jharkhand_house_2", "Jharkhand India", "House_2", "jharkhand_house_2", "jharkhand_house_2_merged.parquet", ("voltage_v", 11.5, 11.25, 12.0), "Battery_V"),
    ("jharkhand_house_3", "Jharkhand India", "House_3", "jharkhand_house_3", "jharkhand_house_3_merged.parquet", ("voltage_v", 11.5, 11.25, 12.0), "Battery_V"),
    ("jharkhand_house_4", "Jharkhand India", "House_4", "jharkhand_house_4", "jharkhand_house_4_merged.parquet", ("voltage_v", 11.5, 11.25, 12.0), "Battery_V"),
    ("song_kenya_h42", "SONG Kenya", "42", "song_kenya_h42", "song_kenya_h42_merged.parquet", SOC, "soc_pct"),
    ("song_kenya_h48", "SONG Kenya", "48", "song_kenya_h48", "song_kenya_h48_merged.parquet", SOC, "soc_pct"),
    ("song_kenya_h49", "SONG Kenya", "49", "song_kenya_h49", "song_kenya_h49_merged.parquet", SOC, "soc_pct"),
    ("song_kenya_h51", "SONG Kenya", "51", "song_kenya_h51", "song_kenya_h51_merged.parquet", SOC, "soc_pct"),
]


def _entry(hid, site, unit, stem, parquet, battery, battery_col, frames) -> dict:
    metric, ok, cutoff, volts = battery
    if parquet not in frames:
        frames[parquet] = pd.read_parquet(DATA_DIR / parquet)
    df = frames[parquet]

    with open(MODELS_DIR / f"{stem}_qtau.pkl", "rb") as fh:
        tau = float(pickle.load(fh).named_steps["model"].quantile)

    if metric == "soc_pct":
        float_threshold = 95.0
    else:  # same 0.99 x 90th-percentile rule onboarding.py applies to new voltage-metered units
        float_threshold = round(float(df[battery_col].quantile(0.90)) * 0.99, 2)

    weather = hid.startswith("mafate")
    return {
        "household_id": hid,
        "site": site,
        "unit": unit,
        "origin": "pretrained_benchmark",
        "status": "active",
        "feature_tier": "weather_aware" if weather else "telemetry_only",
        "feature_cols": WEATHER_FEATURES if weather else LEAN_FEATURES,
        "model_paths": {q: f"models/{stem}_{q}.pkl" for q in QUANTILE_NAMES},
        "tau": tau,
        "trend_origin": df.index.min().isoformat(),
        "battery_metric": metric,
        "battery_ok_threshold": ok,
        "battery_cutoff_threshold": cutoff,
        "nominal_system_voltage_v": volts,
        "vacancy_float_threshold": float_threshold,
        "vacancy_load_w": DEFAULT_VACANCY_LOAD_W,
        "advisory_suppression_w": DEFAULT_ADVISORY_W,
        "critical_suppression_w": DEFAULT_CRITICAL_W,
    }


def main() -> None:
    registry = load_registry(REGISTRY_PATH) if REGISTRY_PATH.exists() else {
        "schema_version": SCHEMA_VERSION,
        "households": {},
    }
    frames: dict[str, pd.DataFrame] = {}
    for spec in UNITS:
        entry = _entry(*spec, frames)
        registry["households"][entry["household_id"]] = entry
        for path in entry["model_paths"].values():
            assert (CODE_DIR / path).exists(), f"missing model file {path}"
    save_registry(registry, REGISTRY_PATH)
    print(f"registered {len(UNITS)} benchmark units -> {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
