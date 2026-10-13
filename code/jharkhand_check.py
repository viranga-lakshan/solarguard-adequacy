"""Task 12 -- Jharkhand binding check.

Elevated priority: the dissertation's own "Future Recommendations" (Sec. 7.6)
names this as the single immediate next step, since Jharkhand is the
smallest system in the collection (12 V, lights + fan) and therefore the
most likely of the four sites to show real (not just injected) binding.

Three corrections versus a naive reading of the supplied documentation, all
verified directly against the raw files and the supplied MATLAB calibration
script (data_log_load_profile_calibration.m):

1. The CSV already carries PRE-CALIBRATED "PV Current" and "Battery Current"
   columns appended after the five raw channels. The MATLAB script is not
   needed to reproduce them from scratch.
2. Channel #3 (12.2-12.9 V range, narrow) is Battery Voltage; channel #1
   (0-20 V range, swings with sunlight) is PV Voltage -- the OPPOSITE of
   what a naive "channel #1 = battery voltage" assumption would give. Using
   channel #1 for the unconstrained threshold would apply a 12 V battery
   rule to a PV voltage trace and produce a meaningless binding rate.
3. The documentation describes channel #2 as "Load current" and channel #4
   as "Solar current", but the MATLAB script's own calibration constants
   (slope.channel_2/zero.channel_2) are commented "+ battery current means
   discharging; - means charging" -- i.e. channel #2 calibrates into the
   provided "Battery Current" column, not a load channel, and channel #4
   into "PV Current". There is no direct load-current channel in this
   dataset at all; see train_jharkhand.py for how load is derived from the
   two current channels that do exist.

File quirks handled: the real header sits 16 metadata rows into the file
(statistics block, channel/calibration/unit/description rows); the file is
not UTF-8 (the degree sign in "CH0 [deg C]" is Windows-1252), so latin-1 is
used; timestamps are "M/D/YY H:MM" with no leading zeros.
"""

from __future__ import annotations

import pandas as pd

from paths import DATA_DIR, JHARKHAND_RAW

HEADER_SKIPROWS = 16
# Positional names matching the file's actual 8 columns: Time, CH0..CH4, then
# the two pre-calibrated current columns. CH2/CH4 are kept as raw digits
# (unused directly -- PV_Current_A/Battery_Current_A are the calibrated
# values) but named correctly per the MATLAB script, not per the misleading
# documentation (see module docstring, point 3).
COLUMN_NAMES = [
    "Time", "Temp_C", "PV_V", "Battery_current_digit_raw", "Battery_V",
    "PV_current_digit_raw", "PV_Current_A", "Battery_Current_A",
]
UNCONSTRAINED_BATTERY_V = 11.5  # 12 V lead-acid system
BINDING_RATE_EVIDENCE_THRESHOLD_PCT = 2.0
# A connected 12 V lead-acid cell does not physically read near 0 V; values
# below this are sensor dropouts/sentinels, not deep discharge, and are
# dropped before binding is assessed (verified: House_4's recorded 0.12 V
# minimum is exactly this kind of single-sample artifact).
PLAUSIBLE_BATTERY_V_FLOOR = 5.0

HOUSES = ["House_2_January_2016.csv", "House_3_January_2016.csv", "House_4_January_2016.csv"]


def load_house(filename: str) -> pd.DataFrame:
    df = pd.read_csv(
        JHARKHAND_RAW / filename,
        skiprows=HEADER_SKIPROWS,
        encoding="latin-1",
        on_bad_lines="skip",
    )
    df.columns = COLUMN_NAMES
    df["Time"] = pd.to_datetime(df["Time"], format="%m/%d/%y %H:%M", errors="coerce")
    df = df.dropna(subset=["Time"]).set_index("Time").sort_index()
    for col in COLUMN_NAMES[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def binding_check(df: pd.DataFrame, house: str) -> dict:
    raw_voltage = df["Battery_V"].dropna()
    dropped_sentinels = int((raw_voltage < PLAUSIBLE_BATTERY_V_FLOOR).sum())
    voltage = raw_voltage[raw_voltage >= PLAUSIBLE_BATTERY_V_FLOOR]
    total_hours = len(voltage) / 60.0
    binding_hours = (voltage <= UNCONSTRAINED_BATTERY_V).sum() / 60.0
    binding_rate_pct = 100.0 * binding_hours / total_hours if total_hours else float("nan")

    return {
        "house": house,
        "total_hours": round(total_hours, 1),
        "binding_hours": round(binding_hours, 1),
        "binding_rate_pct": round(binding_rate_pct, 4),
        "min_battery_v": round(float(voltage.min()), 2) if len(voltage) else float("nan"),
        "max_battery_v": round(float(voltage.max()), 2) if len(voltage) else float("nan"),
        "mean_battery_v": round(float(voltage.mean()), 2) if len(voltage) else float("nan"),
        "dropped_sentinel_readings": dropped_sentinels,
    }


def main() -> None:
    results = []
    for filename in HOUSES:
        print(f"loading {filename} ...")
        df = load_house(filename)
        result = binding_check(df, filename.split("_January")[0])
        results.append(result)

        print(f"  {result['house']} binding rate: {result['binding_rate_pct']}%")
        print(f"  Min voltage: {result['min_battery_v']} V  Max: {result['max_battery_v']} V  Mean: {result['mean_battery_v']} V")
        if result["binding_rate_pct"] > BINDING_RATE_EVIDENCE_THRESHOLD_PCT:
            print("  REAL SUPPRESSION EVIDENCE FOUND")
        else:
            print("  Site adequately provisioned (like Mafate)")
        print()

        out_path = DATA_DIR / f"jharkhand_{result['house'].lower()}_merged.parquet"
        df.to_parquet(out_path)
        print(f"  saved -> {out_path}")
        print()

    table = pd.DataFrame(results)
    print("Jharkhand binding summary:")
    print(table.to_string(index=False))
    table.to_csv(DATA_DIR / "jharkhand_binding_rates.csv", index=False)
    print()
    print(f"saved -> {DATA_DIR / 'jharkhand_binding_rates.csv'}")


if __name__ == "__main__":
    main()
