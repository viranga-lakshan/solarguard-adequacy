"""Task 11 -- STAR (Brazil) loader and binding check, sites A-D.

STAR is used as a secondary site only (Sec. 5.4): the record requires
substantial cleaning (sentinel values, an implied SOC above 100%, zero-
voltage readings), confirmed directly against the raw files here --
multi-day stretches read a constant SOC of exactly 1000 (=100% once
descaled), which is a plausible well-charged period rather than a defect by
itself, but genuine sentinel rows (Voltage <= 0) do appear and are dropped.

Timestamps also need cleaning: roughly 1,600 rows per site carry a Unix-epoch
sentinel timestamp (1969-12-31/1970, paired with Voltage=0 -- a "logger not
yet synced" artifact, not real data at that clock time), plus a handful of
unparseable rows. Both are dropped before the span or binding rate is
computed; leaving them in corrupts a site's reported span by decades.

Units verified empirically rather than assumed from the column names alone:
  Voltage: raw value is millivolts (divide by 1000 for V) -- confirmed
    sensible pack voltages (23-27 V) after conversion.
  SOC: raw value is parts-per-thousand (divide by 10 for %).
  Power: raw value is ALREADY IN WATTS. A naive reading of similarly-named
    Victron-style fields elsewhere might assume milliwatts, but
    Current(mA)/1000 * Voltage(mV)/1000 reproduces the raw Power column
    almost exactly (e.g. 8.747 A x 26.375 V = 230.8 W against a raw Power
    of 231) -- dividing Power by 1000 again, as that assumption would imply,
    would understate it by 1000x.

Household load (added for E3): the Battery/PV files alone do not carry a
usable net household load signal -- Battery "Power" is charging power only
(verified: it is never negative and tracks the solar curve exactly, and
DischargeEnergy is flat across multi-day stretches, i.e. this export simply
does not record discharge), so deriving load from an energy balance would be
guesswork. STAR's own MonitoringDataReadMe.txt documents a THIRD file type
this project had not previously loaded: Individual Appliance Monitor (IAM)
files, columns Time, Real_Power (W), Reactive_Power (W), Voltage, Sensor_ID
-- a direct per-circuit load meter, the same kind of signal Mafate/HEED
already provide. Total household load is the sum of Real_Power across all
Sensor_IDs active at each timestamp. Coverage is intermittent (a handful of
dated files per site, not a continuous year), and site C's IAM file carries
the same corrupted-timestamp defect as its Battery file (span reads back to
year 2000 and forward to 2036 before cleaning).
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from paths import DATA_DIR, STAR_RAW

SITES = ["A", "B", "C", "D"]
UNCONSTRAINED_SOC_PCT = 20.0
MIN_PLAUSIBLE_VOLTAGE_V = 5.0  # below this a 24V-class pack reading is a sensor artifact


def load_site_battery(site: str) -> pd.DataFrame:
    pattern = str(STAR_RAW / f"*_STAR-{site}_Battery.csv")
    files = sorted(glob.glob(pattern))
    frames = [pd.read_csv(f, parse_dates=["Time"]) for f in files]
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Time"])
    df = df[df["Time"] >= pd.Timestamp("2010-01-01")]  # drop Unix-epoch and other pre-deployment sentinel rows
    df = df.sort_values("Time").drop_duplicates(subset="Time").set_index("Time")

    df["voltage_v"] = df["Voltage"] / 1000.0
    df["soc_pct"] = df["SOC"] / 10.0
    df["power_w"] = df["Power"]  # already Watts -- verified, see module docstring
    df["current_a"] = df["Current"] / 1000.0

    df = df[df["voltage_v"] >= MIN_PLAUSIBLE_VOLTAGE_V]
    df = df[df["soc_pct"] <= 100.5]  # drop implausible over-100% sentinel rows

    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["unconstrained"] = df["soc_pct"] > UNCONSTRAINED_SOC_PCT
    return df[["voltage_v", "soc_pct", "power_w", "current_a", "Hour", "DayOfWeek", "Month", "unconstrained"]]


def load_site_iam(site: str) -> pd.Series:
    """Total household load (W), summed across IAM sensors, 1-minute mean."""
    pattern = str(STAR_RAW / f"*_STAR-{site}_IAM.csv")
    files = sorted(glob.glob(pattern))
    frames = [pd.read_csv(f, parse_dates=["Time"]) for f in files]
    if not frames:
        return pd.Series(dtype=float)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Time"])
    df = df[df["Time"] >= pd.Timestamp("2010-01-01")]
    df = df[df["Time"] <= pd.Timestamp("2025-01-01")]  # site C also carries a far-future sentinel tail
    df["Real_Power"] = pd.to_numeric(df["Real_Power"], errors="coerce")

    pivot = df.pivot_table(index="Time", columns="Sensor_ID", values="Real_Power", aggfunc="mean")
    total = pivot.sum(axis=1, min_count=1).rename("load_w")
    return total.resample("1min").mean()


def load_site_merged(site: str) -> pd.DataFrame:
    """Battery/SOC telemetry joined with IAM household load, for the E3 counterfactual model."""
    battery = load_site_battery(site)
    load = load_site_iam(site)
    if battery.empty or load.empty:
        return pd.DataFrame()

    df = battery.join(load, how="inner")  # only timestamps where both exist are usable
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    df["Trend_days"] = (df.index - df.index.min()).total_seconds() / 86400.0
    return df


def binding_check(df: pd.DataFrame, site: str) -> dict:
    if df.empty:
        return {"site": site, "rows": 0, "span_days": 0, "binding_rate_pct": float("nan"),
                "min_soc_pct": float("nan"), "min_voltage_v": float("nan")}

    span_days = (df.index.max() - df.index.min()).total_seconds() / 86400.0
    binding_rate = 100.0 * (~df["unconstrained"]).mean()
    return {
        "site": site,
        "rows": len(df),
        "span_days": round(span_days, 1),
        "binding_rate_pct": round(binding_rate, 4),
        "min_soc_pct": round(float(df["soc_pct"].min()), 1),
        "min_voltage_v": round(float(df["voltage_v"].min()), 2),
    }


def main() -> None:
    results = []
    for site in SITES:
        print(f"loading STAR-{site} ...")
        df = load_site_battery(site)
        result = binding_check(df, site)
        results.append(result)
        print(f"  {result}")
        if not df.empty:
            out_path = DATA_DIR / f"star_{site.lower()}_merged.parquet"
            df.to_parquet(out_path)
            print(f"  saved -> {out_path}")

    table = pd.DataFrame(results)
    print()
    print("STAR binding summary:")
    print(table.to_string(index=False))
    table.to_csv(DATA_DIR / "star_binding_rates.csv", index=False)
    print()
    print(f"saved -> {DATA_DIR / 'star_binding_rates.csv'}")


if __name__ == "__main__":
    main()
