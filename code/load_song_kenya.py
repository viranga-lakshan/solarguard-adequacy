"""Task 15 (new site, added post-dissertation-draft) -- Solar Nano-Grid (SONG)
Kenya loader.

Found and added as an additional, genuinely independent fifth deployment,
specifically because it is the smallest system of any dataset in this
project -- a 62 Wh Li-ion battery per household, versus Mafate's 1320 Ah
bank -- and therefore the most plausible candidate for REAL (not just
injected) capability binding, per the same "smallest system = most likely
to bind" reasoning already used to prioritise Jharkhand.

Source: Clements, A.F. & McCulloch, M. (2019). "Time series lighting
electricity data for rural households using Solar Nano-grids in Kenya."
Mendeley Data, V1, DOI: 10.17632/4yv37hngp6.1. CC BY 4.0. Methodology
described in: Clements, A.F., "Data-driven approaches enabling the design
of community energy systems in the Global South", DPhil thesis, University
of Oxford, 2019.

System: batteries charged at a central solar hub, carried home, and used to
power a 3x3W LED lighting circuit (mobile phone USB charging excluded from
this export). Two separate half-hourly CSVs per household: "lighting"
(household power draw, W, always >=0) and "charging" (hub charging power, W,
always <=0 by the file's own sign convention -- verified directly: every
value in every charging file sampled is <=0).

Three things required handling, all verified directly against the raw
files rather than assumed from the Mendeley description alone:

1. Every household's file spans the SAME fixed 146-day grid
   (2016-07-01 to 2016-11-22) regardless of how much real data that
   household actually has -- missing/non-recorded periods are zero-filled,
   with no NaN marker distinguishing "recorded zero" from "not recorded".
   The description's claim that households have "3 days to 5 months" of
   data refers to the span of ACTUAL recording, hidden inside this fixed
   grid. Verified by counting days with any nonzero charging activity per
   household: household 1 has only 4 such days despite spanning the same
   146-day file as household 51's 82. Restricting to each household's own
   [first_active_day, last_active_day] window is a necessary cleaning step,
   not optional.
2. No direct battery SOC/voltage channel exists. SOC is reconstructed by
   Coulomb-counting the half-hourly net energy flow (charging in minus
   lighting out) against the documented 62 Wh capacity, clipped to [0, 100]%
   -- the clip itself acts as an implicit re-anchor (assumes the hub
   recharges a battery to full between collections, which is the system's
   intended operating mode), since there is no voltage-based full-charge
   signature available here the way there was for Mafate.
3. No single "household" index spans the IAM-style sensor-per-circuit
   structure of Mafate/HEED -- each household file IS the whole load
   already (one lighting circuit only), so no appliance summing is needed.

Four households were selected for use -- 48, 49, 42, 51 -- the ones with
the most real (non-filler) recorded days (45-82 of 145), verified directly
rather than taking the file ordering (ascending by data length, per the
dataset's own documentation) on faith.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from paths import DATA_DIR, SONG_KENYA_RAW

BATTERY_CAPACITY_WH = 62.0
UNCONSTRAINED_SOC_THRESHOLD = 20.0
HALF_HOUR = pd.Timedelta(minutes=30)

# Households with the most real recorded days (45-82 of the 145-day grid),
# verified by counting nonzero-charging days directly against the raw files.
HOUSEHOLDS = [51, 42, 49, 48]


def _read_wide_power_file(path) -> pd.Series:
    """Wide format: rows=dates, 48 columns=half-hour-of-day slots (1=00:00,
    48=23:30). Melts to a proper half-hourly time-indexed Series of Watts."""
    raw = pd.read_csv(path, index_col=0)
    raw.index = pd.to_datetime(raw.index)
    raw.columns = raw.columns.astype(int)

    long = raw.stack()
    long.index = long.index.set_names(["date", "slot"])
    long = long.reset_index()
    long["timestamp"] = long["date"] + (long["slot"] - 1) * HALF_HOUR
    return long.set_index("timestamp")[0].sort_index().rename("power_w")


def _active_window(series: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First/last timestamp with any nonzero reading -- the real recorded
    span hidden inside the fixed 146-day grid (see module docstring, point 1)."""
    nonzero = series[series != 0]
    if nonzero.empty:
        return series.index.min(), series.index.min()
    return nonzero.index.min(), nonzero.index.max()


def load_household(household: int) -> pd.DataFrame:
    lighting = _read_wide_power_file(SONG_KENYA_RAW / f"Power_lighting_household_{household}.csv")
    charging = _read_wide_power_file(SONG_KENYA_RAW / f"Power_charging_household_{household}.csv")
    charging_in = (-charging).rename("charging_w")  # flip to a positive "power into battery"

    df = pd.DataFrame({"load_w": lighting, "charging_w": charging_in})

    start, end = _active_window(charging_in.add(lighting, fill_value=0))
    df = df.loc[start:end].copy()

    # Coulomb-count SOC: net Wh per half-hour, cumulative against capacity,
    # clipped to [0,100]% (the clip is the re-anchor -- see module docstring).
    net_wh = (df["charging_w"] - df["load_w"]) * 0.5
    soc = np.empty(len(df))
    level = 100.0
    for i, delta in enumerate(net_wh.to_numpy()):
        level = min(max(level + delta / BATTERY_CAPACITY_WH * 100.0, 0.0), 100.0)
        soc[i] = level
    df["soc_pct"] = soc

    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    df["Trend_days"] = (df.index - df.index.min()).total_seconds() / 86400.0
    df["unconstrained"] = df["soc_pct"] > UNCONSTRAINED_SOC_THRESHOLD
    return df


def binding_check(df: pd.DataFrame, household: int) -> dict:
    total_hours = len(df) * 0.5
    binding_hours = (~df["unconstrained"]).sum() * 0.5
    binding_rate = 100.0 * binding_hours / total_hours if total_hours else float("nan")
    return {
        "household": household,
        "total_hours": round(total_hours, 1),
        "binding_hours": round(binding_hours, 1),
        "binding_rate_pct": round(binding_rate, 4),
        "min_soc_pct": round(float(df["soc_pct"].min()), 1),
        "mean_soc_pct": round(float(df["soc_pct"].mean()), 1),
        "mean_load_w": round(float(df["load_w"].mean()), 3),
    }


def main() -> None:
    results = []
    for h in HOUSEHOLDS:
        print(f"household {h} ...")
        df = load_household(h)
        result = binding_check(df, h)
        results.append(result)
        print(f"  {result}")
        if result["binding_rate_pct"] > 2.0:
            print("  REAL SUPPRESSION EVIDENCE FOUND")
        out_path = DATA_DIR / f"song_kenya_h{h}_merged.parquet"
        df.to_parquet(out_path)
        print(f"  saved -> {out_path}")

    table = pd.DataFrame(results)
    print()
    print("SONG Kenya binding summary:")
    print(table.to_string(index=False))
    table.to_csv(DATA_DIR / "song_kenya_binding_rates.csv", index=False)
    print(f"\nsaved -> {DATA_DIR / 'song_kenya_binding_rates.csv'}")


if __name__ == "__main__":
    main()
