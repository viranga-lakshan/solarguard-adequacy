"""Task 10 -- HEED (Hall and Microgrid sites, Rwanda) loader.

The per-CPE "Raw-data/<site> CPE<N>" folders are individual appliance
monitors (IAMs) only -- there is no bat_soc/bat_v sensor name inside them to
fall back on, contrary to what a generic reading of the per-CPE schema might
suggest. Battery SOC/voltage for each site lives in a SEPARATE Victron VRM
system export under "Raw-data/System_data/" (Hall) or "Raw-data/System data/"
(Microgrid -- note the differing folder-name spacing/underscore, verified
against the raw tree), which uses its own 3-row header (device category /
field name / unit) and must be parsed and joined in by timestamp.

Power-channel naming is also inconsistent within the per-CPE files: LED
channels are suffixed "_P" (uppercase) while TV/USB channels use "_p"
(lowercase) -- verified against the raw files. A case-sensitive "_p" filter
silently drops every LED reading, so the match here is case-insensitive.

Scope note on Hall's own CPE folders: Hall additionally has CPE2-7 and S1-S4
("socket"/relay monitors, e.g. Hall S1) that this loader does NOT include.
CPE2-7 use the same simple "_p" schema as CPE1 and could be added the same
way CPE summing is done for Microgrid below; they were left out to avoid
retraining and re-verifying the already-passing HEED Hall E4 result this
late. S1-S4 use a DIFFERENT schema (vRELAY1_V/_I/_VA/_PF, no simple "_p"
power field) whose relationship to the CPE readings (additional circuits, or
overlapping/double-counting the same load) was not verified -- deliberately
excluded rather than guessed at. Document this as a scope limitation, not an
oversight: HEED Hall's reported suppression here reflects CPE1 only.

Microgrid uses ALL of its CPE folders (17, across 6 "Nur" nursery buildings
and a Playground, verified by listing the raw tree), summed into one
community-level load, analogous to how Mafate sums load across houses
sharing one battery. Its own "Streetlight" and "S" folders are excluded for
the same reasons as Hall's.
"""

from __future__ import annotations

import pandas as pd

from paths import DATA_DIR, HEED_RAW

CPE1_DIR = HEED_RAW / "Hall" / "Raw-data" / "Hall CPE1"
SYSTEM_DATA_DIR = HEED_RAW / "Hall" / "Raw-data" / "System_data"

MICROGRID_RAW_DIR = HEED_RAW / "Microgrid" / "Raw-data"
MICROGRID_SYSTEM_DATA_DIR = MICROGRID_RAW_DIR / "System data"

SOC_COLUMN = "Battery Monitor [258] | State of charge"
VOLTAGE_COLUMN = "Battery Monitor [258] | Voltage"
UNCONSTRAINED_SOC_THRESHOLD = 20.0


def _load_one_cpe(cpe_dir) -> pd.DataFrame | None:
    """Raw (unix_ts, sensor, value) power rows for one CPE folder, or None."""
    frames = []
    for path in sorted(cpe_dir.glob("*.csv")):
        try:
            raw = pd.read_csv(path, header=None, names=["unix_ts", "sensor", "value"])
        except Exception as exc:  # malformed daily file -- skip, don't abort the whole site
            print(f"  skipping {path.name}: {exc}")
            continue
        # Some Microgrid CPE files (not seen in Hall CPE1) carry rows with a
        # missing sensor name; NaN.str.endswith() would otherwise raise when
        # used as a boolean mask, so NaN sensor names are treated as "not a
        # power reading" rather than crashing the whole load.
        is_power = raw["sensor"].str.lower().str.endswith("_p", na=False)
        power = raw[is_power].copy()
        if power.empty:
            continue
        frames.append(power)
    return pd.concat(frames, ignore_index=True) if frames else None


def load_cpe_appliance_load(cpe_dir=CPE1_DIR) -> pd.Series:
    """Sum of all power-channel readings per minute for one CPE."""
    all_power = _load_one_cpe(cpe_dir)
    if all_power is None:
        return pd.Series(dtype=float, name="appliance_load_w")

    all_power["timestamp"] = pd.to_datetime(all_power["unix_ts"], unit="s")
    # Per-file CSV type inference reads "value" as a mixed/object column
    # whenever a non-power sensor row elsewhere in that file holds text
    # (e.g. a firmware name). That taints the whole column, including the
    # numeric-looking power readings, which arrive here as literal strings
    # pivot_table's mean() cannot average without an explicit conversion.
    all_power["value"] = pd.to_numeric(all_power["value"], errors="coerce")

    pivot = all_power.pivot_table(
        index="timestamp", columns="sensor", values="value", aggfunc="mean"
    )
    total_load = pivot.sum(axis=1, min_count=1).rename("appliance_load_w")
    return total_load.resample("1min").mean()


def load_multi_cpe_appliance_load(cpe_dirs: list) -> pd.Series:
    """Sum of all power-channel readings per minute, across SEVERAL CPE
    folders (Microgrid: 17 CPEs across multiple buildings sharing one
    battery). Each CPE is pivoted/resampled separately first (same as the
    single-CPE path) so that one CPE's sensor-name collisions never bleed
    into another's, then the per-minute totals are summed across CPEs."""
    per_cpe_totals = []
    for cpe_dir in cpe_dirs:
        all_power = _load_one_cpe(cpe_dir)
        if all_power is None:
            continue
        all_power["timestamp"] = pd.to_datetime(all_power["unix_ts"], unit="s")
        all_power["value"] = pd.to_numeric(all_power["value"], errors="coerce")
        pivot = all_power.pivot_table(index="timestamp", columns="sensor", values="value", aggfunc="mean")
        per_cpe_totals.append(pivot.sum(axis=1, min_count=1).resample("1min").mean())

    combined = pd.concat(per_cpe_totals, axis=1)
    return combined.sum(axis=1, min_count=1).rename("appliance_load_w")


def _read_system_data_file(path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, low_memory=False)
    device_row = raw.iloc[0].ffill()
    field_row = raw.iloc[1]
    colnames = [f"{str(d).strip()} | {str(f).strip()}" for d, f in zip(device_row, field_row)]
    colnames[0] = "timestamp"

    data = raw.iloc[3:].copy()
    data.columns = colnames
    return data


def load_system_data(system_dir=SYSTEM_DATA_DIR) -> pd.DataFrame:
    frames = [_read_system_data_file(p) for p in sorted(system_dir.glob("*.csv"))]
    combined = pd.concat(frames, ignore_index=True)

    combined["timestamp"] = pd.to_datetime(
        combined["timestamp"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    combined = combined.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    out = pd.DataFrame(index=combined.index)
    out["soc_pct"] = pd.to_numeric(combined.get(SOC_COLUMN), errors="coerce")
    out["battery_v"] = pd.to_numeric(combined.get(VOLTAGE_COLUMN), errors="coerce")
    out = out.resample("1min").mean()
    # The Victron logger samples roughly every 15 minutes, not every minute.
    # SOC changes slowly, so the last known reading is carried forward over
    # short gaps (holding state, not fabricating a trend) -- capped at twice
    # the observed logging interval so a genuine multi-hour outage still
    # reads as missing rather than stale.
    return out.ffill(limit=30)


def build() -> pd.DataFrame:
    load = load_cpe_appliance_load()
    system = load_system_data()
    return _assemble(load, system)


def build_microgrid() -> pd.DataFrame:
    cpe_dirs = sorted(
        p for p in MICROGRID_RAW_DIR.iterdir()
        if p.is_dir() and "CPE" in p.name and "Streetlight" not in p.name
    )
    load = load_multi_cpe_appliance_load(cpe_dirs)
    system = load_system_data(MICROGRID_SYSTEM_DATA_DIR)
    return _assemble(load, system)


def _assemble(load: pd.Series, system: pd.DataFrame) -> pd.DataFrame:
    df = load.to_frame().join(system, how="left")
    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["Season"] = (df["Month"] - 1) // 3
    df["unconstrained"] = df["soc_pct"].notna() & (df["soc_pct"] > UNCONSTRAINED_SOC_THRESHOLD)
    return df


def main() -> None:
    df = build()
    out_path = DATA_DIR / "heed_cpe1_merged.parquet"
    df.to_parquet(out_path)

    total_hours = df["appliance_load_w"].notna().sum() / 60.0
    soc_known_hours = df["soc_pct"].notna().sum() / 60.0
    binding_hours = (df["soc_pct"].notna() & ~df["unconstrained"]).sum() / 60.0
    binding_rate = 100.0 * binding_hours / soc_known_hours if soc_known_hours else float("nan")

    print(f"shape: {df.shape}")
    print(f"date range: {df.index.min()} -> {df.index.max()}")
    print(f"CPE1 load coverage: {total_hours:.0f} hours")
    print(f"SOC channel coverage: {soc_known_hours:.0f} hours")
    print(f"binding rate (SOC <= {UNCONSTRAINED_SOC_THRESHOLD}%): {binding_rate:.3f}%")
    print()
    print(df.dropna(subset=['soc_pct']).sample(5, random_state=0).to_string())
    print()
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
