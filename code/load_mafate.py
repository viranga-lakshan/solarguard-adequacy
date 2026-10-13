"""Task 1 -- Mafate data loader.

Source: PIMENT Laboratory, "Data in experimental stand-alone microgrid",
Roche-Plate, Cirque de Mafate, Reunion Island (Zenodo 10.5281/zenodo.8186303).

Merges the plant monitor, the three household demand loggers and the
meteorological station onto a common 1-minute grid and engineers the
covariates used by the quantile-regression suppression model (Tasks 3-5).

Column-mapping note: the dataset's own README documents a DIFFERENT "main
house total" input channel per house file -- it is NOT a uniform "Input3"
across files, even though the three files share the same Date,Input1..N
shape. Using the wrong channel silently trains the demand model on a single
appliance (e.g. a freezer) instead of the whole house.

    House 1 (Demand_house_1_dataset.csv): Input4 = Main, all appliances (W)
    House 2 (Demand_house_2_dataset.csv): Input5 = Main, all appliances (W)
    House 3 (Demand_house_3_dataset.csv): Input1 = Main, all appliances (W)

Unit note: Solar_plant_dataset's Pout/Ppv1/Ppv2 are documented in kW, while
every Demand_house_*.csv Input channel is documented in W. PV_total_w below
is converted to W so it is comparable to the load columns. Likewise
Meteo_dataset_'s Gincl/Ghi are documented in kW/m^2 and are converted to the
conventional W/m^2 here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from paths import DATA_DIR, MAFATE_RAW

DATE_FMT = "%Y-%m-%d %H:%M:%S"

# Per-house "main, all appliances" channel -- see module docstring.
HOUSE_MAIN_COLUMN = {
    "Demand_house_1_dataset.csv": "Input4",
    "Demand_house_2_dataset.csv": "Input5",
    "Demand_house_3_dataset.csv": "Input1",
}

# House 2's logger timestamps as "DD/MM/YYYY HH:MM" (no seconds), unlike every
# other Mafate file, which uses the ISO "YYYY-MM-DD HH:MM:SS" format in
# DATE_FMT. Forcing the ISO format onto this file turns every timestamp into
# NaT; since NaT == NaT, the subsequent duplicate-drop then collapses the
# whole file down to a single row. Verified against the raw file directly.
HOUSE_DATE_FMT = {
    "Demand_house_1_dataset.csv": DATE_FMT,
    "Demand_house_2_dataset.csv": "%d/%m/%Y %H:%M",
    "Demand_house_3_dataset.csv": DATE_FMT,
}


def _read_plant(raw_dir) -> pd.DataFrame:
    frame = pd.read_csv(
        raw_dir / "Solar_plant_dataset.csv",
        skiprows=1,
        parse_dates=["Date"],
        date_format=DATE_FMT,
    )
    frame = frame.set_index("Date").sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    return frame[["Pout", "Ppv1", "Ppv2", "Upv1", "Upv2", "Ubat", "Ibat", "Tbat"]]


def _read_house(raw_dir, filename: str, out_col: str) -> pd.Series:
    frame = pd.read_csv(
        raw_dir / filename,
        skiprows=1,
        parse_dates=["Date"],
        date_format=HOUSE_DATE_FMT[filename],
    )
    frame = frame.set_index("Date").sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    main_col = HOUSE_MAIN_COLUMN[filename]
    return frame[main_col].rename(out_col)


def _read_meteo(raw_dir) -> pd.DataFrame:
    # Row 1: placeholder "ColumnN" names. Row 2: lab banner. Row 3: real headers.
    frame = pd.read_csv(
        raw_dir / "Meteo_dataset_.csv",
        sep=";",
        skiprows=2,
        parse_dates=["Date"],
        date_format=DATE_FMT,
    )
    frame = frame.set_index("Date").sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    frame = frame[["Gincl", "Ghi", "Tpv", "Ta", "Rh", "Wd", "Ws", "Press"]]
    # kW/m^2 -> W/m^2 (README units).
    frame["Gincl"] = frame["Gincl"] * 1000.0
    frame["Ghi"] = frame["Ghi"] * 1000.0
    # Native resolution is 20 s; resample to the 1-minute grid used elsewhere.
    return frame.resample("1min").mean()


def build() -> pd.DataFrame:
    plant = _read_plant(MAFATE_RAW)

    load_h1 = _read_house(MAFATE_RAW, "Demand_house_1_dataset.csv", "Load_h1")
    load_h2 = _read_house(MAFATE_RAW, "Demand_house_2_dataset.csv", "Load_h2")
    load_h3 = _read_house(MAFATE_RAW, "Demand_house_3_dataset.csv", "Load_h3")

    meteo = _read_meteo(MAFATE_RAW)

    df = plant.join([load_h1, load_h2, load_h3, meteo], how="left")

    df["PV_total_w"] = (df["Ppv1"] + df["Ppv2"]) * 1000.0

    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"] = df.index.month
    # Calendar-quarter bucket (Q1=0 .. Q4=3). Deliberately NOT labelled as a
    # literal "Spring/Summer/Autumn/Winter" season: Mafate sits at 21 S, so
    # temperate season names would be wrong for this site. Kept for
    # descriptive/table use only -- NOT fed to the quantile model (see
    # Month_sin/Month_cos below).
    df["Season"] = (df["Month"] - 1) // 3
    df["Is_Night"] = (df["Gincl"] < 10.0).astype(int)

    # Month_sin/Month_cos: a smooth, continuous encoding of calendar position,
    # used instead of a one-hot Month/Season in the quantile model. A
    # one-hot Month collapses to an unusable zero-vector for any calendar
    # month absent from the training split -- which happens by construction
    # under Task 3/7's chronological 75/25 (or 9-month/3-month) split, since
    # 12 months of data means the held-out tail necessarily covers months the
    # model never saw. The cyclical encoding degrades gracefully instead.
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)

    # Trend_days: days since the start of the record. Mafate's own documented
    # behaviour is ~3x household demand growth over this single year of data
    # (verified separately: ~102 W -> ~300 W), consistent with the adoption
    # ramp-up phase this dissertation's cited literature describes. Without
    # an explicit trend term a linear model trained on the early (low-usage)
    # months has no way to extrapolate that growth into a later held-out
    # period -- it is not a stylistic addition, it is required for the
    # prescribed chronological train/test split to be learnable at all.
    df["Trend_days"] = (df.index - df.index.min()).total_seconds() / 86400.0

    return df


def main() -> None:
    df = build()
    out_path = DATA_DIR / "mafate_merged.parquet"
    df.to_parquet(out_path)

    print(f"shape: {df.shape}")
    print(f"date range: {df.index.min()} -> {df.index.max()}")
    print(f"columns: {list(df.columns)}")
    print()
    print("sample rows:")
    print(df.sample(5, random_state=0).to_string())
    print()
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
