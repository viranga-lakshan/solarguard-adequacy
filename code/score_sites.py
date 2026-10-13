"""Persist scored frames for every non-Mafate site, for the dashboard (FR-09).

The training scripts already score their full records in memory -- they use it
for the per-site plots and the summary statistics -- but only Mafate persisted
the result. This script re-runs that same scoring step and writes it to disk so
the dashboard can read it, keeping the monitoring tier a pure reader of
artefacts rather than something that refits models at page load (Sec. 4.9).

Nothing here changes the methodology. Each site is rebuilt through its OWN
loader/prepare function -- the identical one its training script used -- and
scored with the pickled models that training script fitted. The suppression
definition is the same clip(qtau_pred - observed, 0) used everywhere else, and
the Q10/Q90 pair goes through the same monotone rearrangement.

Mafate is deliberately excluded: it already has mafate_with_suppression.parquet
with per-house columns and extra weather channels, and the dashboard keeps its
richer Mafate view (which also drives the injection demonstration).
"""

from __future__ import annotations

import pickle

import pandas as pd

from paths import DATA_DIR, MODELS_DIR
from train_model import rearrange_quantile_pair

FEATURE_COLS = ["Hour", "DayOfWeek", "Month_sin", "Month_cos", "Trend_days"]


def _build_heed() -> list[tuple[str, str, pd.DataFrame, str, str | None]]:
    from train_heed import TARGET_COL, prepare_frame

    raw = pd.read_parquet(DATA_DIR / "heed_cpe1_merged.parquet")
    _, full_df = prepare_frame(raw)
    return [("HEED Hall", "CPE1", full_df, TARGET_COL, "soc_pct")]


def _build_star() -> list[tuple[str, str, pd.DataFrame, str, str | None]]:
    from load_star import load_site_merged
    from train_star import SITES, TARGET_COL

    out = []
    for site in SITES:
        out.append(("STAR", site, load_site_merged(site), TARGET_COL, "soc_pct"))
    return out


def _build_jharkhand() -> list[tuple[str, str, pd.DataFrame, str, str | None]]:
    from jharkhand_check import HOUSES
    from train_jharkhand import TARGET_COL, prepare_house

    out = []
    for filename in HOUSES:
        # same tag the training script derived its model filenames from
        unit = filename.split("_January")[0]
        out.append(("Jharkhand", unit, prepare_house(filename), TARGET_COL, None))
    return out


def _build_song() -> list[tuple[str, str, pd.DataFrame, str, str | None]]:
    from load_song_kenya import HOUSEHOLDS, load_household
    from train_song_kenya import TARGET_COL

    out = []
    for h in HOUSEHOLDS:
        out.append(("SONG Kenya", str(h), load_household(h), TARGET_COL, "soc_pct"))
    return out


# unit label -> stem of the pickled models that training wrote for it
MODEL_STEM = {
    ("HEED Hall", "CPE1"): "heed_cpe1",
    ("STAR", "A"): "star_a",
    ("STAR", "B"): "star_b",
    ("STAR", "C"): "star_c",
    ("STAR", "D"): "star_d",
    ("Jharkhand", "House_2"): "jharkhand_house_2",
    ("Jharkhand", "House_3"): "jharkhand_house_3",
    ("Jharkhand", "House_4"): "jharkhand_house_4",
    ("SONG Kenya", "51"): "song_kenya_h51",
    ("SONG Kenya", "42"): "song_kenya_h42",
    ("SONG Kenya", "49"): "song_kenya_h49",
    ("SONG Kenya", "48"): "song_kenya_h48",
}

BUILDERS = [_build_heed, _build_star, _build_jharkhand, _build_song]


def _load_model(stem: str, name: str):
    path = MODELS_DIR / f"{stem}_{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def score_unit(site: str, unit: str, df: pd.DataFrame, target: str, soc_col: str | None):
    stem = MODEL_STEM.get((site, unit))
    if stem is None:
        print(f"  {site}/{unit}: no model stem registered -- skipped")
        return None

    models = {name: _load_model(stem, name) for name in ("q10", "q50", "q90", "qtau")}
    if models["qtau"] is None:
        print(f"  {site}/{unit}: no fitted model on disk -- skipped (site declined at training)")
        return None

    frame = df.dropna(subset=FEATURE_COLS + [target]).copy()
    if frame.empty:
        print(f"  {site}/{unit}: no usable rows -- skipped")
        return None

    X = frame[FEATURE_COLS]
    qtau = models["qtau"].predict(X)
    q10 = models["q10"].predict(X)
    q90 = models["q90"].predict(X)
    q10, q90 = rearrange_quantile_pair(q10, q90)

    out = pd.DataFrame(index=frame.index)
    out["load_w"] = frame[target].to_numpy()
    out["qtau_pred"] = qtau
    out["q10_pred"] = q10
    out["q90_pred"] = q90
    out["suppression_w"] = (out["qtau_pred"] - out["load_w"]).clip(lower=0.0)
    out["soc_pct"] = frame[soc_col].to_numpy() if soc_col and soc_col in frame else pd.NA
    out["unconstrained"] = (
        frame["unconstrained"].to_numpy() if "unconstrained" in frame else True
    )

    path = DATA_DIR / f"scored_{stem}.parquet"
    out.to_parquet(path)
    print(
        f"  {site}/{unit}: {len(out):,} rows, mean suppression "
        f"{out['suppression_w'].mean():.4f} W -> {path.name}"
    )
    return {
        "site": site,
        "unit": unit,
        "stem": stem,
        "n_rows": len(out),
        "mean_suppression_w": round(float(out["suppression_w"].mean()), 4),
        "mean_load_w": round(float(out["load_w"].mean()), 4),
    }


def main() -> None:
    index_rows = []
    for builder in BUILDERS:
        print(f"{builder.__name__} ...")
        for site, unit, df, target, soc_col in builder():
            row = score_unit(site, unit, df, target, soc_col)
            if row is not None:
                index_rows.append(row)

    index = pd.DataFrame(index_rows)
    index.to_csv(DATA_DIR / "scored_sites_index.csv", index=False)
    print()
    print(index.to_string(index=False))
    print(f"\nsaved -> {DATA_DIR / 'scored_sites_index.csv'}")


if __name__ == "__main__":
    main()
