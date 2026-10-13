"""TC-09 (extension) -- the dashboard must not contradict the dissertation.

The dashboard reads scored_*.parquet written by score_sites.py, while every
table in Chapter 6 is built from the *_e3_results.csv / placebo CSVs written by
the training scripts. Those are two independent paths to the same quantity: the
training scripts score their full record in memory, score_sites.py rescores it
from the pickled models. If the two ever diverge, the dashboard would be
showing a reader numbers that disagree with the thesis it accompanies.

This is a reproducibility check in the spirit of TC-09: the same models applied
to the same record must give the same answer twice.
"""

from __future__ import annotations

import pandas as pd
import pytest

from paths import DATA_DIR

# stem of the scored parquet -> (results CSV, unit column, unit value, suppression column)
CASES = [
    ("song_kenya_h51", "song_kenya_e3_results.csv", "household", 51, "mean_suppression_w_all"),
    ("song_kenya_h42", "song_kenya_e3_results.csv", "household", 42, "mean_suppression_w_all"),
    ("song_kenya_h49", "song_kenya_e3_results.csv", "household", 49, "mean_suppression_w_all"),
    ("song_kenya_h48", "song_kenya_e3_results.csv", "household", 48, "mean_suppression_w_all"),
    ("star_a", "star_e3_results.csv", "site", "A", "mean_suppression_w"),
    ("star_b", "star_e3_results.csv", "site", "B", "mean_suppression_w"),
    ("star_c", "star_e3_results.csv", "site", "C", "mean_suppression_w"),
    ("star_d", "star_e3_results.csv", "site", "D", "mean_suppression_w"),
    ("jharkhand_house_2", "jharkhand_e3_results.csv", "house", "House_2", "mean_suppression_w"),
    ("jharkhand_house_3", "jharkhand_e3_results.csv", "house", "House_3", "mean_suppression_w"),
    ("jharkhand_house_4", "jharkhand_e3_results.csv", "house", "House_4", "mean_suppression_w"),
]


@pytest.mark.parametrize("stem,csv_name,unit_col,unit_val,supp_col", CASES)
def test_scored_frame_matches_reported_suppression(stem, csv_name, unit_col, unit_val, supp_col):
    scored_path = DATA_DIR / f"scored_{stem}.parquet"
    csv_path = DATA_DIR / csv_name
    if not scored_path.exists() or not csv_path.exists():
        pytest.skip(f"{stem}: artefact missing -- run score_sites.py and the training scripts")

    scored = pd.read_parquet(scored_path)
    reported = pd.read_csv(csv_path)
    row = reported.loc[reported[unit_col].astype(str) == str(unit_val)]
    assert not row.empty, f"{unit_val} absent from {csv_name}"

    raw = str(row.iloc[0][supp_col])
    expected = float(raw)
    actual = float(scored["suppression_w"].mean())

    # The training scripts round before writing, and they do not all round to
    # the same precision (SONG Kenya stores 4 dp, STAR and Jharkhand 2 dp), so
    # the tolerance is derived from what the CSV actually recorded rather than
    # assumed. Anything inside half a stored unit is the same number.
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    tolerance = 0.5 * (10 ** -decimals)

    assert actual == pytest.approx(expected, abs=tolerance), (
        f"{stem}: dashboard would show {actual:.4f} W but the dissertation "
        f"reports {expected} W (tolerance {tolerance})"
    )


@pytest.mark.parametrize("stem", [c[0] for c in CASES])
def test_scored_frame_has_no_quantile_crossing(stem):
    """Q10 must never exceed Q90 after rearrangement -- same invariant as TC-10."""
    path = DATA_DIR / f"scored_{stem}.parquet"
    if not path.exists():
        pytest.skip(f"{stem}: artefact missing -- run score_sites.py")

    scored = pd.read_parquet(path)
    crossings = int((scored["q10_pred"] > scored["q90_pred"]).sum())
    assert crossings == 0, f"{stem}: {crossings} crossed Q10/Q90 intervals in dashboard data"


@pytest.mark.parametrize("stem", [c[0] for c in CASES])
def test_suppression_is_non_negative(stem):
    """Suppression is a positive part by definition; a negative value is a defect."""
    path = DATA_DIR / f"scored_{stem}.parquet"
    if not path.exists():
        pytest.skip(f"{stem}: artefact missing -- run score_sites.py")

    scored = pd.read_parquet(path)
    assert float(scored["suppression_w"].min()) >= 0.0
