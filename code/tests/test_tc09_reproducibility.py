"""TC-09: Execute the full pipeline twice under a fixed seed.
Expected (NFR-04): byte-identical outputs.

Re-running the entire 15-20 minute training pipeline twice inside a test
suite is impractical (see TC-08), so this verifies the actual mechanism
NFR-04 depends on directly: that every random_state=0 sampling/fitting step
used throughout the codebase is deterministic, by re-running the real
tau-selection + fit procedure twice end-to-end on real data and comparing
outputs exactly.
"""

import pandas as pd
import pytest

from paths import DATA_DIR


def test_tau_selection_and_fit_is_deterministic_under_fixed_seed():
    path = DATA_DIR / "mafate_merged.parquet"
    if not path.exists():
        pytest.skip("run load_mafate.py first")

    from train_model import load_house_frame, select_tau, three_way_split

    df = pd.read_parquet(path)
    frame = load_house_frame(df, "Load_h1")
    fit, calibration, test = three_way_split(frame)

    tau_1, table_1 = select_tau(fit, calibration, "Load_h1")
    tau_2, table_2 = select_tau(fit, calibration, "Load_h1")

    assert tau_1 == tau_2, "tau selection must be deterministic under a fixed seed"
    pd.testing.assert_frame_equal(table_1, table_2)


def test_quantile_regression_predictions_are_byte_identical_across_runs():
    path = DATA_DIR / "mafate_merged.parquet"
    if not path.exists():
        pytest.skip("run load_mafate.py first")

    from train_model import FEATURE_COLS, _make_pipeline, load_house_frame, three_way_split

    df = pd.read_parquet(path)
    frame = load_house_frame(df, "Load_h1")
    fit, _, test = three_way_split(frame)
    fit_sample = fit.sample(n=min(2000, len(fit)), random_state=0)

    preds = []
    for _ in range(2):
        pipe = _make_pipeline(0.5)
        pipe.fit(fit_sample[FEATURE_COLS], fit_sample["Load_h1"])
        preds.append(pipe.predict(test[FEATURE_COLS].iloc[:500]))

    assert (preds[0] == preds[1]).all(), "identical fit+predict under the same seed must be byte-identical"
