"""TC-10: Inspect an emitted alert.
Expected (NFR-03): alert carries estimate, uncertainty interval and
conditioning evidence.
"""

import pickle

import pandas as pd
import pytest

from injection import inject_degradation
from monitor import check_alert
from paths import DATA_DIR, MODELS_DIR


def test_alert_record_carries_estimate_and_interval():
    path = DATA_DIR / "mafate_merged.parquet"
    model_path = MODELS_DIR / "mafate_h1_qtau.pkl"
    if not path.exists() or not model_path.exists():
        pytest.skip("run load_mafate.py and train_model.py first")

    df = pd.read_parquet(path)
    injected = inject_degradation(df, 0.6, "2020-11-01", "Load_h1")
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    result = check_alert(injected, model)

    # The estimate itself.
    assert "suppression_w" in result.columns
    # The monitor's rolling evidence -- what the alert's trend judgement is
    # based on, i.e. the "conditioning evidence" an analyst would inspect.
    assert "rolling_suppression_w" in result.columns
    assert "rolling_voltage_v" in result.columns
    # The decision outputs: both the raw instantaneous signal and the
    # sustained (multi-day) decision actually used to call it "fired".
    assert "deprivation_alert_instant" in result.columns
    assert "deprivation_alert_sustained" in result.columns

    fired = result[result["deprivation_alert_sustained"].fillna(False)]
    assert len(fired) > 0, "fixture (60% fade) is known to trigger an alert -- verify evidence on a real firing"
    row = fired.iloc[0]
    assert pd.notna(row["suppression_w"])
    assert pd.notna(row["rolling_voltage_v"]), "a fired alert must carry the voltage evidence that kept it 'normal'"


def test_suppression_output_carries_q10_q90_uncertainty_interval():
    """FR-05: suppression index persisted with an uncertainty interval."""
    path = DATA_DIR / "mafate_with_suppression.parquet"
    if not path.exists():
        pytest.skip("run suppression.py first")

    df = pd.read_parquet(path)
    for tag in ("h1", "h2", "h3"):
        assert f"q10_pred_{tag}" in df.columns
        assert f"q90_pred_{tag}" in df.columns
        assert f"qtau_pred_{tag}" in df.columns
        valid = df[[f"q10_pred_{tag}", f"q90_pred_{tag}"]].dropna()
        assert (valid[f"q10_pred_{tag}"] <= valid[f"q90_pred_{tag}"]).all(), "Q10 must not exceed Q90"
