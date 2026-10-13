"""TC-07: Run all baseline detectors through the shared interface.
Expected: all produce comparable detection timestamps on identical input.

(Fig. 4.3's caption: "Baselines and the suppression monitor share one
interface so that the evaluation compares like with like" -- this is the
property that makes E1's lead-time comparison meaningful at all.)
"""

import pickle

import pytest

from e1_lead_time import detect_cusum, detect_method_a, detect_voltage_alarm
from injection import inject_degradation
from paths import DATA_DIR, MODELS_DIR


@pytest.fixture(scope="module")
def injected_mafate():
    path = DATA_DIR / "mafate_merged.parquet"
    model_path = MODELS_DIR / "mafate_h1_qtau.pkl"
    if not path.exists() or not model_path.exists():
        pytest.skip("run load_mafate.py and train_model.py first")

    import pandas as pd

    df = pd.read_parquet(path)
    injected = inject_degradation(df, 0.6, "2020-11-01", "Load_h1")
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    return injected, model


def _is_day_or_none(value) -> bool:
    return value is None or isinstance(value, (int,))


def test_all_detectors_return_comparable_types(injected_mafate):
    injected, model = injected_mafate

    _, day_a = detect_method_a(injected, model)
    day_b = detect_voltage_alarm(injected)
    day_c = detect_cusum(injected, "2020-11-01")

    for name, day in [("method_a", day_a), ("voltage_alarm", day_b), ("cusum", day_c)]:
        assert _is_day_or_none(day), f"{name} must return an int day-offset or None, got {type(day)}"

    # On a sufficiently severe, real injected fade, the dissertation's own
    # method should detect it (verified separately at 60% fade in E1).
    assert day_a is not None, "method A should detect a 60% fade on this house (matches E1's own result)"
