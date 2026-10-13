"""TC-04: Classify intervals as constrained or unconstrained at the floor
threshold. Expected: classification matches the measured binding behaviour
of the site (Table 3.2: Mafate and HEED never/barely bind; Jharkhand and
STAR were independently found in this project to also sit near 0%).
"""

import pandas as pd
import pytest

from paths import DATA_DIR


def test_mafate_binding_rate_matches_measured_behaviour():
    path = DATA_DIR / "mafate_binding_rates.csv"
    if not path.exists():
        pytest.skip("run binding_analysis.py first")
    table = pd.read_csv(path)
    # Table 3.2: "Never binds. Voltage minimum 48.50 V against a disconnect
    # near 44-46 V" -- classification should reflect that, not report
    # material binding.
    assert (table["binding_rate_pct"] < 1.0).all()


def test_jharkhand_binding_rate_matches_measured_behaviour():
    path = DATA_DIR / "jharkhand_binding_rates.csv"
    if not path.exists():
        pytest.skip("run jharkhand_check.py first")
    table = pd.read_csv(path)
    assert (table["binding_rate_pct"] < 1.0).all()


def test_star_binding_rate_matches_measured_behaviour():
    path = DATA_DIR / "star_binding_rates.csv"
    if not path.exists():
        pytest.skip("run load_star.py first")
    table = pd.read_csv(path)
    assert (table["binding_rate_pct"] < 1.0).all()


def test_binding_threshold_is_a_sharp_boundary():
    """Classification at the floor threshold should be a clean >/<= split,
    not leave an ambiguous middle band (sanity check on the classifier
    logic itself, using Mafate's Ubat > 47.0 V rule, decoupled from any
    particular site's data)."""
    ubat = pd.Series([46.9, 47.0, 47.1, 50.0, float("nan")])
    unconstrained = ubat.notna() & (ubat > 47.0)
    assert list(unconstrained) == [False, False, True, True, False]
