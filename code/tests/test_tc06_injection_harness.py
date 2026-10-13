"""TC-06: Apply a known fade profile through the injection harness.
Expected: output trace exhibits the imposed capability reduction.
"""

import numpy as np
import pandas as pd

from injection import inject_degradation


def test_injection_caps_load_only_after_start_date_and_leaves_history_untouched():
    idx = pd.date_range("2020-01-01", periods=1000, freq="1min")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Load_h1": rng.uniform(50, 150, size=len(idx))}, index=idx)

    start = idx[500]
    fade = 0.5
    injected = inject_degradation(df, fade, str(start), "Load_h1")

    before = injected.loc[: start - pd.Timedelta(minutes=1), "Load_h1"]
    assert (before == df.loc[before.index, "Load_h1"]).all(), "history before the injection date must be untouched"

    baseline = df.loc[: start - pd.Timedelta(minutes=1), "Load_h1"].quantile(0.75)
    max_avail = baseline * (1 - fade)
    after = injected.loc[start:, "Load_h1"]
    assert (after <= max_avail + 1e-9).all(), "every post-injection reading must respect the capped ceiling"

    # The imposed reduction must actually bind somewhere -- otherwise the
    # test fixture's fade level was too mild to prove anything.
    original_after = df.loc[start:, "Load_h1"]
    assert (original_after > max_avail).any(), "fixture should include values the cap actually clips"
    assert (after < original_after).any(), "injection must visibly reduce at least some post-injection load"


def test_zero_fade_caps_at_the_undegraded_baseline_not_a_true_no_op():
    """fade=0.0 means 'healthy', not 'untouched': per injection.py's own
    docstring, max_available = baseline * (1 - fade), so even at fade=0 the
    post-injection series is capped at the household's own 75th-percentile
    baseline. Values already below that ceiling are unaffected; values
    above it (by construction, part of the fixture here) are clipped."""
    idx = pd.date_range("2020-01-01", periods=200, freq="1min")
    df = pd.DataFrame({"Load_h1": np.linspace(10, 200, len(idx))}, index=idx)
    start = idx[100]
    injected = inject_degradation(df, 0.0, str(start), "Load_h1")

    baseline = df.loc[: start - pd.Timedelta(minutes=1), "Load_h1"].quantile(0.75)
    after = injected.loc[start:, "Load_h1"]
    assert (after <= baseline + 1e-9).all()
    below_baseline = df.loc[start:, "Load_h1"] <= baseline
    pd.testing.assert_series_equal(after[below_baseline], df.loc[start:, "Load_h1"][below_baseline])
