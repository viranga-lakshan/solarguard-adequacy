"""TC-02: Reconstruct state of charge on Mafate and compare against the
battery voltage record. Expected: reconstruction consistent with observed
voltage behaviour.
"""

import pandas as pd
import pytest

from paths import DATA_DIR
from soc_reconstruction import reconstruct_soc


def test_soc_reconstruction_consistent_with_voltage():
    path = DATA_DIR / "mafate_merged.parquet"
    if not path.exists():
        pytest.skip("run load_mafate.py first")

    df = pd.read_parquet(path).dropna(subset=["Ubat", "Ibat"])
    # A representative slice, not the full 474k rows: this test checks
    # correctness of the reconstruction logic, not pipeline performance
    # (that is TC-08's job).
    sample = df.iloc[:60_000]

    soc, anchors = reconstruct_soc(sample)
    assert anchors > 0, "expected at least one detected full-charge re-anchor point"
    assert (soc >= 0).all() and (soc <= 100).all(), "SOC must stay within [0, 100]%"

    corr = soc.corr(sample["Ubat"])
    assert corr > 0.3, f"reconstructed SOC should track battery voltage positively, got corr={corr:.3f}"

    # The known finding this dissertation reports (min voltage 48.50 V, i.e.
    # never near the disconnect threshold) should correspond to a
    # reconstruction that never reads a low SOC while voltage is clearly high.
    drift_flag = ((soc < 50) & (sample["Ubat"] > 50)).mean()
    assert drift_flag < 0.01, "reconstruction should not read low SOC while voltage is comfortably high"
