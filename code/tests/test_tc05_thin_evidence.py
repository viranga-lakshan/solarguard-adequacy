"""TC-05: Fit the estimator where matched evidence is deliberately thin.
Expected: low-confidence result returned; no alert raised. (Algorithm 1,
Sec. 5.2.3, step 6-7: "if |U| < minimum sample per condition stratum then
return low-confidence result".)
"""

import numpy as np
import pandas as pd


def _tiny_synthetic_frame(n: int, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(0)
    data = {
        "Hour": idx.hour, "DayOfWeek": idx.dayofweek,
        "Month_sin": np.sin(2 * np.pi * idx.month / 12), "Month_cos": np.cos(2 * np.pi * idx.month / 12),
        "Trend_days": np.arange(n) / 24.0,
        target_col: rng.uniform(0, 10, size=n),
    }
    return pd.DataFrame(data, index=idx)[feature_cols + [target_col]]


def test_star_declines_to_fit_on_thin_evidence(monkeypatch):
    import train_star

    tiny = _tiny_synthetic_frame(50, train_star.FEATURE_COLS, train_star.TARGET_COL)
    tiny["unconstrained"] = True
    monkeypatch.setattr(train_star, "load_site_merged", lambda site: tiny)

    result = train_star.run_site("A")
    assert result["status"] != "fitted", "thin evidence must not produce a confident fit"
    assert "low_confidence" in result["status"]


def test_jharkhand_declines_to_fit_on_thin_evidence(monkeypatch):
    import train_jharkhand

    tiny = _tiny_synthetic_frame(50, train_jharkhand.FEATURE_COLS, train_jharkhand.TARGET_COL)
    tiny["unconstrained"] = True

    monkeypatch.setattr(train_jharkhand, "prepare_house", lambda filename: tiny)
    result = train_jharkhand.run_house("House_2_January_2016.csv")
    assert result["status"] != "fitted"
    assert "low_confidence" in result["status"]
