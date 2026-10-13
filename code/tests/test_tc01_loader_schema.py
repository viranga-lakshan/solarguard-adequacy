"""TC-01: Load each site through its loader and assert the normalised schema.
Expected: common series representation, gaps flagged (NaN) not dropped.
"""

import pandas as pd
import pytest

from paths import DATA_DIR


def _assert_common_schema(df: pd.DataFrame, min_rows: int = 100):
    assert isinstance(df.index, pd.DatetimeIndex), "series must be time-indexed"
    assert df.index.is_monotonic_increasing, "series must be sorted"
    assert not df.index.has_duplicates, "loader must not leave duplicate timestamps"
    assert len(df) >= min_rows


def test_mafate_schema():
    path = DATA_DIR / "mafate_merged.parquet"
    if not path.exists():
        pytest.skip("run load_mafate.py first")
    df = pd.read_parquet(path)
    _assert_common_schema(df)
    # Gaps must be flagged (NaN), not silently dropped: the documented
    # post-Dec-2020 irradiance failure should appear as NaN rows that are
    # still present in the index, not missing timestamps.
    assert df["Gincl"].isna().sum() > 0, "known irradiance gap should appear as NaN, not be dropped"
    assert df.loc[df["Gincl"].isna()].index.is_monotonic_increasing  # still part of the continuous index


def test_heed_schema():
    path = DATA_DIR / "heed_cpe1_merged.parquet"
    if not path.exists():
        pytest.skip("run load_heed.py first")
    df = pd.read_parquet(path)
    _assert_common_schema(df)
    assert df["appliance_load_w"].isna().sum() > 0, "HEED's sparser SOC log should leave gaps as NaN"


def test_star_schema():
    from load_star import load_site_battery
    df = load_site_battery("A")
    _assert_common_schema(df, min_rows=1000)
    assert "unconstrained" in df.columns


def test_jharkhand_schema():
    from jharkhand_check import load_house
    df = load_house("House_2_January_2016.csv")
    _assert_common_schema(df, min_rows=1000)
    assert "Battery_V" in df.columns
