"""TC-03: Feed a record containing sentinel values and out-of-range state of
charge. Expected: sentinels removed; run completes without failure (NFR-02).

Tested against the real raw files known to contain sentinels (verified
during development), rather than synthetic data, so this also stands as a
regression test for the specific cleaning rules each loader applies.
"""

import pandas as pd
import pytest


def test_star_drops_epoch_sentinel_timestamps_and_over_range_soc():
    from load_star import load_site_battery
    # Site A's raw Battery CSVs carry ~1,600 Unix-epoch (1969/1970) sentinel
    # rows paired with Voltage=0, and some SOC readings parse to >100%.
    df = load_site_battery("A")
    assert not df.empty
    assert df.index.min().year >= 2010, "epoch-sentinel rows must be dropped, not leak into the series"
    assert (df["soc_pct"] <= 100.5).all(), "over-range SOC sentinels must be dropped"
    assert (df["voltage_v"] >= 5.0).all(), "zero/implausible voltage sentinels must be dropped"


def test_jharkhand_drops_implausible_battery_voltage():
    from jharkhand_check import PLAUSIBLE_BATTERY_V_FLOOR, binding_check, load_house
    # House_4's raw file is known (verified during development) to contain
    # an implausible 0.12 V reading -- a sensor dropout, not real data for a
    # connected 12 V lead-acid pack.
    df = load_house("House_4_January_2016.csv")
    result = binding_check(df, "House_4")
    assert result["dropped_sentinel_readings"] > 0, "known sentinel voltage reading should have been dropped"
    assert result["min_battery_v"] >= PLAUSIBLE_BATTERY_V_FLOOR


def test_loaders_do_not_raise_on_malformed_rows():
    # NFR-02: tolerate gaps/sentinels without failing. A loader raising here
    # would fail the test outright.
    from load_star import load_site_battery
    from jharkhand_check import load_house

    for site in ["A", "B", "C", "D"]:
        load_site_battery(site)  # must not raise
    for house in ["House_2_January_2016.csv", "House_3_January_2016.csv", "House_4_January_2016.csv"]:
        df = load_house(house)
        assert isinstance(df, pd.DataFrame)
