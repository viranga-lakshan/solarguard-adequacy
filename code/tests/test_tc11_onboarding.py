"""TC-11: Household onboarding, vacancy guard, multi-tier alerts and
monitor_household. All fixtures are synthetic and write only to tmp_path, so
the shipped registry, models and results are never touched.
"""

import json
import pickle

import numpy as np
import pandas as pd
import pytest

import onboarding
from monitor import monitor_household
from paths import CODE_DIR, DATA_DIR

REGISTRY = DATA_DIR / "household_registry.json"


def make_telemetry(days=45, start="2024-01-01", seed=0, weather=False, soc=False):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    hour = idx.hour.to_numpy()
    profile = np.where(hour < 6, 100.0, np.where(hour < 17, 300.0, 700.0))
    df = pd.DataFrame(
        {"timestamp": idx, "load_w": profile + rng.normal(0, 20, len(idx))}
    )
    if soc:
        df["soc_pct"] = 80 + rng.normal(0, 3, len(idx))
    else:
        df["voltage_v"] = 51 + rng.normal(0, 0.5, len(idx))
    if weather:
        df["irradiance_w_m2"] = np.clip(800 * np.sin(np.pi * (hour - 6) / 12), 0, None)
        df["temperature_c"] = 25 + 5 * np.sin(np.pi * (hour - 9) / 12)
    return df


@pytest.fixture
def env(tmp_path):
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"schema_version": 1, "households": {}}))
    return reg, tmp_path / "models"


def onboard(env, hid="new_hh", df=None, **kw):
    reg, models = env
    return onboarding.onboard_household(
        hid, make_telemetry() if df is None else df, registry_path=reg, models_dir=models, **kw
    )


# --- registry ---------------------------------------------------------------
def test_registry_indexes_all_15_pretrained_units():
    if not REGISTRY.exists():
        pytest.skip("run build_registry.py first")
    households = json.loads(REGISTRY.read_text())["households"]
    assert len(households) >= 15
    bench = {k: v for k, v in households.items() if v["origin"] == "pretrained_benchmark"}
    assert len(bench) == 15
    assert {v["feature_tier"] for v in bench.values()} == {"weather_aware", "telemetry_only"}
    assert sum(v["feature_tier"] == "weather_aware" for v in bench.values()) == 3  # Mafate only
    for v in bench.values():
        assert set(v["model_paths"]) == {"q10", "q50", "q90", "qtau"}
        for p in v["model_paths"].values():
            assert (CODE_DIR / p).exists()


# --- gatekeeper -------------------------------------------------------------
def test_good_household_is_onboarded_and_registered(env):
    res = onboard(env)
    assert res.onboarded and res.status == "ONBOARDED"
    reg, models = env
    entry = json.loads(reg.read_text())["households"]["new_hh"]
    assert entry["feature_tier"] == "telemetry_only"
    assert entry["feature_cols"] == onboarding.LEAN_FEATURES
    for q in ("q10", "q50", "q90", "qtau"):
        with open(models / f"new_hh_{q}.pkl", "rb") as fh:
            pickle.load(fh)


def test_weather_columns_select_weather_aware_tier(env):
    res = onboard(env, df=make_telemetry(weather=True))
    assert res.onboarded
    entry = json.loads(env[0].read_text())["households"]["new_hh"]
    assert entry["feature_tier"] == "weather_aware"
    assert entry["feature_cols"] == onboarding.WEATHER_FEATURES


def test_soc_only_telemetry_is_accepted(env):
    assert onboard(env, df=make_telemetry(soc=True)).onboarded


def test_missing_load_column_is_rejected(env):
    res = onboard(env, df=make_telemetry().drop(columns="load_w"))
    assert not res.onboarded and "load_w" in res.message


def test_too_few_days_is_rejected(env):
    res = onboard(env, df=make_telemetry(days=20))
    assert res.status == "INSUFFICIENT_BASELINE"
    assert not env[1].exists()


def test_incomplete_days_do_not_count_as_valid(env):
    df = make_telemetry(days=40)
    df = df[df["timestamp"].dt.hour % 2 == 0]  # 50% packet loss -> every day incomplete
    assert not onboard(env, df=df).onboarded


def test_gsm_dropout_below_20_percent_is_tolerated(env):
    df = make_telemetry(days=45)
    df = df[df["timestamp"].dt.hour != 3]  # 1/24 missing per day
    assert onboard(env, df=df).onboarded


def test_commissioning_defect_is_rejected(env):
    df = make_telemetry(days=45)
    mask = (df["timestamp"].dt.hour == 19) & (df["timestamp"].dt.day <= 10)
    df.loc[mask, "voltage_v"] = 45.0  # LVD cutoff in the evening on 10 days
    res = onboard(env, df=df)
    assert res.status == "COMMISSIONING_DEFECT"
    assert str(res) == "STATUS: COMMISSIONING_DEFECT (Battery already degraded upon installation)"
    assert json.loads(env[0].read_text())["households"] == {}


def test_single_glitch_is_not_a_defect(env):
    df = make_telemetry(days=45)
    df.loc[df["timestamp"] == "2024-01-05 19:00", "voltage_v"] = 45.0
    assert onboard(env, df=df).onboarded


def test_duplicate_id_never_overwrites(env):
    assert onboard(env).onboarded
    reg_before = env[0].read_text()
    assert not onboard(env).onboarded
    assert env[0].read_text() == reg_before


def test_bad_household_id_rejected(env):
    assert not onboard(env, hid="../evil").onboarded


def test_dry_run_writes_nothing(env):
    assert onboard(env, dry_run=True).onboarded
    assert not env[1].exists()
    assert json.loads(env[0].read_text())["households"] == {}


# --- vacancy guard ------------------------------------------------------------
def test_vacancy_needs_24h_flatline_at_float():
    idx = pd.date_range("2024-01-01", periods=100, freq="h")
    load = pd.Series(300.0, index=idx)
    batt = pd.Series(54.0, index=idx)
    load.iloc[10:34] = 1.0   # exactly 24 h
    load.iloc[50:73] = 1.0   # 23 h -> not vacancy
    vac = onboarding.detect_vacancy(load, batt, 53.0)
    assert vac.iloc[10:34].all() and vac.sum() == 24


def test_flatline_with_low_battery_is_not_vacancy():
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    vac = onboarding.detect_vacancy(pd.Series(0.0, index=idx), pd.Series(48.0, index=idx), 53.0)
    assert not vac.any()


def test_data_gap_is_not_vacancy():
    idx = pd.date_range("2024-01-01", periods=47, freq="h")
    load = pd.Series(0.0, index=idx)
    load.iloc[23] = np.nan  # 23 h either side of the gap
    assert not onboarding.detect_vacancy(load, pd.Series(54.0, index=idx), 53.0).any()


# --- alert tiers ----------------------------------------------------------------
def _tiers(supp_level, days=8, batt=51.0, vacant_days=()):
    idx = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    vac = pd.Series(idx.normalize().isin(pd.to_datetime(list(vacant_days))), index=idx)
    return onboarding.compute_alerts(
        pd.Series(float(supp_level), index=idx), pd.Series(batt, index=idx), vac, 47.0
    )


def test_normal_below_50w():
    assert (_tiers(30)["alert_tier"] == "NORMAL").all()


def test_advisory_needs_three_days():
    out = _tiers(70)
    assert out.loc["2024-01-02", "alert_tier"].iloc[0] == "NORMAL"
    assert out.loc["2024-01-08", "alert_tier"].iloc[0] == "ADVISORY"


def test_critical_requires_normal_voltage():
    assert _tiers(150).loc["2024-01-08", "alert_tier"].iloc[0] == "CRITICAL"
    low_v = _tiers(150, batt=44.0)
    assert "CRITICAL" not in set(low_v["alert_tier"]) | set(low_v["instant_tier"])
    assert "VOLTAGE_ALARM" in set(low_v["instant_tier"])


def test_vacant_days_pause_alerts_and_reset_streak():
    out = _tiers(150, days=10, vacant_days=["2024-01-04"])
    assert out.loc["2024-01-04", "alert_tier"].iloc[0] == "VACANT/INACTIVE"
    # 3 fresh days needed after the vacancy: days 5,6,7 -> fires on 7, not before
    assert out.loc["2024-01-06", "alert_tier"].iloc[0] != "CRITICAL"
    assert out.loc["2024-01-08", "alert_tier"].iloc[0] == "CRITICAL"


# --- monitor_household --------------------------------------------------------------
def test_monitor_flags_silent_rationing_on_onboarded_household(env):
    onboard(env)
    reg, _ = env
    new = make_telemetry(days=14, start="2024-02-20", seed=1)
    fade = new["timestamp"] >= "2024-02-25"
    new.loc[fade, "load_w"] *= 0.4  # household silently rations; voltage stays normal
    res = monitor_household("new_hh", new, registry_path=reg)
    assert {"counterfactual_w", "suppression_w", "alert_tier", "instant_tier"} <= set(res.columns)
    assert res.loc[:"2024-02-24", "alert_tier"].eq("NORMAL").all()
    assert res["alert_tier"].iloc[-1] == "CRITICAL"
    assert (res["q10_w"] <= res["q90_w"]).all()


def test_monitor_no_false_alarm_on_unchanged_household(env):
    onboard(env)
    res = monitor_household("new_hh", make_telemetry(days=14, start="2024-02-20", seed=2), registry_path=env[0])
    assert res["alert_tier"].eq("NORMAL").all()


def test_monitor_vacation_not_flagged_as_deprivation(env):
    onboard(env)
    new = make_telemetry(days=14, start="2024-02-20", seed=3)
    away = (new["timestamp"] >= "2024-02-25") & (new["timestamp"] < "2024-03-04")
    new.loc[away, "load_w"] = 0.0
    new.loc[away, "voltage_v"] = 56.0
    res = monitor_household("new_hh", new, registry_path=env[0])
    assert "CRITICAL" not in set(res["alert_tier"])
    assert res.loc["2024-02-27", "alert_tier"].iloc[0] == "VACANT/INACTIVE"


def test_monitor_unknown_and_wrong_schema_raise(env):
    onboard(env)
    with pytest.raises(KeyError):
        monitor_household("nope", make_telemetry(), registry_path=env[0])
    with pytest.raises(ValueError):
        monitor_household("new_hh", make_telemetry(soc=True), registry_path=env[0])


def test_monitor_weather_aware_unit_requires_weather(env):
    onboard(env, df=make_telemetry(weather=True))
    with pytest.raises(ValueError):
        monitor_household("new_hh", make_telemetry(days=5), registry_path=env[0])


def test_monitor_runs_on_pretrained_benchmark_unit():
    if not REGISTRY.exists():
        pytest.skip("run build_registry.py first")
    res = monitor_household("star_a", make_telemetry(days=10, soc=True).assign(load_w=lambda d: d.load_w / 50))
    assert "alert_tier" in res.columns and res["suppression_w"].notna().any()
