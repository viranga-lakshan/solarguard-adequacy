"""TC-12: the dashboard builds its site/unit dropdown from household_registry.json,
keeps every existing site working, and live-scores newly onboarded households.
Uses streamlit's headless AppTest; registry/telemetry paths are redirected to
tmp_path so the shipped registry and data/ are never written.
"""

import json

import pytest
from streamlit.testing.v1 import AppTest

import onboarding
import paths
from paths import CODE_DIR, DATA_DIR
from tests.test_tc11_onboarding import make_telemetry

DASHBOARD = str(CODE_DIR / "dashboard.py")
TIMEOUT = 180

LEGACY_UNITS = {
    "Mafate": ["h1", "h2", "h3"],
    "SONG Kenya": ["Household 51", "Household 42", "Household 49", "Household 48"],
    "HEED Hall": ["CPE1"],
    "STAR": ["Site A", "Site B", "Site C", "Site D"],
    "Jharkhand": ["House 2", "House 3", "House 4"],
}


def _run():
    at = AppTest.from_file(DASHBOARD, default_timeout=TIMEOUT)
    at.run()
    return at


def _alert_status(at):
    return next(m.value for m in at.metric if m.label == "Alert status")


@pytest.fixture
def onboarded_env(tmp_path, monkeypatch):
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"schema_version": 1, "households": {}}))
    res = onboarding.onboard_household(
        "new_hh", make_telemetry(days=45), site="Acme Solar", registry_path=reg,
        models_dir=tmp_path / "models", telemetry_dir=tmp_path / "telemetry",
    )
    assert res.onboarded
    monkeypatch.setattr(onboarding, "REGISTRY_PATH", reg)
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    return reg


def test_existing_sites_and_units_still_listed():
    if not (DATA_DIR / "household_registry.json").exists():
        pytest.skip("run build_registry.py first")
    at = _run()
    assert not at.exception
    assert list(at.selectbox[0].options) == list(LEGACY_UNITS)
    for site, units in LEGACY_UNITS.items():
        at.selectbox[0].select(site).run()
        assert not at.exception, site
        assert list(at.selectbox[1].options) == units


def test_mafate_injection_view_unchanged():
    if not (DATA_DIR / "mafate_with_suppression.parquet").exists():
        pytest.skip("mafate artefacts missing")
    at = _run()
    assert not at.exception
    assert any(s.label.startswith("Injection level") for s in at.slider)
    assert _alert_status(at)


@pytest.mark.parametrize("site,unit", [("SONG Kenya", "Household 51"), ("STAR", "Site A"), ("HEED Hall", "CPE1")])
def test_legacy_non_mafate_views_render(site, unit):
    if not (DATA_DIR / "scored_sites_index.csv").exists():
        pytest.skip("scored artefacts missing")
    at = _run()
    at.selectbox[0].select(site).run()
    at.selectbox[1].select(unit).run()
    assert not at.exception
    assert any(m.label == "Mean load" for m in at.metric)


def test_onboarded_household_appears_in_dropdown(onboarded_env):
    at = _run()
    assert not at.exception
    assert "Acme Solar" in at.selectbox[0].options
    for site in LEGACY_UNITS:
        assert site in at.selectbox[0].options
    at.selectbox[0].select("Acme Solar").run()
    assert list(at.selectbox[1].options) == ["new_hh"]


def test_onboarded_household_shows_telemetry_counterfactual_suppression_alert(onboarded_env):
    at = _run()
    at.selectbox[0].select("Acme Solar").run()
    assert not at.exception, [e.value for e in at.exception]
    assert _alert_status(at).endswith("NORMAL")
    headings = [s.value for s in at.subheader]
    assert any("Telemetry" in h for h in headings)
    assert any("counterfactual" in h for h in headings)
    assert any("suppression" in h.lower() for h in headings)
    assert any("Alert tier" in h for h in headings)
    assert any(m.label == "Mean suppression" for m in at.metric)


def test_onboarded_household_without_telemetry_shows_hint(onboarded_env, tmp_path):
    (tmp_path / "telemetry" / "new_hh.parquet").unlink()
    at = _run()
    at.selectbox[0].select("Acme Solar").run()
    assert not at.exception
    assert any("No telemetry stored" in i.value for i in at.info)
