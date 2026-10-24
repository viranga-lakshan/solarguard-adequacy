"""Task 13 -- SolarGuard Adequacy Monitor dashboard (FR-09).

Run with: streamlit run dashboard.py

Reads only artefacts already written by the pipeline scripts (parquet/CSV under
data/) -- it does not retrain or rescore anything itself, consistent with the
monitoring tier being separate from the estimation tier (Sec. 4.9).

Covers all five sites. Mafate keeps the richer view because it is the only site
with on-site irradiance and battery voltage channels, and it is the site the
injection demonstration (FR-07) runs on. The other four are read from the
normalised frames written by score_sites.py.

The site/unit dropdown is built from data/household_registry.json, so a household
registered by onboarding.py appears automatically. Those households have no
pre-scored artefact: their view is computed live from their stored telemetry
(data/telemetry/<id>.parquet, plus an optional CSV upload) through
monitor.monitor_household, which keeps the registry the single source of truth.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from monitor import monitor_household
from onboarding import REGISTRY_PATH, load_registry
from paths import DATA_DIR

st.set_page_config(page_title="SolarGuard -- Adequacy Monitor", layout="wide")
st.title("SolarGuard -- Adequacy Monitor")

# site -> {unit label: model stem}, matching score_sites.py
OTHER_SITES = {
    "SONG Kenya": {
        "Household 51": "song_kenya_h51",
        "Household 42": "song_kenya_h42",
        "Household 49": "song_kenya_h49",
        "Household 48": "song_kenya_h48",
    },
    "HEED Hall": {"CPE1": "heed_cpe1"},
    "STAR": {"Site A": "star_a", "Site B": "star_b", "Site C": "star_c", "Site D": "star_d"},
    "Jharkhand": {
        "House 2": "jharkhand_house_2",
        "House 3": "jharkhand_house_3",
        "House 4": "jharkhand_house_4",
    },
}

BINDING_TABLES = {
    "Mafate": "mafate_binding_rates.csv",
    "SONG Kenya": "song_kenya_binding_rates.csv",
    "STAR": "star_binding_rates.csv",
    "Jharkhand": "jharkhand_binding_rates.csv",
}

SITE_NOTES = {
    "SONG Kenya": (
        "The only site in this study that binds in reality rather than by injection: "
        "households 51 and 48 reach reconstructed SOC of 0.0% and 8.7%. Note that "
        "estimated suppression during binding is **lower** than outside it, not higher "
        "-- binding here falls at night when the counterfactual already predicts almost "
        "no lighting load, so the measure has no headroom. This is reported as a "
        "limitation in Section 6.6 of the dissertation, not as a success."
    ),
    "HEED Hall": "Verified negative control. Never binds; E4 placebo passes at 0.00 W.",
    "STAR": (
        "Appliance-monitor coverage is sparse day-clusters with month-long calendar "
        "gaps, so these numbers carry LOW reliability. Charts will show long flat "
        "stretches where no real data exists."
    ),
    "Jharkhand": (
        "Load is a derived proxy reconstructed from current and voltage channels, not "
        "a direct meter -- medium reliability. Effectively never binds (max 0.0726%)."
    ),
    "Mafate": (
        "Verified negative control with full weather and battery-voltage channels. "
        "Never binds, which is why the injection harness (FR-07) runs here."
    ),
}


# cache_resource keeps ONE shared copy in memory. cache_data would pickle and copy the
# 159 MB Mafate frame on every rerun (every slider move); these frames are only ever
# read here -- each view slices and .copy()s its own window before changing anything.
MAFATE_COLUMNS = ["PV_total_w", "Ubat"] + [
    f"{col}_{h}" for h in ("h1", "h2", "h3")
    for col in ("q10_pred", "q90_pred", "qtau_pred", "suppression_w")
] + ["Load_h1", "Load_h2", "Load_h3"]


@st.cache_resource
def load_mafate() -> pd.DataFrame:
    # Only the 17 of 46 columns this view draws: about a third of the memory and read time,
    # which matters on the small Streamlit Cloud machine.
    return pd.read_parquet(DATA_DIR / "mafate_with_suppression.parquet", columns=MAFATE_COLUMNS)


@st.cache_resource
def load_scored(stem: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"scored_{stem}.parquet")


@st.cache_data
def load_table(name: str) -> pd.DataFrame | None:
    path = DATA_DIR / name
    return pd.read_csv(path) if path.exists() else None


def downsample_for_chart(frame: pd.DataFrame, max_points: int = 2000) -> pd.DataFrame:
    """Resample to a coarser frequency before handing data to st.line_chart.

    st.line_chart renders client-side via Vega-Lite, which is not built to
    handle hundreds of thousands of points: verified directly -- passing a
    full year of 1-minute Mafate data (474,603 rows) to it froze the browser
    tab and drove headless Chromium to 3+ GB of memory during testing. This
    only changes what is DRAWN; the metrics below are computed from the
    full-resolution window, not from this.
    """
    if len(frame) <= max_points:
        return frame
    span_minutes = (frame.index.max() - frame.index.min()).total_seconds() / 60.0
    target_freq_min = max(1, int(span_minutes / max_points))
    return frame.resample(f"{target_freq_min}min").mean()


@st.cache_data(show_spinner="Computing the Mafate view ...")
def mafate_view(house_tag: str, start: str, end: str, injection_level: int) -> dict:
    """Metrics and chart frames for one Mafate house, date range and fade level.

    Cached per setting, so moving a slider back to a position already seen is instant
    and only small, downsampled frames are kept -- not the full-resolution window."""
    df = load_mafate()
    house_col = {"h1": "Load_h1", "h2": "Load_h2", "h3": "Load_h3"}[house_tag]
    supp_col = f"suppression_w_{house_tag}"
    window = df.loc[start:end].copy()

    if injection_level > 0:
        from injection import inject_degradation  # imported on first use: it pulls in matplotlib (~1 s)

        # The fade only needs the load column, so copy that rather than the whole frame.
        injected = inject_degradation(df[[house_col]], injection_level / 100.0, start, house_col)
        window[house_col] = injected.loc[window.index, house_col]
        # Suppression must be recomputed against the injected load -- the saved
        # suppression column reflects the real (uninjected) trace.
        window[supp_col] = (window[f"qtau_pred_{house_tag}"] - window[house_col]).clip(lower=0.0)

    # min_periods = 80% of the window, the same as monitor.check_alert: Mafate's 1-minute
    # record has gaps, and a strict full-window requirement leaves the signal blank
    # everywhere (verified: 100% NaN), so the alert light could never turn on.
    min_p = int(24 * 60 * 0.8)
    rolling_supp = window[supp_col].rolling(24 * 60, min_periods=min_p).mean()
    rolling_volt = window["Ubat"].rolling(24 * 60, min_periods=min_p).mean()
    fired = (rolling_supp > 100.0) & (rolling_volt > 47.0)

    volt_df = window[["Ubat"]].copy()
    volt_df["disconnect_threshold"] = 47.0
    band = window[[house_col, f"qtau_pred_{house_tag}", f"q10_pred_{house_tag}", f"q90_pred_{house_tag}"]].copy()
    band.columns = ["Observed", "Qτ counterfactual", "Q10", "Q90"]
    alert_df = pd.DataFrame(
        {
            "rolling_suppression_w": rolling_supp,
            "alert_fired": fired.astype(int) * rolling_supp.max() if rolling_supp.notna().any() else 0,
        }
    )
    return {
        "mean_suppression": float(window[supp_col].mean()),
        "binding_rate": 100.0 * float((window["Ubat"] <= 47.0).mean()),
        "alert_active": bool(fired.fillna(False).iloc[-1]) if len(window) else False,
        "pv_chart": downsample_for_chart(window[["PV_total_w"]]),
        "volt_chart": downsample_for_chart(volt_df),
        "band_chart": downsample_for_chart(band),
        "alert_chart": downsample_for_chart(alert_df),
    }


def render_binding_table(site: str) -> None:
    name = BINDING_TABLES.get(site)
    if not name:
        return
    table = load_table(name)
    if table is not None:
        st.subheader(f"Measured binding rate -- {site}")
        st.dataframe(table, width="stretch")


def build_unit_map() -> tuple[dict[str, dict[str, str]], set[str]]:
    """site -> {unit label: household_id}. Benchmark units keep their original
    site/unit labels; any other registry entry (newly onboarded) is grouped
    under the site name it was registered with. Falls back to the original
    hard-coded lists if the registry is unreadable."""
    units = {"Mafate": {tag: f"mafate_{tag}" for tag in ("h1", "h2", "h3")}}
    units.update({name: dict(unit_map) for name, unit_map in OTHER_SITES.items()})
    try:
        households = load_registry(REGISTRY_PATH)["households"]
    except (OSError, ValueError, KeyError):
        return units, set()
    known = {hid for unit_map in units.values() for hid in unit_map.values()}
    onboarded = set()
    for hid, entry in households.items():
        if hid in known or entry.get("status") != "active":
            continue
        units.setdefault(entry.get("site", "Unspecified"), {})[entry.get("unit", hid)] = hid
        onboarded.add(hid)
    return units, onboarded


UNIT_MAP, ONBOARDED_IDS = build_unit_map()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
# A household can be opened directly by link, e.g. from the Maintenance Alerts page (?household=<id>).
requested_id = st.query_params.get("household")
requested_site = next((s for s, unit_map in UNIT_MAP.items() if requested_id in unit_map.values()), None)

with st.sidebar:
    st.header("Controls")
    site_names = list(UNIT_MAP)
    site = st.selectbox("Site", site_names, index=site_names.index(requested_site) if requested_site else 0)

    unit_names = list(UNIT_MAP[site])
    requested_unit = next((u for u, hid in UNIT_MAP[site].items() if hid == requested_id), None)
    unit_index = unit_names.index(requested_unit) if requested_unit else 0
    if site == "Mafate":
        unit_label = st.selectbox("House", unit_names, index=unit_index)
    else:
        unit_label = st.selectbox("Unit", unit_names, index=unit_index)

selected_id = UNIT_MAP[site][unit_label]
is_onboarded = selected_id in ONBOARDED_IDS

st.caption(
    SITE_NOTES.get(site, "")
    if not is_onboarded
    else "Newly onboarded household: scored live from its own telemetry with its own Algorithm 1 models."
)


# --------------------------------------------------------------------------
# Mafate: full view, with the injection demonstration
# --------------------------------------------------------------------------
if site == "Mafate" and not is_onboarded:
    df = load_mafate()
    house_tag = unit_label

    with st.sidebar:
        date_min, date_max = df.index.min().date(), df.index.max().date()
        # Open on 45 days from 1 Nov 2020 -- day 184, the start date of the E1 fade test --
        # rather than the whole year: 474k one-minute rows made every rerun slow, and this
        # window lines the fade demonstration up with Section 5.4.5. The full year is
        # still one slider drag away.
        default_start = min(max(pd.Timestamp("2020-11-01").date(), date_min), date_max)
        default_end = min(date_max, default_start + pd.Timedelta(days=44).to_pytimedelta())
        date_range = st.slider(
            "Date range", min_value=date_min, max_value=date_max, value=(default_start, default_end)
        )
        injection_level = st.slider("Injection level (capacity fade, %)", 0, 80, 0, step=10)

    if injection_level > 0 and date_range[0] <= date_min:
        st.warning(
            "The fade has no effect with the Date range starting on the first day of the record: "
            "its healthy baseline is the load BEFORE the start date, and there is none. "
            "Move the start of the Date range later (for example to 2020-11-01)."
        )

    view = mafate_view(house_tag, str(date_range[0]), str(date_range[1]), injection_level)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean suppression", f"{view['mean_suppression']:.1f} W")
    c2.metric("Binding rate", f"{view['binding_rate']:.3f}%")
    c3.metric("Alert status", "\U0001F534 ALERT" if view["alert_active"] else "\U0001F7E2 OK")
    c4.metric("System status", "Degraded (injected)" if injection_level > 0 else "Nominal")

    st.subheader("Solar PV power and battery voltage")
    left, right = st.columns(2)
    with left:
        st.line_chart(view["pv_chart"])
    with right:
        st.line_chart(view["volt_chart"])

    st.subheader(f"Observed load vs. Qτ counterfactual -- {house_tag}")
    st.line_chart(view["band_chart"])

    st.subheader("Rolling suppression (24h) and alert timeline")
    st.line_chart(view["alert_chart"])

    render_binding_table("Mafate")


# --------------------------------------------------------------------------
# Every other site: normalised scored frame from score_sites.py
# --------------------------------------------------------------------------
elif not is_onboarded:
    stem = UNIT_MAP[site][unit_label]
    df = load_scored(stem)

    with st.sidebar:
        date_min, date_max = df.index.min().date(), df.index.max().date()
        # A full record is 146 days to 7 months of dense telemetry; drawn end to
        # end the daily cycle collapses into a solid band. Default to a trailing
        # window so the shape is legible, with the full span still selectable.
        default_start = max(date_min, date_max - pd.Timedelta(days=21).to_pytimedelta())
        date_range = st.slider(
            "Date range", min_value=date_min, max_value=date_max,
            value=(default_start, date_max),
        )
        # Sites differ by three orders of magnitude in load, so one absolute
        # threshold cannot serve all of them. Scale it to each unit's own mean
        # load at roughly the proportion Mafate's documented 100 W rule
        # represents there, so the indicator stays comparable across sites
        # instead of sitting permanently red.
        default_threshold = max(0.05, round(0.5 * float(df["load_w"].mean()), 4))
        threshold = st.number_input(
            "Alert threshold (W)",
            min_value=0.0,
            value=float(default_threshold),
            step=float(max(0.01, default_threshold / 10)),
            help="Rolling 24h mean suppression above this raises an alert. Defaults "
            "to 50% of this unit's own mean load, which is roughly what Mafate's "
            "fixed 100 W rule corresponds to at that site.",
        )

    window = df.loc[str(date_range[0]): str(date_range[1])].copy()

    if window.empty:
        st.warning("No data in the selected date range.")
        st.stop()

    # Time-based window, not a fixed row count: these sites range from
    # 30-minute to 60-second resolution and some are irregular.
    rolling_supp = window["suppression_w"].rolling("24h").mean()
    alert_series = (rolling_supp > threshold).fillna(False)
    alert_active = bool(alert_series.iloc[-1])

    mean_suppression = float(window["suppression_w"].mean())
    constrained = ~window["unconstrained"].astype(bool)
    binding_rate = 100.0 * float(constrained.mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean suppression", f"{mean_suppression:.4f} W")
    c2.metric("Binding rate", f"{binding_rate:.4f}%")
    c3.metric("Alert status", "\U0001F534 ALERT" if alert_active else "\U0001F7E2 OK")
    c4.metric("Mean load", f"{float(window['load_w'].mean()):.4f} W")

    # The metrics above cover the selected window only. The dissertation reports
    # whole-record figures, so show those too rather than leaving a reader to
    # think the dashboard contradicts Chapter 6.
    st.caption(
        f"Selected window: {date_range[0]} to {date_range[1]} "
        f"({len(window):,} of {len(df):,} intervals). "
        f"Whole-record figures as reported in the dissertation -- mean suppression "
        f"{float(df['suppression_w'].mean()):.4f} W, binding rate "
        f"{100.0 * float((~df['unconstrained'].astype(bool)).mean()):.4f}%, "
        f"mean load {float(df['load_w'].mean()):.4f} W."
    )

    # Computed over the whole record, not the selected window: this is a
    # statement about the unit's character and the study's central caveat, so
    # it must not vanish just because the date slider excludes the few
    # binding intervals.
    all_constrained = ~df["unconstrained"].astype(bool)
    if all_constrained.any():
        supp_constrained = float(df.loc[all_constrained, "suppression_w"].mean())
        supp_free = float(df.loc[~all_constrained, "suppression_w"].mean())
        verdict = (
            "The measure is **not** responding to real binding here -- suppression is "
            "lower during binding than outside it. This is the limitation reported in "
            "Section 6.6, not a success."
            if supp_constrained <= supp_free
            else "Suppression is higher during binding than outside it, which is the "
            "direction the method predicts."
        )
        st.info(
            f"Over the whole record this unit genuinely binds in "
            f"{100.0 * float(all_constrained.mean()):.4f}% of intervals. Mean "
            f"suppression during binding is **{supp_constrained:.4f} W** against "
            f"**{supp_free:.4f} W** outside it. {verdict}"
        )

    st.subheader(f"Observed load vs. Qτ counterfactual -- {site} / {unit_label}")
    band = window[["load_w", "qtau_pred", "q10_pred", "q90_pred"]].copy()
    band.columns = ["Observed", "Qτ counterfactual", "Q10", "Q90"]
    st.line_chart(downsample_for_chart(band))

    if window["soc_pct"].notna().any():
        st.subheader("Reconstructed state of charge")
        soc = window[["soc_pct"]].astype(float).copy()
        soc["binding_floor"] = 20.0
        st.line_chart(downsample_for_chart(soc))

    st.subheader("Rolling suppression (24h) and alert timeline")
    alert_df = pd.DataFrame(
        {
            "rolling_suppression_w": rolling_supp,
            "alert_threshold": threshold,
        }
    )
    st.line_chart(downsample_for_chart(alert_df))

    render_binding_table(site)


# --------------------------------------------------------------------------
# Newly onboarded household: live scoring via monitor_household
# --------------------------------------------------------------------------
else:
    TIER_BADGE = {
        "NORMAL": "\U0001F7E2 NORMAL",
        "ADVISORY": "\U0001F7E1 ADVISORY",
        "CRITICAL": "\U0001F534 CRITICAL",
        "VACANT/INACTIVE": "\u26AA VACANT / INACTIVE",
        "VOLTAGE_ALARM": "\U0001F7E0 VOLTAGE ALARM",
        "UNKNOWN": "\u26AB INSUFFICIENT DATA",
    }
    TIER_LEVEL = {"NORMAL": 0, "ADVISORY": 1, "CRITICAL": 2}

    entry = load_registry(REGISTRY_PATH)["households"][selected_id]
    stored = DATA_DIR / "telemetry" / f"{selected_id}.parquet"

    telemetry = pd.read_parquet(stored) if stored.exists() else None
    with st.sidebar:
        upload = st.file_uploader(
            "Add telemetry (CSV)", type="csv",
            help="Columns: timestamp, load_w, voltage_v or soc_pct, and irradiance_w_m2 / "
            "temperature_c for weather-aware units. Merged with any stored telemetry.",
        )
    if upload is not None:
        extra = pd.read_csv(upload, parse_dates=["timestamp"])
        telemetry = extra if telemetry is None else pd.concat([telemetry, extra], ignore_index=True)
    if telemetry is None:
        st.info(
            f"No telemetry stored for **{selected_id}** yet. Upload a CSV in the sidebar, or place "
            f"it at `data/telemetry/{selected_id}.parquet` (written automatically by `onboarding.py`)."
        )
        st.stop()

    @st.cache_data(show_spinner="Scoring household ...")
    def score_household(household_id: str, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.drop_duplicates(subset="timestamp").sort_values("timestamp")
        return monitor_household(household_id, frame, registry_path=REGISTRY_PATH)

    try:
        df = score_household(selected_id, telemetry)
    except ValueError as exc:
        st.error(f"Cannot score {selected_id}: {exc}")
        st.stop()

    with st.sidebar:
        date_min, date_max = df.index.min().date(), df.index.max().date()
        default_start = max(date_min, date_max - pd.Timedelta(days=21).to_pytimedelta())
        date_range = st.slider(
            "Date range", min_value=date_min, max_value=date_max, value=(default_start, date_max)
        )
    window = df.loc[str(date_range[0]): str(date_range[1])]
    if window.empty:
        st.warning("No data in the selected date range.")
        st.stop()

    batt_label = "Battery voltage (V)" if entry["battery_metric"] == "voltage_v" else "Battery SOC (%)"
    ok = entry["battery_ok_threshold"]
    latest_tier = str(df["alert_tier"].iloc[-1])  # whole record: the current status, not the slider's

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean suppression", f"{window['suppression_w'].mean():.1f} W")
    c2.metric("Binding rate", f"{100.0 * (window['battery'] <= ok).mean():.3f}%")
    c3.metric("Alert status", TIER_BADGE.get(latest_tier, latest_tier))
    c4.metric("Vacant hours", f"{int(window['vacant'].sum())}")
    st.caption(
        f"{selected_id} | feature tier: {entry['feature_tier']} | tau = {entry['tau']} | "
        f"Advisory {entry['advisory_suppression_w']:.0f}-{entry['critical_suppression_w']:.0f} W, "
        f"Critical > {entry['critical_suppression_w']:.0f} W with {batt_label.lower()} above {ok:g}. "
        f"Hourly resolution; the status shown is the latest day-level verdict."
    )

    st.subheader(f"Telemetry -- {batt_label}")
    batt = window[["battery"]].rename(columns={"battery": batt_label}).copy()
    batt["normal threshold"] = ok
    st.line_chart(downsample_for_chart(batt))

    st.subheader(f"Observed load vs. Q\u03c4 counterfactual -- {site} / {unit_label}")
    band = window[["load_w", "counterfactual_w", "q10_w", "q90_w"]].copy()
    band.columns = ["Observed", "Q\u03c4 counterfactual", "Q10", "Q90"]
    st.line_chart(downsample_for_chart(band))

    st.subheader("Estimated suppression (hourly) and 24h rolling mean")
    supp = window[["suppression_w", "rolling_suppression_w"]].copy()
    supp["advisory_threshold"] = entry["advisory_suppression_w"]
    supp["critical_threshold"] = entry["critical_suppression_w"]
    st.line_chart(downsample_for_chart(supp))

    st.subheader("Alert tier timeline")
    timeline = pd.DataFrame(
        {"tier level (0 normal, 1 advisory, 2 critical)": window["alert_tier"].map(TIER_LEVEL)}
    )
    st.line_chart(downsample_for_chart(timeline))
    daily_records = df.attrs.get("daily")
    if daily_records:
        daily = pd.DataFrame.from_dict(daily_records, orient="index")
        st.dataframe(
            daily[["alert_tier", "advisory_run_days", "critical_run_days", "vacant"]].tail(30),
            width="stretch",
        )
