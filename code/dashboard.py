"""Task 13 -- SolarGuard Adequacy Monitor dashboard (FR-09).

Run with: streamlit run dashboard.py

Reads only artefacts already written by the pipeline scripts (parquet/CSV under
data/) -- it does not retrain or rescore anything itself, consistent with the
monitoring tier being separate from the estimation tier (Sec. 4.9).

Covers all five sites. Mafate keeps the richer view because it is the only site
with on-site irradiance and battery voltage channels, and it is the site the
injection demonstration (FR-07) runs on. The other four are read from the
normalised frames written by score_sites.py.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from injection import inject_degradation
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


@st.cache_data
def load_mafate() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "mafate_with_suppression.parquet")


@st.cache_data
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


def render_binding_table(site: str) -> None:
    name = BINDING_TABLES.get(site)
    if not name:
        return
    table = load_table(name)
    if table is not None:
        st.subheader(f"Measured binding rate -- {site}")
        st.dataframe(table, use_container_width=True)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    site = st.selectbox("Site", ["Mafate"] + list(OTHER_SITES))

    if site == "Mafate":
        unit_label = st.selectbox("House", ["h1", "h2", "h3"])
    else:
        unit_label = st.selectbox("Unit", list(OTHER_SITES[site]))

st.caption(SITE_NOTES.get(site, ""))


# --------------------------------------------------------------------------
# Mafate: full view, with the injection demonstration
# --------------------------------------------------------------------------
if site == "Mafate":
    df = load_mafate()
    house_tag = unit_label
    house_col = {"h1": "Load_h1", "h2": "Load_h2", "h3": "Load_h3"}[house_tag]

    with st.sidebar:
        date_min, date_max = df.index.min().date(), df.index.max().date()
        date_range = st.slider(
            "Date range", min_value=date_min, max_value=date_max, value=(date_min, date_max)
        )
        injection_level = st.slider("Injection level (capacity fade, %)", 0, 80, 0, step=10)

    window = df.loc[str(date_range[0]): str(date_range[1])].copy()

    if injection_level > 0:
        injected = inject_degradation(df, injection_level / 100.0, str(date_range[0]), house_col)
        window[house_col] = injected.loc[window.index, house_col]
        # Suppression must be recomputed against the injected load -- the saved
        # suppression column reflects the real (uninjected) trace.
        window[f"suppression_w_{house_tag}"] = (
            window[f"qtau_pred_{house_tag}"] - window[house_col]
        ).clip(lower=0.0)

    mean_suppression = window[f"suppression_w_{house_tag}"].mean()
    binding_rate = 100.0 * (window["Ubat"] <= 47.0).mean() if "Ubat" in window else float("nan")
    rolling_supp = window[f"suppression_w_{house_tag}"].rolling(24 * 60).mean()
    rolling_volt = window["Ubat"].rolling(24 * 60).mean()
    alert_active = (
        bool(((rolling_supp > 100.0) & (rolling_volt > 47.0)).fillna(False).iloc[-1])
        if len(window)
        else False
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean suppression", f"{mean_suppression:.1f} W")
    c2.metric("Binding rate", f"{binding_rate:.3f}%")
    c3.metric("Alert status", "\U0001F534 ALERT" if alert_active else "\U0001F7E2 OK")
    c4.metric("System status", "Degraded (injected)" if injection_level > 0 else "Nominal")

    st.subheader("Solar PV power and battery voltage")
    left, right = st.columns(2)
    with left:
        st.line_chart(downsample_for_chart(window[["PV_total_w"]]))
    with right:
        volt_df = window[["Ubat"]].copy()
        volt_df["disconnect_threshold"] = 47.0
        st.line_chart(downsample_for_chart(volt_df))

    st.subheader(f"Observed load vs. Qτ counterfactual -- {house_tag}")
    band = window[
        [house_col, f"qtau_pred_{house_tag}", f"q10_pred_{house_tag}", f"q90_pred_{house_tag}"]
    ].copy()
    band.columns = ["Observed", "Qτ counterfactual", "Q10", "Q90"]
    st.line_chart(downsample_for_chart(band))

    st.subheader("Rolling suppression (24h) and alert timeline")
    alert_df = pd.DataFrame(
        {
            "rolling_suppression_w": rolling_supp,
            "alert_fired": ((rolling_supp > 100.0) & (rolling_volt > 47.0)).astype(int)
            * rolling_supp.max()
            if rolling_supp.notna().any()
            else 0,
        }
    )
    st.line_chart(downsample_for_chart(alert_df))

    render_binding_table("Mafate")


# --------------------------------------------------------------------------
# Every other site: normalised scored frame from score_sites.py
# --------------------------------------------------------------------------
else:
    stem = OTHER_SITES[site][unit_label]
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
