"""Maintenance Alerts -- the view the maintenance team works from.

Every actively monitored (onboarded) household is scored from its stored
telemetry with its own models (monitor.monitor_household), and its latest
daily alert level is listed with the evidence behind it. Alerts are routed to
maintenance, not billing (Section 3.12 / 6.6), so the recommended action is a
technical check, with payment status checked first.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from monitor import monitor_household
from onboarding import REGISTRY_PATH, load_registry
from paths import DATA_DIR

st.set_page_config(page_title="SolarGuard -- Maintenance Alerts", layout="wide")
st.title("Maintenance Alerts")
st.caption(
    "Households monitored live from their own telemetry and models. An alert is raised only when the "
    "24-hour average suppression stays high for three days in a row while the battery still looks healthy. "
    "Alerts go to the maintenance team, not billing."
)

BADGE = {
    "CRITICAL": "\U0001F534 CRITICAL",
    "ADVISORY": "\U0001F7E1 ADVISORY",
    "NORMAL": "\U0001F7E2 NORMAL",
    "VACANT/INACTIVE": "⚪ VACANT / INACTIVE",
    "VOLTAGE_ALARM": "\U0001F7E0 VOLTAGE ALARM",
    "UNKNOWN": "⚫ INSUFFICIENT DATA",
}
ORDER = {"CRITICAL": 0, "VOLTAGE_ALARM": 1, "ADVISORY": 2, "UNKNOWN": 3, "VACANT/INACTIVE": 4, "NORMAL": 5}


@st.cache_data(show_spinner="Scoring monitored households ...")
def score(household_id: str, mtime: float) -> pd.DataFrame | None:
    path = DATA_DIR / "telemetry" / f"{household_id}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path).drop_duplicates(subset="timestamp").sort_values("timestamp")
    return monitor_household(household_id, frame, registry_path=REGISTRY_PATH)


def summarise(household_id: str, entry: dict, df: pd.DataFrame) -> dict:
    daily = df.attrs.get("daily", {})
    days = sorted(daily)
    last_day = days[-1] if days else df.index.max().normalize()
    tier = daily[last_day]["alert_tier"] if days else str(df["alert_tier"].iloc[-1])
    since = last_day
    for day in reversed(days):  # start of the current run of this alert level
        if daily[day]["alert_tier"] != tier:
            break
        since = day
    window = df.loc[str(last_day.date())]
    return {
        "household_id": household_id,
        "site": entry.get("site", ""),
        "tier": tier,
        "since": since.date(),
        "last_day": last_day.date(),
        "rolling_suppression_w": float(df["rolling_suppression_w"].iloc[-1]),
        "rolling_battery": float(df["rolling_battery"].iloc[-1]),
        "battery_unit": "V" if entry.get("battery_metric") == "voltage_v" else "% SOC",
        "battery_ok": entry.get("battery_ok_threshold"),
        "critical_w": entry.get("critical_suppression_w", 100.0),
        "normal_kwh": float(window["counterfactual_w"].sum()) / 1000.0,
        "actual_kwh": float(window["load_w"].sum()) / 1000.0,
    }


registry = load_registry(REGISTRY_PATH)["households"]
monitored = {hid: e for hid, e in registry.items() if e.get("origin") == "onboarded" and e.get("status") == "active"}
if not monitored:
    st.info("No households are monitored live yet. Onboard one with `python onboarding.py`.")
    st.stop()

rows, frames = [], {}
for hid, entry in monitored.items():
    path = DATA_DIR / "telemetry" / f"{hid}.parquet"
    try:
        df = score(hid, path.stat().st_mtime if path.exists() else 0.0)
    except ValueError as exc:
        st.warning(f"{hid}: cannot be scored ({exc})")
        continue
    if df is None or df.empty:
        continue
    frames[hid] = df
    rows.append(summarise(hid, entry, df))
rows.sort(key=lambda r: (ORDER.get(r["tier"], 9), r["household_id"]))

c1, c2, c3 = st.columns(3)
c1.metric("Households monitored", len(rows))
c2.metric("CRITICAL alerts", sum(r["tier"] == "CRITICAL" for r in rows))
c3.metric("ADVISORY alerts", sum(r["tier"] == "ADVISORY" for r in rows))

st.subheader("Open alerts")
open_alerts = [r for r in rows if r["tier"] in ("CRITICAL", "ADVISORY")]
if not open_alerts:
    st.success("No open alerts. Every monitored household is receiving its normal level of service.")
for r in open_alerts:
    box = st.error if r["tier"] == "CRITICAL" else st.warning
    shortfall = max(0.0, r["normal_kwh"] - r["actual_kwh"])
    box(
        f"**{BADGE[r['tier']]} -- {r['household_id']}** ({r['site']})  \n"
        f"Alert level since **{r['since']}** (latest data {r['last_day']}).  \n"
        f"**Evidence:** 24-hour average suppression **{r['rolling_suppression_w']:.0f} W** "
        f"(critical above {r['critical_w']:.0f} W), while the battery still looks healthy at "
        f"**{r['rolling_battery']:.1f} {r['battery_unit']}** (healthy above {r['battery_ok']:g} {r['battery_unit']}). "
        f"On {r['last_day']} the household used **{r['actual_kwh']:.2f} kWh** against a normal "
        f"**{r['normal_kwh']:.2f} kWh** (about {shortfall:.2f} kWh not supplied).  \n"
        f"**Action:** send a technician to check the battery's usable capacity and the household's "
        f"connection. Check payment status first; do not pass this alert to billing."
    )
    recent = frames[r["household_id"]].last("14D")
    st.line_chart(
        pd.DataFrame(
            {
                "24-hour average suppression (W)": recent["rolling_suppression_w"],
                "Critical threshold (W)": r["critical_w"],
            }
        )
    )

st.subheader("All monitored households")
st.dataframe(
    pd.DataFrame(
        {
            "Household": [r["household_id"] for r in rows],
            "Site": [r["site"] for r in rows],
            "Alert level": [BADGE.get(r["tier"], r["tier"]) for r in rows],
            "Since": [r["since"] for r in rows],
            "24 h suppression (W)": [round(r["rolling_suppression_w"], 1) for r in rows],
            "Battery (24 h avg)": [f"{r['rolling_battery']:.1f} {r['battery_unit']}" for r in rows],
        }
    ),
    hide_index=True,
    width="stretch",
)
st.caption("Households marked 'Demo (synthetic)' use made-up data to demonstrate the workflow; "
           "all results in the dissertation come from the fifteen real households.")
