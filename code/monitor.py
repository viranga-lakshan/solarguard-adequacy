"""Task 9 -- Monitor and alert.

Fires an alert only when rolling suppression is high AND rolling voltage
still looks normal. That conjunction is the dissertation's key finding: it
specifically flags the failure mode conventional monitoring cannot see
(equipment telemetry stays nominal while the household silently absorbs a
real capability loss), rather than duplicating what a voltage alarm already
catches.

"Fired" requires the same SUSTAIN_DAYS of persistence as e1_lead_time.py's
baselines, not a single instantaneous rolling-window crossing: Sec. 5.2.4
states the monitor "evaluates the trend of S-hat over a rolling window
rather than an instantaneous threshold, because the failure mode develops
over months", and Fig. 4.3's caption requires the monitor and every baseline
detector to share one interface "so that the evaluation compares like with
like" -- using a looser firing rule here than E1 uses for its baselines
would silently break that comparability. The full per-minute rolling trace
is still returned (as deprivation_alert_instant) for the dashboard's alert
timeline, where showing the raw underlying signal is useful even though the
"officially fired" day requires persistence.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from injection import inject_degradation
from paths import DATA_DIR, MODELS_DIR
from train_model import FEATURE_COLS

HOUSE_COL = "Load_h1"
HOUSE_TAG = "h1"
INJECTION_START = "2020-11-01"
# 60%, not 40%: E1 (e1_lead_time.py) already established that a 100 W
# suppression threshold on this house does not cross for 20% or 40% fade --
# a legitimate negative finding there, but a bad choice for THIS script's
# illustrative demo, which exists to show what a genuine detection looks
# like. Using a fade level E1 already measured as detectable keeps this
# demo honest rather than silently picking whatever makes the printed
# narrative look best.
INJECTION_FADE = 0.6
SUSTAIN_DAYS = 3  # matches e1_lead_time.py's baselines -- see module docstring


def check_alert(
    df: pd.DataFrame,
    model_qtau,
    window_hours: int = 24,
    suppression_threshold: float = 100.0,
    voltage_ok_threshold: float = 47.0,
) -> pd.DataFrame:
    scorable = df[FEATURE_COLS].notna().all(axis=1)
    pred = pd.Series(np.nan, index=df.index)
    pred.loc[scorable] = model_qtau.predict(df.loc[scorable, FEATURE_COLS])
    suppression = (pred - df[HOUSE_COL]).clip(lower=0.0)

    # min_periods=80% of the window: real telemetry has frequent short
    # dropouts (verified on Mafate -- the longest unbroken run of scored
    # suppression is well under a full 24h window), and a strict
    # full-window requirement would leave the monitor permanently blind.
    window = window_hours * 60
    rolling_supp = suppression.rolling(window, min_periods=int(window * 0.8)).mean()
    rolling_volt = df["Ubat"].rolling(window, min_periods=int(window * 0.8)).mean()

    alert_instant = (rolling_supp > suppression_threshold) & (rolling_volt > voltage_ok_threshold)
    voltage_alarm_instant = rolling_volt <= voltage_ok_threshold

    # Daily-resampled, SUSTAIN_DAYS-persistent version of each flag, then
    # broadcast back to the 1-min grid so "sustained" and "instant" share an
    # index for plotting/joining.
    daily_alert = alert_instant.resample("1D").mean() > 0.5
    daily_volt_alarm = voltage_alarm_instant.resample("1D").mean() > 0.5
    sustained_alert_daily = daily_alert.rolling(SUSTAIN_DAYS).sum() >= SUSTAIN_DAYS
    sustained_volt_daily = daily_volt_alarm.rolling(SUSTAIN_DAYS).sum() >= SUSTAIN_DAYS

    return pd.DataFrame(
        {
            "suppression_w": suppression,
            "rolling_suppression_w": rolling_supp,
            "rolling_voltage_v": rolling_volt,
            "deprivation_alert_instant": alert_instant,
            "voltage_alarm_instant": voltage_alarm_instant,
            "deprivation_alert_sustained": sustained_alert_daily.reindex(df.index, method="ffill"),
            "voltage_alarm_sustained": sustained_volt_daily.reindex(df.index, method="ffill"),
        },
        index=df.index,
    )


def first_true_day(flag: pd.Series, start: pd.Timestamp) -> str:
    hits = flag.loc[start:][flag.loc[start:].fillna(False)]
    if hits.empty:
        return "never fired"
    return str(hits.index[0])


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "mafate_merged.parquet")
    injected = inject_degradation(df, INJECTION_FADE, INJECTION_START, HOUSE_COL)

    with open(MODELS_DIR / f"mafate_{HOUSE_TAG}_qtau.pkl", "rb") as fh:
        model_qtau = pickle.load(fh)

    result = check_alert(injected, model_qtau)

    start = pd.Timestamp(INJECTION_START)
    alert_time = first_true_day(result["deprivation_alert_sustained"], start)
    voltage_time = first_true_day(result["voltage_alarm_sustained"], start)

    print(f"Injection: fade={INJECTION_FADE:.0%}, start={INJECTION_START}, house={HOUSE_COL}")
    print(f"Deprivation alert (suppression high, voltage normal) fired: {alert_time}")
    print(f"Voltage alarm fired: {voltage_time}")
    print()
    alert_fired = alert_time != "never fired"
    voltage_fired = voltage_time != "never fired"
    if alert_fired and not voltage_fired:
        print(
            "Core finding reproduced: the deprivation alert fires on a real "
            "injected capability loss while the conventional voltage alarm "
            "does not fire at all, because the injection caps load without "
            "touching battery voltage -- exactly the invisible-failure mode "
            "described in the research problem."
        )
    elif alert_fired and voltage_fired:
        print("Both signals fired; see timestamps above for lead/lag.")
    elif not alert_fired and not voltage_fired:
        print(
            f"Neither signal fired at {INJECTION_FADE:.0%} fade on this house -- "
            "a negative result at this fade level (consistent with E1's own "
            "finding that fade levels below ~60% do not cross this house's "
            "100 W suppression threshold)."
        )
    else:
        print("Only the voltage alarm fired -- the deprivation alert did not; this would be unexpected.")

    out_path = DATA_DIR / "mafate_monitor_result.parquet"
    result.to_parquet(out_path)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
