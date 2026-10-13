"""Task 8 -- E1 detection lead-time test.

Injects a known capacity-fade event on day 90 of the Mafate record (House 1)
and compares how many days each method takes to raise a sustained (3-day)
alert against it:

  Method A (this dissertation): rolling 3-day mean suppression > 100 W,
      where suppression = Qtau_predicted(features) - observed_load (Eq. 5.3),
      using the Task 3 model fitted at the selected upper quantile tau
      (Sec. 5.2.2) on real, pre-injection, weather/time-conditioned demand.
  Baseline B: voltage alarm, rolling 3-day mean Ubat < 46.0 V.
  Baseline C: CUSUM change-point detection on the rolling 7-day mean of the
      (injected) load series itself, flagged when the cumulative deviation
      from the pre-injection mean exceeds 5 standard deviations.

Baseline B is expected to never fire: the injection harness caps load only
(simulating the household silently adapting to a shrunk envelope) and never
touches Ubat, so a voltage telemetry channel genuinely has nothing to alarm
on here. That non-detection is not a bug in the experiment design -- it is
the dissertation's central empirical claim (SQ3): conventional equipment-
centred signals stay green while the household absorbs a real capability
loss. The table reports "never" rather than an infinite lead time in that
case, which is a more honest instrument reading than a numeric placeholder.
"""

from __future__ import annotations

import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from injection import inject_degradation
from paths import DATA_DIR, MODELS_DIR, PLOTS_DIR
from train_model import FEATURE_COLS

HOUSE_COL = "Load_h1"
HOUSE_TAG = "h1"
INJECTION_START = "2020-11-01"  # ~day 184, chosen to land inside logged data; see note in main()
SUPPRESSION_THRESHOLD_W = 100.0
VOLTAGE_ALARM_V = 46.0
SUSTAIN_DAYS = 3
CUSUM_SIGMA_MULTIPLIER = 5.0

COLOR_SUPPRESSION = "#d97706"
COLOR_VOLTAGE = "#4b5563"
COLOR_DETECT_A = "#1f6feb"
COLOR_DETECT_B = "#dc2626"


def _first_sustained_day(daily_flag: pd.Series, sustain_days: int) -> int | None:
    """Index (0-based, relative to daily_flag start) of the first day that
    begins a run of `sustain_days` consecutive True values."""
    run = daily_flag.rolling(sustain_days).sum()
    hits = run[run >= sustain_days]
    if hits.empty:
        return None
    end_pos = daily_flag.index.get_loc(hits.index[0])
    return end_pos - sustain_days + 1


def detect_method_a(df: pd.DataFrame, model_qtau) -> tuple[pd.Series, int | None]:
    scorable = df[FEATURE_COLS].notna().all(axis=1)
    pred = pd.Series(np.nan, index=df.index)
    pred.loc[scorable] = model_qtau.predict(df.loc[scorable, FEATURE_COLS])
    suppression = (pred - df[HOUSE_COL]).clip(lower=0.0)

    daily = suppression.resample("1D").mean()
    flag = daily > SUPPRESSION_THRESHOLD_W
    day = _first_sustained_day(flag, SUSTAIN_DAYS)
    return suppression, day


def detect_voltage_alarm(df: pd.DataFrame) -> int | None:
    daily = df["Ubat"].resample("1D").mean()
    flag = daily < VOLTAGE_ALARM_V
    return _first_sustained_day(flag, SUSTAIN_DAYS)


def detect_cusum(df: pd.DataFrame, start_date: str) -> int | None:
    daily_load = df[HOUSE_COL].resample("1D").mean()
    rolling7 = daily_load.rolling(7, min_periods=7).mean()

    pre = rolling7.loc[:start_date].dropna()
    mu, sigma = pre.mean(), pre.std()
    if sigma == 0 or np.isnan(sigma):
        return None

    deviation = (mu - rolling7).clip(lower=0.0)  # one-sided: only a DROP in load is a fade symptom
    cusum = deviation.cumsum()
    # Re-baseline cumulative sum from the injection start so the statistic
    # reflects accumulated evidence of a NEW regime, not pre-injection noise.
    pre_cusum = cusum.loc[:start_date]
    baseline_offset = pre_cusum.iloc[-1] if len(pre_cusum) else 0.0
    cusum = cusum - baseline_offset
    threshold = CUSUM_SIGMA_MULTIPLIER * sigma
    flag = cusum > threshold
    return _first_sustained_day(flag, 1)  # CUSUM crossing is itself cumulative evidence; no extra sustain window


def run_fade_level(df: pd.DataFrame, model_qtau, fade: float) -> dict:
    injected = inject_degradation(df, fade, INJECTION_START, HOUSE_COL)

    suppression, day_a = detect_method_a(injected, model_qtau)
    day_b = detect_voltage_alarm(injected)
    day_c = detect_cusum(injected, INJECTION_START)

    def fmt(day):
        return day if day is not None else "never"

    lead_vs_b = (day_b - day_a) if (day_a is not None and day_b is not None) else "N/A (baseline B never fired)"
    lead_vs_c = (day_c - day_a) if (day_a is not None and day_c is not None) else "N/A (baseline C never fired)"

    return {
        "fade_pct": f"{fade:.0%}",
        "your_method_day": fmt(day_a),
        "voltage_alarm_day": fmt(day_b),
        "cusum_day": fmt(day_c),
        "lead_time_vs_voltage_days": lead_vs_b,
        "lead_time_vs_cusum_days": lead_vs_c,
    }, suppression, injected


def plot_fade(suppression: pd.Series, injected: pd.DataFrame, fade: float) -> None:
    window = injected.loc[pd.Timestamp(INJECTION_START) - pd.Timedelta(days=7):].first("45D")
    supp_window = suppression.loc[window.index]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax1.plot(supp_window.index, supp_window, color=COLOR_SUPPRESSION, linewidth=1.0)
    ax1.axhline(SUPPRESSION_THRESHOLD_W, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax1.set_ylabel("Suppression (W)")
    ax1.set_title(f"House 1 -- E1 lead time, fade={fade:.0%}")

    ax2.plot(window.index, window["Ubat"], color=COLOR_VOLTAGE, linewidth=1.0)
    ax2.axhline(VOLTAGE_ALARM_V, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax2.set_ylabel("Ubat (V)")
    ax2.set_xlabel("Time")

    for ax in (ax1, ax2):
        ax.axvline(pd.Timestamp(INJECTION_START), color=COLOR_DETECT_B, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = PLOTS_DIR / f"e1_lead_time_fade{int(fade*100)}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  plot saved -> {out_path}")


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "mafate_merged.parquet")

    with open(MODELS_DIR / f"mafate_{HOUSE_TAG}_qtau.pkl", "rb") as fh:
        model_qtau = pickle.load(fh)

    results = []
    for fade in (0.2, 0.4, 0.6):
        result, suppression, injected = run_fade_level(df, model_qtau, fade)
        results.append(result)
        print(result)
        plot_fade(suppression, injected, fade)

    table = pd.DataFrame(results)
    print()
    print(table.to_string(index=False))
    table.to_csv(DATA_DIR / "e1_lead_time_results.csv", index=False)
    print()
    print(f"saved -> {DATA_DIR / 'e1_lead_time_results.csv'}")


if __name__ == "__main__":
    main()
