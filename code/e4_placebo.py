"""Task 5 -- E4 placebo test.

Mafate is the dissertation's verified negative control: independent analysis
(battery voltage, SOC reconstruction, autonomy) already establishes it never
runs out of capability. If the suppression estimator reports a material
suppression signal here anyway, it is not detecting real deprivation -- it is
detecting a confound (e.g. unmodelled seasonality leaking through the Qtau
counterfactual). This script is the falsification test for that failure mode.

Also reports M3 (Sec. 6.5/Table 6.3): the proportion of rolling 24h windows
on this negative control in which monitor.py's alert condition would have
fired, using the same rolling-suppression / rolling-voltage logic as the
real monitor (duplicated here rather than imported, to keep this a read-only
report over the already-scored parquet instead of re-loading every model).

Metrics are computed over the FULL scored dataset (not a plotting window).
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

from paths import DATA_DIR, PLOTS_DIR

HOUSES = ["h1", "h2", "h3"]
PASS_THRESHOLD_W = 50.0
PLOT_WINDOW_DAYS = 14
# Early May 2020 is a near-vacant startup period for these households (load
# grows roughly 3x over the year per the dissertation's own binding-check
# notes), so plotting from the dataset's literal start produces an
# uninformative flat-zero figure. September falls after that ramp-up and
# before the documented post-Dec-2020 irradiance gap.
PLOT_WINDOW_START = "2020-09-01"

# Colour-blind-safe, muted palette: predicted = blue, observed = dark grey,
# suppression fill = a single warm hue reserved for the "deprivation" signal.
COLOR_PREDICTED = "#1f6feb"
COLOR_OBSERVED = "#4b5563"
COLOR_SUPPRESSION = "#d97706"


MONITOR_WINDOW_HOURS = 24
MONITOR_SUPPRESSION_THRESHOLD_W = 100.0
MONITOR_VOLTAGE_OK_V = 47.0


def summarise(df: pd.DataFrame, tag: str) -> dict:
    suppression = df[f"suppression_w_{tag}"].dropna()
    scored_hours = len(suppression) / 60.0
    binding_hours = df[f"binding_flag_{tag}"].sum() / 60.0
    binding_rate_pct = 100.0 * binding_hours / scored_hours if scored_hours else float("nan")
    over_100w_pct = 100.0 * (suppression > 100.0).mean() if len(suppression) else float("nan")

    # M3: proportion of rolling monitoring windows that would have alerted.
    # min_periods is set to 80% of the window rather than requiring every
    # minute present: real telemetry has frequent short dropouts (verified --
    # the longest unbroken run of suppression_w_h1 is only 469 of the 1440
    # minutes a strict window would need), and NFR-02 requires tolerating
    # gaps rather than the monitor silently going blind whenever one occurs.
    window = MONITOR_WINDOW_HOURS * 60
    rolling_supp = df[f"suppression_w_{tag}"].rolling(window, min_periods=int(window * 0.8)).mean()
    rolling_volt = df["Ubat"].rolling(window, min_periods=int(window * 0.8)).mean()
    alert = (rolling_supp > MONITOR_SUPPRESSION_THRESHOLD_W) & (rolling_volt > MONITOR_VOLTAGE_OK_V)
    valid_windows = rolling_supp.notna() & rolling_volt.notna()
    alert_rate_pct = 100.0 * alert[valid_windows].mean() if valid_windows.any() else float("nan")

    return {
        "house": tag,
        "mean_suppression_w": round(float(suppression.mean()), 2),
        "binding_rate_pct": round(binding_rate_pct, 4),
        "pct_hours_suppression_gt_100w": round(over_100w_pct, 4),
        "alert_window_rate_pct": round(alert_rate_pct, 4),
        "result": "E4 PASS" if suppression.mean() < PASS_THRESHOLD_W else "E4 FAIL",
    }


def plot_house(df: pd.DataFrame, tag: str, load_col: str) -> None:
    window = df.loc[PLOT_WINDOW_START:].iloc[: PLOT_WINDOW_DAYS * 24 * 60]
    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.fill_between(
        window.index,
        window[load_col],
        window[f"qtau_pred_{tag}"],
        where=window[f"suppression_w_{tag}"] > 0,
        color=COLOR_SUPPRESSION,
        alpha=0.35,
        label="Suppression",
        linewidth=0,
    )
    ax.plot(window.index, window[f"qtau_pred_{tag}"], color=COLOR_PREDICTED, linewidth=1.2, label="Qτ counterfactual")
    ax.plot(window.index, window[load_col], color=COLOR_OBSERVED, linewidth=1.0, label="Observed load", alpha=0.85)

    ax.set_title(f"Mafate {tag} -- E4 placebo (first {PLOT_WINDOW_DAYS} days)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Watts")
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_path = PLOTS_DIR / f"e4_placebo_{tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  plot saved -> {out_path}")


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "mafate_with_suppression.parquet")
    load_cols = {"h1": "Load_h1", "h2": "Load_h2", "h3": "Load_h3"}

    results = []
    for tag in HOUSES:
        result = summarise(df, tag)
        results.append(result)
        print(result)
        plot_house(df, tag, load_cols[tag])

    table = pd.DataFrame(results)
    table.to_csv(DATA_DIR / "e4_placebo_results.csv", index=False)
    print()
    print(table.to_string(index=False))
    print()
    print(f"saved -> {DATA_DIR / 'e4_placebo_results.csv'}")


if __name__ == "__main__":
    main()
