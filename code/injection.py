"""Task 6 -- Semi-synthetic capacity-fade injection harness.

Real suppression evidence is thin in these open deployments (verified: three
of four sites essentially never bind). This harness manufactures a known
ground truth by capping a house's load at a fraction of its healthy baseline
from a chosen start date onward, simulating progressive battery capacity
fade. Downstream detectors (Tasks 8/9) are evaluated against this known
injection point, which is the only way to get a dated ground truth when the
real data barely has one.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

from paths import DATA_DIR, PLOTS_DIR

COLOR_BEFORE = "#4b5563"
COLOR_AFTER = "#dc2626"


def inject_degradation(
    df: pd.DataFrame,
    fade_fraction: float,
    start_date: str,
    house_col: str,
) -> pd.DataFrame:
    """Simulate battery capacity fade by capping available power.

    fade_fraction: 0.0 = healthy, 0.8 = 80% degraded.
    max_available = baseline_load * (1 - fade_fraction)
    simulated_load = min(actual_load, max_available)

    Baseline is the household's own 75th-percentile load (a robust proxy for
    "typical comfortable demand"), computed BEFORE the injection window so
    the fade is relative to the household's own healthy behaviour, not
    contaminated by the injected period itself.
    """
    df_injected = df.copy()
    mask = df_injected.index >= start_date

    baseline = df_injected.loc[~mask, house_col].quantile(0.75)
    max_avail = baseline * (1.0 - fade_fraction)

    df_injected.loc[mask, house_col] = df_injected.loc[mask, house_col].clip(upper=max_avail)
    return df_injected


def plot_injection(df: pd.DataFrame, df_injected: pd.DataFrame, house_col: str, start_date: str, fade: float) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))

    before = df.loc[: pd.Timestamp(start_date) - pd.Timedelta(days=1)].last("14D")
    after = df_injected.loc[start_date:].first("14D")

    ax.plot(before.index, before[house_col], color=COLOR_BEFORE, linewidth=1.0, label="Before injection (healthy)")
    ax.plot(after.index, after[house_col], color=COLOR_AFTER, linewidth=1.0, label=f"After injection (fade={fade:.0%})")
    ax.axvline(pd.Timestamp(start_date), color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_title(f"House 1 load profile -- capacity-fade injection ({fade:.0%})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Watts")
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_path = PLOTS_DIR / f"injection_fade_{int(fade*100)}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  plot saved -> {out_path}")


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "mafate_merged.parquet")
    house_col = "Load_h1"
    start_date = "2020-11-01"

    for fade in (0.2, 0.4, 0.6):
        injected = inject_degradation(df, fade, start_date, house_col)
        before_mean = df.loc[:start_date, house_col].mean()
        after_mean = injected.loc[start_date:, house_col].mean()
        print(
            f"fade={fade:.0%}  mean_before={before_mean:.1f} W  "
            f"mean_after_injection={after_mean:.1f} W"
        )
        plot_injection(df, injected, house_col, start_date, fade)


if __name__ == "__main__":
    main()
