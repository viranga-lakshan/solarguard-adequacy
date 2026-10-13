"""Task 7 -- E2 counterfactual hold-out test (Sec. 6.2/6.5, metric M2).

Splits Mafate's unconstrained data by CALENDAR POSITION in the record, not
by literal month-of-year: the dataset runs 2020-05-01 to 2021-05-01, so
"month 1" is May 2020, not January. Within the fit months, the final
calibration months are further held out to select tau exactly as Task 3
does (Sec. 5.2.2) -- the estimator under test here is the full Algorithm 1
procedure, tau-selection included, not a model pre-committed to tau=0.5.
The final test months are never seen during fitting OR tau selection, and
that is what "E2 tests the counterfactual on genuinely unconstrained
held-out intervals" (Sec. 6.2) means in this implementation.

Split is PROPORTIONAL to each house's own usable month range (~70/15/15),
not the pasted spec's fixed "train 9 months". Two compounding reasons a
fixed split does not work here, both confirmed by running it and getting an
empty-slice ValueError rather than a useful error message:

  1. The documented irradiance-channel failure (Sec. 5.4 -- Gincl/Ta go NaN
     after Dec 2020) means weather-complete unconstrained data only covers
     month-offsets 0-7 (May-Dec 2020) for houses 1 and 3, not the full 0-11
     a "first 9 months" split assumes.
  2. House 2's own logger has a longer start-up gap than houses 1/3 (its
     first usable row is 2020-07-16, not 2020-05-01), so even an 8-month
     split sized for houses 1/3 leaves house 2 with only 6 usable months
     (offsets 0-5) -- a single shared FIT_MONTHS/CAL_MONTHS constant that
     works for one house silently empties another's calibration split.

Per house: n_months = max_offset + 1; fit gets ~70% of that (rounded, floor
1), calibration ~15% (floor 1), test gets whatever remains (floor 1).

M2 is reported as MAE and CRPS (Sec. 6.5), both against the qtau prediction,
with Q10/Q90 coverage reported alongside as the descriptive interval check.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from paths import DATA_DIR, PLOTS_DIR
from train_model import (
    FEATURE_COLS,
    QUANTILES,
    TRAIN_SAMPLE_N,
    _crps_from_quantiles,
    _make_pipeline,
    rearrange_quantile_pair,
    select_tau,
)

HOUSES = {"h1": "Load_h1", "h2": "Load_h2", "h3": "Load_h3"}
FIT_SHARE = 0.70
CAL_SHARE = 0.15
COLOR_OBSERVED = "#4b5563"
COLOR_PREDICTED = "#1f6feb"


def month_offset(index: pd.DatetimeIndex, start: pd.Timestamp) -> pd.Series:
    return (index.year - start.year) * 12 + (index.month - start.month)


def run_house(df: pd.DataFrame, house_col: str, house_tag: str) -> tuple[dict, pd.DataFrame]:
    unconstrained = df["Ubat"].notna() & (df["Ubat"] > 47.0)
    frame = df.loc[unconstrained, FEATURE_COLS + [house_col]].dropna()

    start = frame.index.min()
    offset = month_offset(frame.index, start)
    n_months = int(offset.max()) + 1
    fit_months = max(1, round(n_months * FIT_SHARE))
    cal_months = max(1, round(n_months * CAL_SHARE))
    # Shrink fit, then calibration, until at least one month remains for test.
    while fit_months + cal_months + 1 > n_months and fit_months > 1:
        fit_months -= 1
    while fit_months + cal_months + 1 > n_months and cal_months > 1:
        cal_months -= 1
    test_start_months = fit_months + cal_months

    fit = frame.loc[offset < fit_months]
    calibration = frame.loc[(offset >= fit_months) & (offset < test_start_months)]
    test = frame.loc[offset >= test_start_months]

    tau, _ = select_tau(fit, calibration, house_col)

    fit_sample = fit.sample(n=min(TRAIN_SAMPLE_N, len(fit)), random_state=0)
    X_fit, y_fit = fit_sample[FEATURE_COLS], fit_sample[house_col]
    X_test, y_test = test[FEATURE_COLS], test[house_col]

    preds = {}
    for q in sorted(set(QUANTILES.values()) | {tau}):
        pipe = _make_pipeline(q)
        pipe.fit(X_fit, y_fit)
        preds[q] = pipe.predict(X_test)

    preds[0.10], preds[0.90] = rearrange_quantile_pair(preds[0.10], preds[0.90])

    pred_qtau = preds[tau]
    mae = float(np.mean(np.abs(pred_qtau - y_test)))
    rmse = float(np.sqrt(np.mean((pred_qtau - y_test) ** 2)))
    coverage = float(np.mean((y_test >= preds[0.10]) & (y_test <= preds[0.90])) * 100.0)
    crps = _crps_from_quantiles(y_test.to_numpy(), preds)

    result = {
        "house": house_tag,
        "tau": tau,
        "fit_rows": len(fit),
        "fit_sampled": len(fit_sample),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "test_months": f"{test_start_months + 1}-{n_months}",
        "MAE_W": round(mae, 2),
        "RMSE_W": round(rmse, 2),
        "CRPS_W": round(crps, 2),
        "Q10_Q90_coverage_pct": round(coverage, 2),
    }

    holdout = pd.DataFrame(
        {"observed": y_test, "qtau_pred": pred_qtau, "q10_pred": preds[0.10], "q90_pred": preds[0.90]},
        index=test.index,
    )
    return result, holdout


def plot_holdout(holdout: pd.DataFrame, house_tag: str, tau: float) -> None:
    window = holdout.first("14D")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(window.index, window["observed"], color=COLOR_OBSERVED, linewidth=1.0, label="Observed (held-out)")
    ax.plot(window.index, window["qtau_pred"], color=COLOR_PREDICTED, linewidth=1.2, label=f"Q{tau:.2f} counterfactual")
    ax.fill_between(window.index, window["q10_pred"], window["q90_pred"], color=COLOR_PREDICTED, alpha=0.15, linewidth=0, label="Q10-Q90 band")

    ax.set_title(f"Mafate {house_tag} -- E2 hold-out, first 14 days of test window")
    ax.set_xlabel("Time")
    ax.set_ylabel("Watts")
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_path = PLOTS_DIR / f"e2_holdout_{house_tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  plot saved -> {out_path}")


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "mafate_merged.parquet")

    results = []
    for tag, col in HOUSES.items():
        print(f"E2 hold-out: house {tag} ...")
        result, holdout = run_house(df, col, tag)
        results.append(result)
        print(f"  {result}")
        plot_holdout(holdout, tag, result["tau"])

    table = pd.DataFrame(results)
    print()
    print(table.to_string(index=False))
    table.to_csv(DATA_DIR / "e2_holdout_results.csv", index=False)
    print()
    print(f"saved -> {DATA_DIR / 'e2_holdout_results.csv'}")


if __name__ == "__main__":
    main()
