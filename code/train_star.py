"""Task 11 (E3 continuation) -- STAR counterfactual model, sites A-D.

Same Algorithm 1 procedure as train_model.py/train_heed.py, re-specified for
this site's available covariates only -- no methodology change, per NFR-06
("a new site requires only a new loader, not changes to the estimator") and
per the explicit instruction not to tune the method for better results.
Like HEED, STAR carries no on-site irradiance channel (Sec. 5.4), so the
conditioning set is Hour, DayOfWeek, cyclical Month and a linear trend.

Evidence is thin by construction, not by choice: IAM (load) coverage is a
handful of dated files per site rather than a continuous year, and the
Battery x IAM inner join (both signals must exist at the same minute) leaves
672-3,754 usable rows per site -- reported honestly per site rather than
padded. Every site's binding rate is already independently known to be
~0% (load_star.py), so there is no genuinely constrained data to hold out
separately; "unconstrained" here is effectively the whole available record,
same structural situation as Mafate and HEED.
"""

from __future__ import annotations

import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")

from load_star import load_site_merged
from paths import DATA_DIR, MODELS_DIR, PLOTS_DIR
from train_model import rearrange_quantile_pair

SITES = ["A", "B", "C", "D"]
FEATURE_COLS = ["Hour", "DayOfWeek", "Month_sin", "Month_cos", "Trend_days"]
CATEGORICAL_COLS = ["Hour", "DayOfWeek"]
QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}
TAU_GRID = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
TAU_SELECTION_SAMPLE_N = 2_000
TRAIN_SAMPLE_N = 5_000
FIT_FRACTION = 0.60
CALIBRATION_FRACTION = 0.15
ALPHA = 0.001
TARGET_COL = "load_w"
MIN_USABLE_ROWS = 200  # below this, Algorithm 1's own "decline to answer" rule applies (Sec. 5.2.3)


def _make_pipeline(quantile: float) -> Pipeline:
    ct = ColumnTransformer(
        [("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_COLS)],
        remainder="passthrough",
    )
    return Pipeline(
        [("features", ct), ("model", QuantileRegressor(quantile=quantile, alpha=ALPHA, solver="highs"))]
    )


def _crps_from_quantiles(y_true: np.ndarray, quantile_preds: dict[float, np.ndarray]) -> float:
    losses = []
    for tau in sorted(quantile_preds):
        diff = y_true - quantile_preds[tau]
        pinball = np.maximum(tau * diff, (tau - 1) * diff)
        losses.append(pinball.mean())
    return float(2.0 * np.mean(losses))


def select_tau(fit: pd.DataFrame, calibration: pd.DataFrame, placebo_tolerance_w: float) -> tuple[float, pd.DataFrame]:
    fit_sample = fit.sample(n=min(TAU_SELECTION_SAMPLE_N, len(fit)), random_state=0)
    X_fit, y_fit = fit_sample[FEATURE_COLS], fit_sample[TARGET_COL]
    X_cal, y_cal = calibration[FEATURE_COLS], calibration[TARGET_COL]

    rows, chosen = [], None
    for tau in TAU_GRID:
        pipe = _make_pipeline(tau)
        pipe.fit(X_fit, y_fit)
        mean_supp = float(np.mean(np.clip(pipe.predict(X_cal) - y_cal, 0.0, None)))
        rows.append({"tau": tau, "calibration_mean_suppression_w": round(mean_supp, 2)})
        if chosen is None and mean_supp < placebo_tolerance_w:
            chosen = tau
    return chosen if chosen is not None else TAU_GRID[-1], pd.DataFrame(rows)


def run_site(site: str) -> dict | None:
    df = load_site_merged(site)
    frame = df[FEATURE_COLS + [TARGET_COL]].dropna()

    if len(frame) < MIN_USABLE_ROWS:
        print(f"  STAR-{site}: only {len(frame)} usable rows (< {MIN_USABLE_ROWS}) -- "
              f"declining to fit, per Algorithm 1's low-confidence rule (Sec. 5.2.3)")
        return {"site": site, "status": "low_confidence_insufficient_data", "n_rows": len(frame)}

    n = len(frame)
    fit = frame.iloc[: int(n * FIT_FRACTION)]
    calibration = frame.iloc[int(n * FIT_FRACTION): int(n * (FIT_FRACTION + CALIBRATION_FRACTION))]
    test = frame.iloc[int(n * (FIT_FRACTION + CALIBRATION_FRACTION)):]

    if len(fit) < 30 or len(calibration) < 10 or len(test) < 10:
        print(f"  STAR-{site}: split too thin (fit={len(fit)}, cal={len(calibration)}, test={len(test)}) -- declining")
        return {"site": site, "status": "low_confidence_split_too_thin", "n_rows": n}

    placebo_tolerance_w = max(5.0, 0.1 * fit[TARGET_COL].mean())  # scale to this site's own load level
    tau, tau_table = select_tau(fit, calibration, placebo_tolerance_w)
    print(f"  STAR-{site} tau selection (tolerance={placebo_tolerance_w:.1f} W):")
    print("    " + tau_table.to_string(index=False).replace("\n", "\n    "))
    print(f"  selected tau = {tau}")

    fit_sample = fit.sample(n=min(TRAIN_SAMPLE_N, len(fit)), random_state=0)
    X_fit, y_fit = fit_sample[FEATURE_COLS], fit_sample[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    preds_test = {}
    for name, q in {**QUANTILES, "qtau": tau}.items():
        pipe = _make_pipeline(q)
        pipe.fit(X_fit, y_fit)
        preds_test[q] = pipe.predict(X_test)
        with open(MODELS_DIR / f"star_{site.lower()}_{name}.pkl", "wb") as fh:
            pickle.dump(pipe, fh)

    preds_test[0.10], preds_test[0.90] = rearrange_quantile_pair(preds_test[0.10], preds_test[0.90])

    mae = float(np.mean(np.abs(preds_test[tau] - y_test)))
    crps = _crps_from_quantiles(y_test.to_numpy(), preds_test)
    coverage = float(np.mean((y_test >= preds_test[0.10]) & (y_test <= preds_test[0.90])) * 100.0)

    # Suppression over the FULL site record (all of it is "unconstrained" by
    # the independent SOC check, so this doubles as this site's E4-equivalent
    # placebo result -- see module docstring).
    with open(MODELS_DIR / f"star_{site.lower()}_qtau.pkl", "rb") as fh:
        model_qtau = pickle.load(fh)
    full_pred = model_qtau.predict(frame[FEATURE_COLS])
    suppression = np.clip(full_pred - frame[TARGET_COL], 0.0, None)
    mean_suppression = float(suppression.mean())

    plot_path = PLOTS_DIR / f"e3_star_{site.lower()}.png"
    fig, ax = plt.subplots(figsize=(11, 4.5))
    window = test.iloc[: min(len(test), 2000)]
    ax.plot(window.index, window[TARGET_COL], color="#4b5563", linewidth=1.0, label="Observed (test)")
    ax.plot(window.index, preds_test[tau][: len(window)], color="#1f6feb", linewidth=1.2, label=f"Q{tau:.2f} counterfactual")
    ax.set_title(f"STAR-{site} -- E3 cross-site, held-out test window")
    ax.set_xlabel("Time")
    ax.set_ylabel("Watts")
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return {
        "site": site,
        "status": "fitted",
        "tau": tau,
        "n_rows": n,
        "n_fit": len(fit),
        "n_calibration": len(calibration),
        "n_test": len(test),
        "test_MAE_W": round(mae, 2),
        "test_CRPS_W": round(crps, 2),
        "Q10_Q90_coverage_pct": round(coverage, 2),
        "mean_suppression_w": round(mean_suppression, 2),
        "mean_load_w": round(float(frame[TARGET_COL].mean()), 2),
    }


def main() -> None:
    results = []
    for site in SITES:
        print(f"STAR-{site} ...")
        result = run_site(site)
        if result is not None:
            results.append(result)
            print(f"  {result}")

    table = pd.DataFrame(results)
    print()
    print(table.to_string(index=False))
    table.to_csv(DATA_DIR / "star_e3_results.csv", index=False)
    print()
    print(f"saved -> {DATA_DIR / 'star_e3_results.csv'}")


if __name__ == "__main__":
    main()
