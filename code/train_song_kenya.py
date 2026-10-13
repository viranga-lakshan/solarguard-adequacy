"""Task 15 (continued) -- SONG Kenya: Algorithm 1 counterfactual + suppression.

Same procedure as every other site (train_model.py/train_heed.py/
train_star.py/train_jharkhand.py) -- no methodology change. The point of
this site is that, unlike the other four, its own binding check
(load_song_kenya.py) already shows REAL (not just injected) constraint
binding at two of four households (51: 1.47%, min SOC 0.0%; 48: 0.36%, min
SOC 8.7%), so this is the first genuine opportunity across the whole
project to see whether the suppression estimator reports a non-trivial,
non-injected result on real evidence -- directly answering part of the
dissertation's own "Future Recommendations" (obtain a site in the regime
the method targets).

Because the lighting circuit is tiny (3x3W LEDs, averages well under 1 W
across a day dominated by off periods), PLACEBO_TOLERANCE_W is scaled to
this site's own load level rather than reusing Mafate's 50 W constant --
the same proportional-tolerance approach already used for STAR/Jharkhand.
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

from load_song_kenya import HOUSEHOLDS, load_household
from paths import DATA_DIR, MODELS_DIR, PLOTS_DIR
from train_model import rearrange_quantile_pair

FEATURE_COLS = ["Hour", "DayOfWeek", "Month_sin", "Month_cos", "Trend_days"]
CATEGORICAL_COLS = ["Hour", "DayOfWeek"]
QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}
TAU_GRID = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
TAU_SELECTION_SAMPLE_N = 2_000
TRAIN_SAMPLE_N = 4_000
FIT_FRACTION = 0.60
CALIBRATION_FRACTION = 0.15
ALPHA = 0.001
TARGET_COL = "load_w"
MIN_USABLE_ROWS = 200


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
        rows.append({"tau": tau, "calibration_mean_suppression_w": round(mean_supp, 4)})
        if chosen is None and mean_supp < placebo_tolerance_w:
            chosen = tau
    return chosen if chosen is not None else TAU_GRID[-1], pd.DataFrame(rows)


def run_household(household: int) -> dict | None:
    df = load_household(household)
    frame = df.loc[df["unconstrained"], FEATURE_COLS + [TARGET_COL]].dropna()

    if len(frame) < MIN_USABLE_ROWS:
        print(f"  h{household}: only {len(frame)} usable unconstrained rows -- declining to fit")
        return {"household": household, "status": "low_confidence_insufficient_data", "n_rows": len(frame)}

    n = len(frame)
    fit = frame.iloc[: int(n * FIT_FRACTION)]
    calibration = frame.iloc[int(n * FIT_FRACTION): int(n * (FIT_FRACTION + CALIBRATION_FRACTION))]
    test = frame.iloc[int(n * (FIT_FRACTION + CALIBRATION_FRACTION)):]

    if len(fit) < 30 or len(calibration) < 10 or len(test) < 10:
        print(f"  h{household}: split too thin -- declining")
        return {"household": household, "status": "low_confidence_split_too_thin", "n_rows": n}

    placebo_tolerance_w = max(0.05, 0.1 * fit[TARGET_COL].mean())
    tau, tau_table = select_tau(fit, calibration, placebo_tolerance_w)
    print(f"  h{household} tau selection (tolerance={placebo_tolerance_w:.3f} W):")
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
        with open(MODELS_DIR / f"song_kenya_h{household}_{name}.pkl", "wb") as fh:
            pickle.dump(pipe, fh)

    preds_test[0.10], preds_test[0.90] = rearrange_quantile_pair(preds_test[0.10], preds_test[0.90])

    mae = float(np.mean(np.abs(preds_test[tau] - y_test)))
    crps = _crps_from_quantiles(y_test.to_numpy(), preds_test)
    coverage = float(np.mean((y_test >= preds_test[0.10]) & (y_test <= preds_test[0.90])) * 100.0)

    # Score the FULL household record (not just the unconstrained training
    # frame) -- this is where real suppression, if present, shows up, same
    # as suppression.py does for Mafate.
    with open(MODELS_DIR / f"song_kenya_h{household}_qtau.pkl", "rb") as fh:
        model_qtau = pickle.load(fh)
    full_df = df[FEATURE_COLS + [TARGET_COL, "soc_pct", "unconstrained"]].dropna(subset=FEATURE_COLS)
    full_pred = model_qtau.predict(full_df[FEATURE_COLS])
    suppression = np.clip(full_pred - full_df[TARGET_COL], 0.0, None)
    full_df = full_df.assign(qtau_pred=full_pred, suppression_w=suppression)

    mean_suppression_all = float(suppression.mean())
    mean_suppression_constrained = float(full_df.loc[~full_df["unconstrained"], "suppression_w"].mean()) \
        if (~full_df["unconstrained"]).any() else float("nan")
    mean_suppression_unconstrained = float(full_df.loc[full_df["unconstrained"], "suppression_w"].mean())

    plot_path = PLOTS_DIR / f"song_kenya_h{household}.png"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax1.plot(full_df.index, full_df[TARGET_COL], color="#4b5563", linewidth=0.8, label="Observed load")
    ax1.plot(full_df.index, full_df["qtau_pred"], color="#1f6feb", linewidth=1.0, label=f"Q{tau:.2f} counterfactual")
    ax1.fill_between(full_df.index, full_df[TARGET_COL], full_df["qtau_pred"],
                      where=full_df["suppression_w"] > 0, color="#d97706", alpha=0.4, linewidth=0, label="Suppression")
    ax1.set_ylabel("Watts")
    ax1.set_title(f"SONG Kenya household {household}")
    ax1.legend(loc="upper right", frameon=False, fontsize=8)
    ax2.plot(full_df.index, full_df["soc_pct"], color="#059669", linewidth=0.8)
    ax2.axhline(20.0, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax2.set_ylabel("Reconstructed SOC (%)")
    ax2.set_xlabel("Time")
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return {
        "household": household,
        "status": "fitted",
        "tau": tau,
        "n_unconstrained_rows": n,
        "n_fit_sampled": len(fit_sample),
        "n_test": len(test),
        "test_MAE_W": round(mae, 4),
        "test_CRPS_W": round(crps, 4),
        "Q10_Q90_coverage_pct": round(coverage, 2),
        "mean_suppression_w_all": round(mean_suppression_all, 4),
        "mean_suppression_w_during_constrained": round(mean_suppression_constrained, 4),
        "mean_suppression_w_during_unconstrained": round(mean_suppression_unconstrained, 4),
        "mean_load_w": round(float(full_df[TARGET_COL].mean()), 4),
    }


def main() -> None:
    results = []
    for h in HOUSEHOLDS:
        print(f"household {h} ...")
        result = run_household(h)
        if result is not None:
            results.append(result)
            print(f"  {result}")

    table = pd.DataFrame(results)
    print()
    print(table.to_string(index=False))
    table.to_csv(DATA_DIR / "song_kenya_e3_results.csv", index=False)
    print(f"\nsaved -> {DATA_DIR / 'song_kenya_e3_results.csv'}")


if __name__ == "__main__":
    main()
