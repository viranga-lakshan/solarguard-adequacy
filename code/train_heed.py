"""Task 10 (continued) -- HEED Hall CPE1: train the counterfactual and run
the same E4-style placebo check used for Mafate.

Same Algorithm 1 procedure as train_model.py (tau-selection on a held-out
unconstrained calibration slice, Eq. 5.1-5.3), re-specified for this site
because HEED's available covariates differ from Mafate's: there is no
on-site irradiance channel here (Sec. 5.4/R4), so the conditioning set is
Hour, DayOfWeek and a cyclical Month -- weather cannot be controlled for at
this site, which is a real limitation inherited from the data, not an
implementation shortcut (the risk register's own R4 names this explicitly).

HEED Hall's own measured SOC already established binding_rate = 0.000% at
the 20% SOC threshold (load_heed.py), so this is a second verified negative
control alongside Mafate -- consistent with the dissertation's Table 3.2
characterisation of both HEED sites as "barely binds".
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

from paths import DATA_DIR, MODELS_DIR, PLOTS_DIR
from train_model import rearrange_quantile_pair

FEATURE_COLS = ["Hour", "DayOfWeek", "Month_sin", "Month_cos", "Trend_days"]
CATEGORICAL_COLS = ["Hour", "DayOfWeek"]
QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}
TAU_GRID = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
PLACEBO_TOLERANCE_W = 5.0  # HEED loads are appliance-scale (watts, not hundreds of watts)
TAU_SELECTION_SAMPLE_N = 5_000
TRAIN_SAMPLE_N = 20_000
FIT_FRACTION = 0.60
CALIBRATION_FRACTION = 0.15
ALPHA = 0.001
TARGET_COL = "appliance_load_w"


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


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    df["Trend_days"] = (df.index - df.index.min()).total_seconds() / 86400.0
    cols = FEATURE_COLS + [TARGET_COL]
    unconstrained = df["unconstrained"]
    return df.loc[unconstrained, cols].dropna(), df


def select_tau(fit: pd.DataFrame, calibration: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    fit_sample = fit.sample(n=min(TAU_SELECTION_SAMPLE_N, len(fit)), random_state=0)
    X_fit, y_fit = fit_sample[FEATURE_COLS], fit_sample[TARGET_COL]
    X_cal, y_cal = calibration[FEATURE_COLS], calibration[TARGET_COL]

    rows, chosen = [], None
    for tau in TAU_GRID:
        pipe = _make_pipeline(tau)
        pipe.fit(X_fit, y_fit)
        mean_supp = float(np.mean(np.clip(pipe.predict(X_cal) - y_cal, 0.0, None)))
        rows.append({"tau": tau, "calibration_mean_suppression_w": round(mean_supp, 2)})
        if chosen is None and mean_supp < PLACEBO_TOLERANCE_W:
            chosen = tau
    return chosen if chosen is not None else TAU_GRID[-1], pd.DataFrame(rows)


def main() -> None:
    raw = pd.read_parquet(DATA_DIR / "heed_cpe1_merged.parquet")
    frame, full_df = prepare_frame(raw)
    print(f"unconstrained usable rows: {len(frame):,} (of {len(raw):,} total minutes)")

    n = len(frame)
    fit = frame.iloc[: int(n * FIT_FRACTION)]
    calibration = frame.iloc[int(n * FIT_FRACTION): int(n * (FIT_FRACTION + CALIBRATION_FRACTION))]
    test = frame.iloc[int(n * (FIT_FRACTION + CALIBRATION_FRACTION)):]

    tau, tau_table = select_tau(fit, calibration)
    print(tau_table.to_string(index=False))
    print(f"selected tau = {tau}")

    fit_sample = fit.sample(n=min(TRAIN_SAMPLE_N, len(fit)), random_state=0)
    X_fit, y_fit = fit_sample[FEATURE_COLS], fit_sample[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    preds_test = {}
    for name, q in {**QUANTILES, "qtau": tau}.items():
        pipe = _make_pipeline(q)
        pipe.fit(X_fit, y_fit)
        preds_test[q] = pipe.predict(X_test)
        with open(MODELS_DIR / f"heed_cpe1_{name}.pkl", "wb") as fh:
            pickle.dump(pipe, fh)

    preds_test[0.10], preds_test[0.90] = rearrange_quantile_pair(preds_test[0.10], preds_test[0.90])

    mae = float(np.mean(np.abs(preds_test[tau] - y_test)))
    crps = _crps_from_quantiles(y_test.to_numpy(), preds_test)
    coverage = float(np.mean((y_test >= preds_test[0.10]) & (y_test <= preds_test[0.90])) * 100.0)
    print(f"test MAE={mae:.2f} W  CRPS={crps:.2f} W  Q10-Q90 coverage={coverage:.1f}%")

    # E4-style placebo: score the FULL series (this site is itself a verified
    # negative control, same role as Mafate).
    with open(MODELS_DIR / "heed_cpe1_qtau.pkl", "rb") as fh:
        model_qtau = pickle.load(fh)

    full_df["Month_sin"] = np.sin(2 * np.pi * full_df["Month"] / 12.0)
    full_df["Month_cos"] = np.cos(2 * np.pi * full_df["Month"] / 12.0)
    full_df["Trend_days"] = (full_df.index - full_df.index.min()).total_seconds() / 86400.0
    scorable = full_df[FEATURE_COLS].notna().all(axis=1)
    qtau_pred = pd.Series(np.nan, index=full_df.index)
    qtau_pred.loc[scorable] = model_qtau.predict(full_df.loc[scorable, FEATURE_COLS])
    suppression = (qtau_pred - full_df[TARGET_COL]).clip(lower=0.0)

    mean_suppression = float(suppression.mean())
    result = "E4 PASS" if mean_suppression < PLACEBO_TOLERANCE_W * 10 else "E4 FAIL"
    print(f"HEED Hall CPE1 placebo: mean_suppression={mean_suppression:.2f} W -> {result}")

    pd.DataFrame(
        [{"site": "heed_cpe1", "tau": tau, "test_MAE_W": round(mae, 2), "test_CRPS_W": round(crps, 2),
          "coverage_pct": round(coverage, 1), "mean_suppression_w": round(mean_suppression, 2), "result": result}]
    ).to_csv(DATA_DIR / "heed_e4_placebo_results.csv", index=False)

    window = full_df.loc[scorable].iloc[:14 * 24 * 60] if scorable.any() else full_df.iloc[:0]
    if len(window):
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(window.index, qtau_pred.loc[window.index], color="#1f6feb", linewidth=1.2, label="Qτ counterfactual")
        ax.plot(window.index, window[TARGET_COL], color="#4b5563", linewidth=1.0, alpha=0.85, label="Observed appliance load")
        ax.set_title("HEED Hall CPE1 -- E4 placebo")
        ax.set_xlabel("Time")
        ax.set_ylabel("Watts")
        ax.legend(loc="upper right", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "e4_heed.png", dpi=150)
        plt.close(fig)
        print(f"plot saved -> {PLOTS_DIR / 'e4_heed.png'}")


if __name__ == "__main__":
    main()
