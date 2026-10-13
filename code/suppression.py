"""Task 4 -- Suppression index.

Implements Equation 5.3: S_hat = max(0, L_hat_tau(x) - L_obs), where
L_hat_tau is the Task 3 model fitted at the SELECTED upper quantile tau
(Sec. 5.2.2) -- not the median. Q10/Q50/Q90 are also scored and persisted,
but only as the descriptive uncertainty interval (FR-05); they do not feed
the suppression formula.

Applies the trained models to EVERY timestamp (not just the unconstrained
rows used for training), so the suppression signal is also computed for the
handful of rows where the battery is below the 47.0 V threshold -- that is
precisely where real suppression, if any, would show up.

Rows with missing Hour/DayOfWeek/Month_sin/Month_cos/Trend_days/Gincl/Ta
(e.g. the documented irradiance gap after Dec 2020) cannot be scored and are
left as NaN rather than silently imputed.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from paths import DATA_DIR, MODELS_DIR
from train_model import FEATURE_COLS

HOUSES = {"h1": "Load_h1", "h2": "Load_h2", "h3": "Load_h3"}
BINDING_SUPPRESSION_THRESHOLD_W = 50.0
BINDING_UBAT_THRESHOLD_V = 47.0


def _load_models(house_tag: str) -> dict:
    models = {}
    for name in ("q10", "q50", "q90", "qtau"):
        path = MODELS_DIR / f"mafate_{house_tag}_{name}.pkl"
        with open(path, "rb") as fh:
            models[name] = pickle.load(fh)
    return models


def score_house(df: pd.DataFrame, house_col: str, house_tag: str) -> pd.DataFrame:
    models = _load_models(house_tag)
    scorable = df[FEATURE_COLS].notna().all(axis=1)
    X = df.loc[scorable, FEATURE_COLS]

    out = pd.DataFrame(index=df.index)
    for name, model in models.items():
        col = f"{name}_pred_{house_tag}"
        out[col] = np.nan
        out.loc[scorable, col] = model.predict(X)

    # Each quantile is fit independently (separate pinball-loss minimisation
    # per tau), which does not guarantee Q10 <= Q90 pointwise -- verified
    # this crosses in practice (Q90 goes negative while Q10 sits at 0 early
    # in the Mafate record). Monotone rearrangement (Chernozhukov, Fernandez-
    # Val & Galichon, 2010) is the standard fix: swap the pair wherever it
    # crosses. This only tidies the descriptive uncertainty interval
    # (FR-05) -- it does not touch qtau or the suppression formula (Eq. 5.3).
    q10_col, q90_col = f"q10_pred_{house_tag}", f"q90_pred_{house_tag}"
    lo = out[[q10_col, q90_col]].min(axis=1)
    hi = out[[q10_col, q90_col]].max(axis=1)
    out[q10_col], out[q90_col] = lo, hi

    observed = df[house_col]
    # Eq. 5.3: the counterfactual is qtau (the selected upper quantile), not q50.
    suppression = (out[f"qtau_pred_{house_tag}"] - observed).clip(lower=0.0)
    out[f"suppression_w_{house_tag}"] = suppression

    ubat_known = df["Ubat"].notna()
    low_voltage = ubat_known & (df["Ubat"] < BINDING_UBAT_THRESHOLD_V)
    out[f"binding_flag_{house_tag}"] = (
        suppression.gt(BINDING_SUPPRESSION_THRESHOLD_W) & low_voltage
    )
    return out


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "mafate_merged.parquet")

    for tag, col in HOUSES.items():
        print(f"scoring house {tag} ({col}) ...")
        scored = score_house(df, col, tag)
        df = df.join(scored)

    out_path = DATA_DIR / "mafate_with_suppression.parquet"
    df.to_parquet(out_path)

    print()
    for tag in HOUSES:
        s = df[f"suppression_w_{tag}"]
        print(
            f"{tag}: scored_rows={s.notna().sum():,} "
            f"mean_suppression_w={s.mean():.2f} "
            f"binding_flag_count={int(df[f'binding_flag_{tag}'].sum())}"
        )

    print()
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
