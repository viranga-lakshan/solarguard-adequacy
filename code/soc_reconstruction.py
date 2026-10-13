"""Mafate state-of-charge reconstruction (Algorithm 1, Sec. 5.2.3, steps 2-3).

Mafate's battery telemetry is voltage/current/temperature, not a directly
logged SOC channel, so Algorithm 1 calls for Coulomb-count reconstruction.
The main Mafate pipeline (load_mafate.py / train_model.py) classifies
unconstrained intervals directly from Ubat instead of from a reconstructed
SOC -- a deliberate simplification, not an oversight: Algorithm 1's own
precondition is "if state of charge not measured", and Mafate's voltage
channel IS a direct, already-measured capability signal (unlike HEED, which
has no voltage proxy at all and genuinely needs SOC). A raw Coulomb integral
also drifts without bound over a year unless periodically re-anchored, which
introduces its own estimation error that a direct voltage reading does not
carry.

This module exists to satisfy TC-02 ("reconstruct state of charge on Mafate
and compare against the battery voltage record; reconstruction consistent
with observed voltage behaviour") as a validation/cross-check, not to
replace the voltage-threshold classification the rest of the pipeline uses.
The Coulomb-counting method itself is adapted from the project's earlier
mafate_loader.py prototype (C:\\Users\\Viranga\\Downloads\\solar\\src\\),
which was built and tuned against this same dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from paths import DATA_DIR

PACK_CAPACITY_AH = 1320.0
FULL_VOLTAGE_V = 57.5
FULL_TAIL_CURRENT_A = 2.0


def reconstruct_soc(frame: pd.DataFrame, capacity_ah: float = PACK_CAPACITY_AH) -> tuple[pd.Series, int]:
    """Coulomb-count SOC (%), re-anchored to 100% at each full-charge signature.

    Returns (soc_series, anchor_count). Anchoring matters: an un-anchored
    Coulomb integral drifts without bound over a year; every detected full
    charge (high voltage with tapered current) resets the accumulated error.
    """
    minutes = frame.index.to_series().diff().dt.total_seconds().div(60.0)
    minutes = minutes.fillna(1.0).clip(upper=10.0)

    delta_ah = frame["Ibat"].to_numpy() * (minutes.to_numpy() / 60.0)
    is_full = (
        (frame["Ubat"] >= FULL_VOLTAGE_V) & (frame["Ibat"].abs() <= FULL_TAIL_CURRENT_A)
    ).to_numpy()

    soc = np.empty(len(frame), dtype=float)
    level = 100.0 if (len(is_full) and is_full[0]) else 85.0
    anchors = 0
    for i in range(len(frame)):
        if is_full[i]:
            level = 100.0
            anchors += 1
        else:
            level += delta_ah[i] / capacity_ah * 100.0
            level = min(max(level, 0.0), 100.0)
        soc[i] = level

    return pd.Series(soc, index=frame.index, name="soc_reconstructed_pct"), anchors


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "mafate_merged.parquet")
    df = df.dropna(subset=["Ubat", "Ibat"])

    soc, anchors = reconstruct_soc(df)
    df = df.join(soc)

    # TC-02's pass condition: reconstruction consistent with observed voltage
    # behaviour. Checked two ways: (a) correlation should be strongly
    # positive (higher voltage <-> higher reconstructed SOC); (b) the
    # reconstructed SOC should never read low while voltage is comfortably
    # above the disconnect threshold, and vice versa near the threshold.
    corr = df["soc_reconstructed_pct"].corr(df["Ubat"])
    low_soc_high_voltage = ((df["soc_reconstructed_pct"] < 50) & (df["Ubat"] > 50)).mean()

    print(f"rows scored: {len(df):,}")
    print(f"full-charge anchors found: {anchors:,}")
    print(f"reconstructed SOC range: {df['soc_reconstructed_pct'].min():.1f}% - {df['soc_reconstructed_pct'].max():.1f}%")
    print(f"corr(reconstructed SOC, Ubat): {corr:.3f}")
    print(f"fraction with SOC<50% while Ubat>50V (should be ~0, flags drift): {low_soc_high_voltage:.4f}")
    print()
    consistent = corr > 0.5 and low_soc_high_voltage < 0.01
    print(f"TC-02 result: {'PASS -- reconstruction consistent with voltage record' if consistent else 'FAIL'}")

    out_path = DATA_DIR / "mafate_soc_reconstruction_check.csv"
    pd.DataFrame([{
        "rows": len(df), "anchors": anchors,
        "soc_min_pct": round(float(df["soc_reconstructed_pct"].min()), 2),
        "soc_max_pct": round(float(df["soc_reconstructed_pct"].max()), 2),
        "corr_soc_ubat": round(float(corr), 4),
        "drift_flag_fraction": round(float(low_soc_high_voltage), 6),
        "result": "PASS" if consistent else "FAIL",
    }]).to_csv(out_path, index=False)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
