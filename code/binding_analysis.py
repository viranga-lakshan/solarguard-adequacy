"""Task 2 -- Binding analysis for Mafate.

A row is "unconstrained" when the shared 48 V battery bank is above the
Ubat > 47.0 V threshold. Because all three houses draw from the same bank,
binding is a system-wide state: what differs per house is simply which rows
have a valid (non-missing) load reading for that house, so the per-house
breakdown below is "how many of this house's valid hours were spent in a
binding state", not three independently-defined thresholds.

SOC proxy is a simple linear voltage map (44 V = 0%, 54 V = 100%), used only
for an interpretable sanity check here -- it is not the SOC estimate used
anywhere in the actual suppression pipeline.
"""

from __future__ import annotations

import pandas as pd

from paths import DATA_DIR

UBAT_UNCONSTRAINED_THRESHOLD = 47.0
SOC_PROXY_MIN_V = 44.0
SOC_PROXY_MAX_V = 54.0

HOUSES = ["Load_h1", "Load_h2", "Load_h3"]


def soc_proxy(ubat: pd.Series) -> pd.Series:
    scaled = (ubat - SOC_PROXY_MIN_V) / (SOC_PROXY_MAX_V - SOC_PROXY_MIN_V) * 100.0
    return scaled.clip(lower=0.0, upper=100.0)


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "mafate_merged.parquet")
    ubat_known = df["Ubat"].notna()
    df["unconstrained"] = ubat_known & (df["Ubat"] > UBAT_UNCONSTRAINED_THRESHOLD)
    df["constrained"] = ubat_known & (df["Ubat"] <= UBAT_UNCONSTRAINED_THRESHOLD)
    df["soc_proxy_pct"] = soc_proxy(df["Ubat"])

    print(f"Ubat unconstrained threshold: > {UBAT_UNCONSTRAINED_THRESHOLD} V")
    print(f"Ubat range observed: {df['Ubat'].min():.2f} V - {df['Ubat'].max():.2f} V")
    print(f"SOC proxy range observed: {df['soc_proxy_pct'].min():.1f}% - {df['soc_proxy_pct'].max():.1f}%")
    print()

    rows = []
    for house in HOUSES:
        # Restrict to rows where both the house load and Ubat are known --
        # binding cannot be assessed when the voltage reading itself is missing.
        valid = df[house].notna() & ubat_known
        total_hours = valid.sum() / 60.0
        constrained_hours = (valid & df["constrained"]).sum() / 60.0
        binding_rate = 100.0 * constrained_hours / total_hours if total_hours else float("nan")
        rows.append(
            {
                "house": house,
                "total_hours": round(total_hours, 1),
                "constrained_hours": round(constrained_hours, 1),
                "binding_rate_pct": round(binding_rate, 4),
            }
        )

    table = pd.DataFrame(rows)
    print("Binding rate table (per house, shared-bank Ubat threshold):")
    print(table.to_string(index=False))

    table.to_csv(DATA_DIR / "mafate_binding_rates.csv", index=False)
    print()
    print(f"saved -> {DATA_DIR / 'mafate_binding_rates.csv'}")


if __name__ == "__main__":
    main()
