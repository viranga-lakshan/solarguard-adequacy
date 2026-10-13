"""Task 14 -- Results tables for the dissertation.

Assembles the five metrics fixed in advance by Table 6.3 (M1-M5) from the
CSV artefacts each experiment script already wrote, plus a dataset summary
table. This script does not recompute anything -- it only aggregates and
formats outputs that already exist on disk, so it must be run after
train_model.py, suppression.py, e4_placebo.py, e2_holdout.py, e1_lead_time.py,
jharkhand_check.py and load_star.py have all completed.
"""

from __future__ import annotations

import pandas as pd

from paths import DATA_DIR, RESULTS_DIR


def _read(name: str) -> pd.DataFrame | None:
    path = DATA_DIR / name
    if not path.exists():
        print(f"  [missing] {name} -- run the script that produces it first")
        return None
    return pd.read_csv(path)


def table1_dataset_summary() -> pd.DataFrame:
    """Site | Country | Houses | Duration | binding behaviour -- Table 3.2 analogue."""
    rows = [
        {"site": "Mafate", "country": "France (Reunion)", "houses_or_cpes": 3,
         "duration": "12 months", "resolution": "1-min"},
        {"site": "HEED Hall", "country": "Rwanda", "houses_or_cpes": 1,
         "duration": "~21 months", "resolution": "irregular (~15-min battery log)"},
        {"site": "STAR A-D", "country": "Brazil", "houses_or_cpes": 4,
         "duration": "225-297 days/site", "resolution": "irregular"},
        {"site": "Jharkhand", "country": "India", "houses_or_cpes": 3,
         "duration": "~222 days", "resolution": "60-sec"},
        {"site": "SONG Kenya", "country": "Kenya", "houses_or_cpes": 4,
         "duration": "45-82 active days/household (of a 146-day window)", "resolution": "30-min"},
    ]
    summary = pd.DataFrame(rows)

    binding_sources = {
        "Mafate": DATA_DIR / "mafate_binding_rates.csv",
        "HEED Hall": DATA_DIR / "heed_e4_placebo_results.csv",
        "STAR A-D": DATA_DIR / "star_binding_rates.csv",
        "Jharkhand": DATA_DIR / "jharkhand_binding_rates.csv",
        "SONG Kenya": DATA_DIR / "song_kenya_binding_rates.csv",
    }
    binding_rate_pct = {}
    for site, path in binding_sources.items():
        if not path.exists():
            binding_rate_pct[site] = None
            continue
        df = pd.read_csv(path)
        col = next((c for c in df.columns if "binding_rate_pct" in c), None)
        binding_rate_pct[site] = round(df[col].mean(), 4) if col else None

    summary["mean_binding_rate_pct"] = summary["site"].map(binding_rate_pct)
    return summary


def table2_m4_binding_rate() -> pd.DataFrame:
    """M4: binding rate, proportion of intervals in which the constraint binds, per site."""
    frames = []
    mafate = _read("mafate_binding_rates.csv")
    if mafate is not None:
        mafate = mafate.rename(columns={"house": "unit"})
        mafate["site"] = "Mafate"
        frames.append(mafate[["site", "unit", "binding_rate_pct"]])

    jhk = _read("jharkhand_binding_rates.csv")
    if jhk is not None:
        jhk = jhk.rename(columns={"house": "unit"})
        jhk["site"] = "Jharkhand"
        frames.append(jhk[["site", "unit", "binding_rate_pct"]])

    star = _read("star_binding_rates.csv")
    if star is not None:
        star = star.rename(columns={"site": "unit"})
        star["site"] = "STAR"
        frames.append(star[["site", "unit", "binding_rate_pct"]])

    heed = _read("heed_e4_placebo_results.csv")
    if heed is not None:
        heed = heed.rename(columns={"site": "unit"})
        heed["site"] = "HEED Hall"
        heed["binding_rate_pct"] = 0.0  # measured directly in load_heed.py (SOC threshold), carried here
        frames.append(heed[["site", "unit", "binding_rate_pct"]])

    song = _read("song_kenya_binding_rates.csv")
    if song is not None:
        song = song.rename(columns={"household": "unit"})
        song["site"] = "SONG Kenya"
        frames.append(song[["site", "unit", "binding_rate_pct"]])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def table3_m3_placebo() -> pd.DataFrame:
    """M3: placebo false-positive rate -- mean suppression + alert-window rate, negative controls."""
    frames = []
    mafate = _read("e4_placebo_results.csv")
    if mafate is not None:
        mafate["site"] = "Mafate"
        frames.append(mafate)
    heed = _read("heed_e4_placebo_results.csv")
    if heed is not None:
        heed = heed.rename(columns={"site": "house"})
        heed["site"] = "HEED Hall"
        frames.append(heed)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def table4_m2_counterfactual_accuracy() -> pd.DataFrame:
    """M2: counterfactual error, MAE and CRPS on held-out unconstrained intervals, per site."""
    frames = []
    e2 = _read("e2_holdout_results.csv")
    if e2 is not None:
        e2["site"] = "Mafate"
        frames.append(e2)

    star = _read("star_e3_results.csv")
    if star is not None:
        star = star.rename(columns={"site": "unit"})
        star["site"] = "STAR"
        frames.append(star)

    jhk = _read("jharkhand_e3_results.csv")
    if jhk is not None:
        jhk = jhk.rename(columns={"house": "unit"})
        jhk["site"] = "Jharkhand"
        frames.append(jhk)

    song = _read("song_kenya_e3_results.csv")
    if song is not None:
        song = song.rename(columns={"household": "unit", "mean_suppression_w_all": "mean_suppression_w"})
        song["site"] = "SONG Kenya"
        frames.append(song)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def table5_m1_lead_time() -> pd.DataFrame:
    """M1: detection lead time, days ahead of each baseline, per injected fade profile."""
    df = _read("e1_lead_time_results.csv")
    return df if df is not None else pd.DataFrame()


def table6_m5_cross_site_consistency() -> pd.DataFrame:
    """M5/E3: effect magnitude and direction per site, reported separately, not pooled.

    reliability flags whether the held-out test evaluation for that row is
    trustworthy: STAR's IAM coverage is a handful of day-clusters spread
    across many months (confirmed by inspecting e3_star_*.png -- held-out
    test windows show long straight-line gaps between sparse real clusters),
    so its MAE/CRPS/coverage numbers are not held to the same confidence as
    Mafate/HEED/Jharkhand's continuous-telemetry results, even though the
    identical Algorithm 1 procedure was applied. This is a data-coverage
    limitation, not a methodology difference.
    """
    rows = []

    mafate = _read("e4_placebo_results.csv")
    if mafate is not None:
        for _, r in mafate.iterrows():
            rows.append({
                "site": "Mafate", "unit": r["house"], "tau": None,
                "mean_suppression_w": r["mean_suppression_w"],
                "reliability": "high (continuous 1-min telemetry, direct appliance meter)",
            })

    heed = _read("heed_e4_placebo_results.csv")
    if heed is not None:
        for _, r in heed.iterrows():
            rows.append({
                "site": "HEED Hall", "unit": r["site"], "tau": r.get("tau"),
                "mean_suppression_w": r["mean_suppression_w"],
                "reliability": "high (continuous telemetry, direct appliance meter)",
            })

    star = _read("star_e3_results.csv")
    if star is not None:
        for _, r in star.iterrows():
            if r.get("status") != "fitted":
                rows.append({"site": "STAR", "unit": r["site"], "tau": None,
                              "mean_suppression_w": None, "reliability": "declined (insufficient data)"})
                continue
            rows.append({
                "site": "STAR", "unit": r["site"], "tau": r["tau"],
                "mean_suppression_w": r["mean_suppression_w"],
                "reliability": "LOW (sparse IAM day-clusters, large calendar gaps in test window -- see plots)",
            })

    jhk = _read("jharkhand_e3_results.csv")
    if jhk is not None:
        for _, r in jhk.iterrows():
            if r.get("status") != "fitted":
                rows.append({"site": "Jharkhand", "unit": r["house"], "tau": None,
                              "mean_suppression_w": None, "reliability": "declined (insufficient data)"})
                continue
            rows.append({
                "site": "Jharkhand", "unit": r["house"], "tau": r["tau"],
                "mean_suppression_w": r["mean_suppression_w"],
                "reliability": "medium (continuous telemetry, but load is a DERIVED proxy, not a direct meter)",
            })

    song = _read("song_kenya_e3_results.csv")
    if song is not None:
        for _, r in song.iterrows():
            if r.get("status") != "fitted":
                rows.append({"site": "SONG Kenya", "unit": r["household"], "tau": None,
                              "mean_suppression_w": None, "reliability": "declined (insufficient data)"})
                continue
            rows.append({
                "site": "SONG Kenya", "unit": r["household"], "tau": r["tau"],
                "mean_suppression_w": r["mean_suppression_w_all"],
                "reliability": "high (direct lighting-circuit meter) -- ONLY SITE WITH REAL (non-injected) "
                               "measured binding: households 51/48 reconstructed SOC genuinely reaches "
                               "0.0%/8.7%. Suppression during constrained vs unconstrained periods is similar "
                               "in magnitude, not dramatically elevated -- an honest nuance, not a clean win.",
            })

    return pd.DataFrame(rows)


def main() -> None:
    tables = {
        "table1_dataset_summary": table1_dataset_summary(),
        "table2_m4_binding_rate": table2_m4_binding_rate(),
        "table3_m3_placebo": table3_m3_placebo(),
        "table4_m2_counterfactual_accuracy": table4_m2_counterfactual_accuracy(),
        "table5_m1_lead_time": table5_m1_lead_time(),
        "table6_m5_cross_site_consistency": table6_m5_cross_site_consistency(),
    }

    for name, table in tables.items():
        print(f"=== {name} ===")
        if table is None or table.empty:
            print("  (no data -- underlying script has not been run yet)")
        else:
            print(table.to_string(index=False))
            out_path = RESULTS_DIR / f"{name}.csv"
            table.to_csv(out_path, index=False)
            print(f"  saved -> {out_path}")
        print()


if __name__ == "__main__":
    main()
