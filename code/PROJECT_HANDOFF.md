# Project handoff — Deprivation-Aware Monitoring for Off-Grid Solar

Written 2026-10-01 for use when writing up the dissertation without further
Claude access. Everything below traces to an actual script run — nothing
here is invented. All file paths are relative to `C:\Users\Viranga\Downloads\solar2\code\`
unless stated otherwise.

## 1. What this is

Implementation of Algorithm 1 (dissertation Sec. 5.2.3): a matched-condition
quantile-regression counterfactual that estimates household demand
suppression from off-grid solar telemetry, applied across all four datasets
in Table 3.2 (Mafate, HEED, STAR, Jharkhand), plus the monitor/alert layer,
the injection harness, the four evaluation experiments (E1–E4), and a
Streamlit dashboard.

## 2. Pipeline, in run order

| Script | What it does | Status |
|---|---|---|
| `load_mafate.py` | Merge Mafate plant/house/weather CSVs onto 1-min grid | Done |
| `binding_analysis.py` | Mafate binding rate per house | Done |
| `soc_reconstruction.py` | Coulomb-count SOC reconstruction + consistency check (TC-02) | Done |
| `train_model.py` | Algorithm 1 on Mafate: τ-selection, Q10/Q50/Q90/Qτ fit per house | Done |
| `suppression.py` | Applies models to every timestamp, computes suppression (Eq. 5.3) | Done |
| `e4_placebo.py` | E4: placebo test on Mafate (verified negative control) | Done, PASS all 3 houses |
| `e2_holdout.py` | E2: counterfactual hold-out, per house | Done |
| `injection.py` | Semi-synthetic capacity-fade injection harness | Done |
| `e1_lead_time.py` | E1: detection lead time vs. voltage alarm & CUSUM baselines | Done |
| `monitor.py` | Continuous monitor + sustained-alert logic | Done |
| `load_heed.py` | HEED Hall CPE1 loader (appliance load + battery SOC) | Done |
| `train_heed.py` | Algorithm 1 on HEED + E4-style placebo | Done, PASS |
| `load_star.py` | STAR sites A–D: battery/SOC + IAM appliance load loader | Done |
| `train_star.py` | Algorithm 1 on STAR (E3) | Done — **see reliability caveat below** |
| `jharkhand_check.py` | Jharkhand binding check, 3 houses | Done |
| `train_jharkhand.py` | Algorithm 1 on Jharkhand with derived load (E3) | Done |
| `load_song_kenya.py` | SONG Kenya loader, 4 households, Coulomb-count SOC | Done, see §2b |
| `train_song_kenya.py` | Algorithm 1 on SONG Kenya (E3) | Done, see §2b |
| `results_tables.py` | Aggregates everything into `results/table1..6_*.csv` | Done |
| `dashboard.py` | Streamlit monitoring UI | Done, browser-verified |
| `tests/` | pytest suite, TC-01 through TC-10 | Done, 23/23 passing |

Run any script directly: `python <script>.py` from this directory. Full
re-run order: load_mafate → binding_analysis → train_model → suppression →
e4_placebo → e2_holdout → e1_lead_time → monitor → load_heed → train_heed →
load_star → train_star → jharkhand_check → train_jharkhand → results_tables.

**Runtime warning**: `train_model.py`, `train_star.py`, `train_jharkhand.py`
each take 10–20 minutes (quantile-regression τ-selection grid search, 9
candidates × linear-programming solver). `train_heed.py` is fast (~2 min).
Everything else is under a minute.

## 2b. A FIFTH site, added 2026-10-02: SONG Kenya — the first genuinely
    constrained deployment in the whole project

Found via web search (you authorized this explicitly), specifically
targeting your own dissertation's "Future Recommendations" goal of finding
a site in the regime the method actually targets. Two already-cited papers
in your literature review (Aitio & Howey 2021, and the Perriment et al.
2026 study your whole research problem is anchored on) both turned out to
use the SAME proprietary BBOXX commercial data — confirmed by reading both
full papers directly (acknowledgements name BBOXX, no data availability
section exists in either) — so that avenue is a dead end, consistent with
your own research notes' conclusion that genuinely constrained commercial
telemetry is inaccessible.

Instead found: **Clements, A.F. & McCulloch, M. (2019). "Time series
lighting electricity data for rural households using Solar Nano-grids in
Kenya." Mendeley Data, V1, DOI: 10.17632/4yv37hngp6.1. CC BY 4.0.**
Methodology reference: Clements, A.F., DPhil thesis, University of Oxford,
2019 ("Data-driven approaches enabling the design of community energy
systems in the Global South").

**Why it matters**: 51 households, each with a tiny 62 Wh Li-ion battery
(vs. Mafate's 1320 Ah bank) charged at a central solar hub and carried home
to power a 3×3W LED lighting circuit — by far the smallest system in the
whole project, and the first one to show REAL measured binding, not just
injected:
- Household 51: reconstructed SOC genuinely reaches **0.0%** (full
  depletion), binding rate **1.47%** — the highest of any site/unit in this
  entire project by over an order of magnitude (`data/song_kenya_binding_rates.csv`)
- Household 48: reconstructed SOC reaches 8.7%, binding rate 0.36%, at a
  DIFFERENT time than household 51's dip (independent events, not one
  shared data artifact)
- Households 42/49: never bind (0%), consistent with the pattern elsewhere

**The honest nuance, not a clean win**: household 51's deep depletion
coincides with the dataset's own documented central-hub shutdown
(18 Jul–1 Aug 2016, noted in the Mendeley description) — a real
infrastructure disruption, not battery ageing. And even where real binding
is measured, **suppression during the constrained window is NOT
dramatically higher than during unconstrained periods** (h51: 0.048 W
constrained vs. 0.072 W unconstrained; h48: 0.0 W vs. 0.034 W) — see
`plots/song_kenya_h51.png` and `h48.png`. Plausible explanation: at this
tiny scale, depletion tends to follow the evening's lighting use rather
than pre-empt it, so there is little unmet "typical" demand left to detect
by the time SOC crosses the floor. **Report this exactly as found** — it is
a genuine, interesting, defensible result (a real capacity-binding site was
finally found; the estimator's response to it is modest, not dramatic),
not a result to spin as a stronger confirmation than it is.

**Technical notes** (full detail in `load_song_kenya.py`'s docstring):
- No direct battery SOC channel exists; SOC is Coulomb-counted from the
  dataset's own two half-hourly power files (lighting=demand, always ≥0;
  charging=hub supply, always ≤0 — sign convention verified against all
  27,840 raw values, zero exceptions), clipped to [0,100]% as an implicit
  re-anchor.
- Every household's file spans the same fixed 146-day grid regardless of
  how much real data exists, zero-filled for non-recorded periods with no
  NaN marker — verified and handled by restricting each household to its
  own active-recording window (counted nonzero-charging days directly;
  household 1 has 4, household 51 has 82).
- Same Algorithm 1 procedure, same code pattern as every other site
  (`train_song_kenya.py` mirrors `train_star.py`/`train_jharkhand.py`
  exactly) — no methodology change for this site either.
- Raw data copied to `C:\Users\Viranga\Downloads\solar\data\raw\song_kenya\`
  (102 CSVs, 3.5 MB, matching the existing raw-data folder convention).

This belongs in Chapter 3's Table 3.2 as a fifth row, and is a strong
addition to Chapter 6 Results / Chapter 7 — it directly answers your own
"Future Recommendations" §7.6, and should probably be its own short
subsection rather than folded silently into the "four deployments" framing,
since five independent sites with one genuinely confirmed-constrained is a
materially stronger and more interesting claim than four over-provisioned
ones alone.

## 3. Headline findings (all numbers traceable to `data/*.csv` / `results/*.csv`)

1. **All four datasets are over-provisioned, not three of four.** The
   dissertation draft currently says 3 of 4; Jharkhand's binding check
   (elevated to top priority per your own Sec. 7.6 "Future Recommendations")
   found all three houses <0.1% binding, same as Mafate/HEED/STAR. This is
   the single most important update for the Results/Discussion/Conclusion
   chapters.
2. **E4 placebo passes everywhere tested**: Mafate (3 houses, mean
   suppression 17.5–23.8 W) and HEED (0.00 W), both far under the 50 W
   tolerance, 0% false-alert rate. `data/e4_placebo_results.csv`,
   `data/heed_e4_placebo_results.csv`.
3. **E1 lead-time is a genuine mixed result** (`data/e1_lead_time_results.csv`):
   at 20%/40% injected capacity fade, the suppression method never crosses
   its 100 W alert threshold on Mafate house 1, while a naive CUSUM
   baseline eventually does (36–46 days later). At 60% fade, the
   suppression method detects in ~3 days vs. CUSUM's 21-day lag. The
   voltage-alarm baseline never fires at any fade level — which is itself
   the dissertation's central claim (conventional telemetry stays "normal").
   **Do not over-claim this as a clean win** — report both directions
   honestly.
4. **E2 hold-out**: MAE 44–109 W, CRPS 27–51 W across the three Mafate
   houses (`data/e2_holdout_results.csv`). House 1 needed τ=0.95 on this
   particular split vs. τ=0.5 in the main training run — a real
   split-sensitivity finding worth a line in Threats to Validity.
5. **E3 cross-site** (`results/table6_m5_cross_site_consistency.csv`):
   same Algorithm 1 applied unmodified to all 4 sites.
   - Mafate, HEED, Jharkhand: consistent small/near-zero suppression,
     matching each site's independently-known near-0% binding rate. This
     is a genuine positive replication result.
   - **STAR is unreliable and should be reported as such, not as a 4th
     confirmation.** Its appliance (IAM) data is a handful of day-clusters
     scattered across months, not continuous telemetry — confirmed visually
     in `plots/e3_star_*.png` (straight-line gaps of weeks between sparse
     real points) and numerically (site A's Q10–Q90 coverage is 10.7%
     against an ~80% target). This is a STAR data-coverage limitation,
     consistent with the dissertation's own existing "requires substantial
     cleaning... secondary site only" characterization — not a method flaw.

## 4. Corrections made to the generic/pasted task spec, all verified against
   raw files or the dissertation text (valuable material for Ch.5 "Datasets
   and Data Preparation" and Ch.7 "Problems Encountered")

- Mafate's per-house "main load" column is `Input4`/`Input5`/`Input1`
  (house 1/2/3 respectively), not a uniform `Input3` — verified against the
  dataset's own README.
- Mafate House 2's raw file uses a different date format
  (`DD/MM/YYYY HH:MM`) than Houses 1/3 (`YYYY-MM-DD HH:MM:SS`) — a silent
  mismatch here was found to collapse 512k rows to 1 via a duplicate-NaT bug.
- Solar_plant power columns are in kW (not W); Meteo irradiance is in
  kW/m² — both need ×1000 conversion, per the Mafate README.
- **The suppression counterfactual uses a data-driven selected upper
  quantile τ (Algorithm 1, Sec. 5.2.2), not the median** — the single
  biggest correction vs. the generic pasted spec, which assumed Q50.
  τ-selection logic: `train_model.py::select_tau`.
- A real quantile-crossing bug was found and fixed late in the project:
  Q10 and Q90 are fit independently and can cross (Q90 went negative while
  Q10 sat at 0 early in the Mafate record). Fixed via monotone rearrangement
  (`train_model.py::rearrange_quantile_pair`, applied in every training
  script). Does not affect suppression itself (which uses Qτ only) — only
  the descriptive FR-05 uncertainty interval.
- One-hot encoding for Hour/DayOfWeek (not raw integers — avoids forcing a
  monotonic daily pattern); Month represented as sin/cos + a linear trend
  term (not one-hot) — both needed because Mafate's documented weather-gap
  and the chronological train/test splits leave some calendar months absent
  from training, which a one-hot Month handles badly.
- Jharkhand has **no direct load-current channel at all** (verified against
  the supplied MATLAB calibration script — the documentation's channel
  labels were wrong). Load is *derived* via energy balance
  (`train_jharkhand.py`), including a PV-current sign-convention bug caught
  by checking that PV power can't be negative at solar noon.
- Jharkhand's real battery voltage is channel #3 (~12 V), not channel #1
  (which is PV voltage, 0–20 V) — the opposite of the generic prompt's
  assumption.
- STAR's Battery "Power" field is charging-power-only (never negative,
  `DischargeEnergy` is flat for days) — true household load required
  discovering and loading STAR's separate IAM (Individual Appliance
  Monitor) files, which the generic spec never mentioned.
- STAR/Jharkhand both needed sentinel-timestamp cleaning (Unix-epoch
  1969/1970 rows in STAR; implausible near-0V readings in Jharkhand).

## 5. Known limitations — report these honestly, don't paper over them

- **NFR-05 ("completes within minutes") is not met for model training.**
  τ-selection grid search takes 10–20 min per site. It IS met for the
  operational/scoring workload (verified in `tests/test_tc08_performance.py`:
  scoring a full site-year against an already-trained model is well under a
  minute). State this distinction explicitly if asked about NFR-05.
- STAR's E3 results are present but low-confidence — see §3 point 5.
- Jharkhand's load signal is a derived proxy (noisy, low ADC resolution:
  ~354–509 raw count span per the dissertation's own Table 3.2), not a
  direct meter — medium, not high, confidence.
- No suppression model was built for STAR/Jharkhand's *binding-check-only*
  predecessors — superseded by `train_star.py`/`train_jharkhand.py`, which
  now do have models; this note is just to avoid confusion if you see
  references to the earlier binding-only state in old notes.

## 6. Dashboard

`streamlit run dashboard.py` (or `python -m streamlit run dashboard.py`).
Verified end-to-end in a real headless browser (Playwright) — not just an
HTTP check. One real bug was found and fixed: charts were passed the full
474,603-row dataset directly to client-side Vega-Lite rendering, which froze
the browser; fixed with `downsample_for_chart()` (display-only, does not
touch any computed result).

## 6b. A real bug found and fixed late (worth a line in Ch.7 "Problems
   Encountered" — it's a good example of exactly the kind of thing that
   section already narrates for Mafate)

Writing the formal test suite caught a genuine quantile-crossing bug: Q10
and Q90 are fit as independent models (separate pinball-loss minimisations),
which does not guarantee Q10 <= Q90 pointwise. Verified concretely on
Mafate house 1: Q90 predicted negative Watts early in the record while Q10
sat at 0. Does NOT affect the core suppression result (Eq. 5.3 only uses
Qtau), only the descriptive FR-05 uncertainty interval. Fixed via monotone
rearrangement (Chernozhukov, Fernandez-Val & Galichon, 2010) in
`train_model.py::rearrange_quantile_pair`, applied consistently in every
training script. All model files and result CSVs in this handoff already
reflect the fix (retrained after the fix was added) — this note is for the
dissertation narrative, not a pending action.

## 7. Formal test suite

`tests/` — pytest, 23 tests across `test_tc01_*.py` through `test_tc10_*.py`,
one file per test case in your own Chapter 6 Table 6.2. Run with:
```
cd code && python -m pytest tests/ -v
```
All 23 passing as of this handoff. Covers: loader schema (TC-01), SOC
reconstruction (TC-02), sentinel handling (TC-03), binding classification
(TC-04), thin-evidence low-confidence behaviour (TC-05), injection harness
(TC-06), baseline-detector shared interface (TC-07), performance (TC-08),
seed reproducibility (TC-09), alert evidence content (TC-10).

## 8. Exactly what remains for the dissertation document

Nothing in `nibm-thesis-latex/` has been edited. Still needed:

1. **Abstract** — replace "Evaluation results are not yet available."
2. **Ch.5, Table 5.2** "Implementation status by component" — mark components implemented.
3. **Ch.6, §6.6 "Results"** — replace `\pending{}` with the real M1–M5 numbers
   (see §3 above and `results/table1..6_*.csv`).
4. **Ch.6, §6.7 "Discussion"** — replace `\pending{}`; must include the honest
   mixed E1 finding and the STAR reliability caveat, not just the positive results.
5. **Ch.7, Table 7.1** triangulation (RO1–RO4 status) + explicitly update the
   "3 of 4 sites" framing to 4 of 4 (Jharkhand finding).
6. Consider adding the quantile-crossing bug and the STAR IAM-sparsity
   discovery to **§7.3 "Problems Encountered"** — both are genuine,
   well-documented examples of exactly the kind of "feasibility testing
   caught a real issue" narrative the dissertation already uses for Mafate.

## 8b. Precise locations in the LaTeX source (read directly, exact as of this handoff)

Every `\pending{...}` and every "three of four" needs attention. Exact spots found:

**`frontmatter/abstract.tex`**: last sentence of paragraph 3, "Evaluation results
are not yet available." — replace with the headline M1–M5 numbers once Ch.6
§6.6 has them (per the file's own comment: Ch.6 first, then mirror here).

**`chapters/ch5-implementation.tex`**:
- Line ~175, end of the Jharkhand paragraph in §5.4: `\pending{Jharkhand
  binding check not yet run; the primary site cannot be finally confirmed
  until it is.}` — delete; Jharkhand is run (`data/jharkhand_binding_rates.csv`,
  all 3 houses <0.1% binding, so it is NOT the naturally-constrained primary
  site either — same "over-provisioned" conclusion as the other three).
- `tab:implstatus` (§5.5, Table 5.2) — every row needs updating:
  - "Loaders for HEED, STAR, Jharkhand": `\pending{not yet built}` → Existing
    (`load_heed.py`, `load_star.py`, `jharkhand_check.py`)
  - "State of charge reconstruction": "Partial... required for Jharkhand" →
    Mafate: Coulomb-count reconstruction implemented and validated
    (`soc_reconstruction.py`, TC-02 passes). Jharkhand/HEED/STAR: NOT
    reconstructed via Coulomb counting — they have a direct voltage/SOC
    channel instead (Algorithm 1's own precondition is "if state of charge
    not measured"), so reconstruction is correctly not needed there; say
    this explicitly rather than leaving it looking incomplete.
  - "Suppression estimator": `\pending{not yet built}` → Existing
    (`train_model.py`, `suppression.py`, Eq. 5.3 implemented exactly)
  - "Fade injection harness": `\pending{not yet built}` → Existing (`injection.py`)
  - "Baseline detectors": `\pending{not yet built}` → Existing
    (`e1_lead_time.py`: voltage alarm + CUSUM, sharing one interface per
    Fig. 4.3, verified in TC-07)
  - "Monitor and alert layer": `\pending{not yet built}` → Existing (`monitor.py`)
  - "Dispatch advisory": still genuinely not built — leave as pending, or
    cut it from scope if you decide FR-10 ("Should", not "Must") isn't worth
    building this late. Recommend cutting rather than leaving a stray pending
    row this close to submission.
  - "Mafate loader & Existing, defective": the described defect (meteorological
    parsing) is fixed — update to "Existing" with a note, or fold into the
    general loader row.
- §5.6 Source Code Availability `\pending{...}`: decide now whether the
  GitHub repo will actually be made public and pushed to before submission —
  this is a logistics action item, not something I can do for you.

**`chapters/ch6-testing-evaluation.tex`**:
- §6.6 Results (`\pending{No experiment has been run...}`) — replace with
  M1–M5 from §3 above. The chapter already pre-states two expectations to
  check the outcome against (line ~195-199): "binding rate M4 expected at/near
  zero at Mafate and low at both HEED sites" — CONFIRMED, and now also true at
  STAR/Jharkhand, strengthening the point; "weight of positive evidence for M1
  expected to rest on injection rather than natural suppression" — CONFIRMED,
  E1's natural-data evidence is thin, injection carries the finding.
- §6.7 Discussion (`\pending{To be written once Section 6.6 contains
  results.}`) — the chapter already names the exact issue to address (line
  ~205-213): framing a near-zero result as over-provisioning evidence, not a
  disappointment, contingent on E4 passing. E4 passes (§3 point 2) — write
  this up, and add the E1 mixed-result honesty and the STAR caveat here too.

**`chapters/ch7-conclusion.tex`**:
- `tab:triangulation` (Table 7.1): RO2 "implementation not yet built" → built
  (point to Algorithm 1's code, §2 table above). RO3 "not yet run" → run
  (E1 exists, mixed result, §3 point 3). RO4 "experiment E4 not yet run" →
  run, PASS (§3 point 2). All four should move to "Achieved" except possibly
  RO3, which you may want to call "Achieved (mixed result)" rather than a
  clean pass, given the honest E1 finding — your call, not mine.
- Line ~122-123 and ~163-164: **"three of four independent deployments"**
  appears twice — both need to become "all four" (Jharkhand finding, §3
  point 1). This is the single most-repeated stale claim in the document.
- §7.6 Future Recommendations, first bullet (line ~129-132): "The immediate
  next step is the Jharkhand binding check... determines whether the study
  has a naturally constrained primary site" — this is DONE, and the answer is
  no (Jharkhand also over-provisioned). Rewrite this bullet entirely rather
  than deleting it — the fact that this was the top recommendation and was
  then completed is worth narrating, not erasing.
- §7.7 Conclusion, line ~170-174: "What remains is the experimental work...
  no result is claimed here that an experiment has not produced" — this
  entire closing paragraph describes the OLD state (nothing built/run) and
  needs a full rewrite now that the experimental work exists.

## 8bb. SONG Kenya citation — exact entries to add

**`references.bib`**, add (matching the existing dataset-citation format,
e.g. `mehra2016jharkhand` right below it):
```
@online{clements2019song,
  author       = {Clements, Anna F. and McCulloch, Malcolm},
  title        = {Time series lighting electricity data for rural households
                  using Solar Nano-grids in {Kenya}},
  year         = {2019},
  publisher    = {Mendeley Data},
  version      = {V1},
  doi          = {10.17632/4yv37hngp6.1},
  note         = {{CC BY 4.0}. Methodology described in Clements, A.F.,
                  "Data-driven approaches enabling the design of community
                  energy systems in the Global South", DPhil thesis,
                  University of Oxford, 2019},
  urldate      = {2026-10-02}
}
```

**`appendices/appendix-a.tex`**, `tab:provenance` (§A.5) — add a row after
the Jharkhand row (before `\bottomrule`):
```
\addlinespace
SONG Kenya & Solar Nano-grids project, Kenya \parencite{clements2019song} &
Fifty-one households, central-hub battery charging, {CC BY 4.0}; the only
site in this study with directly measured (non-injected) capacity binding\\
```

## 8c. Chapter 3 also needs updating (found late, don't miss it)

**`chapters/ch3-methodology.tex`**, `tab:datasets` (Table 3.2):
- Jharkhand row (line ~145-147): "Not yet determined. Requires the supplied
  calibration for the raw channels" — replace with the real result:
  binding rate 0.0%/0.0%/0.0726% across Houses 2/3/4
  (`data/jharkhand_binding_rates.csv`), min/mean/max battery voltage per
  house also in that file. Note for accuracy: no MATLAB calibration was
  actually needed — the raw files already carry pre-calibrated current
  columns (§4 above) — so "requires the supplied calibration" should be
  corrected, not just have its result filled in.
- Line ~152-153: **another "Three of the four deployments are
  systematically over-provisioned"** — same fix as Ch.7, → "all four".

**Gap worth knowing about, PARTIALLY attempted**: Table 3.2 also lists
**HEED Microgrid** (849k rows) as a separate site from HEED Hall — the main
pipeline only has a model for HEED **Hall**. I did attempt Microgrid
late in this session (`load_heed.py` now has `load_multi_cpe_appliance_load()`
and `build_microgrid()`, generalized from the Hall functions, confirmed NOT
to have broken Hall's existing working pipeline — regression-checked
directly). Findings from the attempt:
- Appliance load DOES aggregate correctly across all 17 of Microgrid's CPE
  folders (6 "Nur" nursery buildings + a Playground, Streetlights excluded)
  — community-scale magnitude (mean ~2.6 kW, up to 40 kW) is plausible for
  a multi-building site.
- One real bug was found and fixed in passing: some Microgrid CPE files
  (not present in Hall CPE1) contain rows with a missing sensor name, which
  crashed the `_p`-suffix filter; fixed with `na=False` on the `endswith`
  mask (`load_heed.py::_load_one_cpe`).
- **Blocking issue, NOT resolved**: the battery SOC join returns zero
  non-null rows for Microgrid (`load_system_data(MICROGRID_SYSTEM_DATA_DIR)`),
  so `unconstrained` classification can't run. The Victron system-export
  column-matching logic that works for Hall (`SOC_COLUMN =
  "Battery Monitor [258] | State of charge"`) likely needs a different
  device-ID or column structure for Microgrid's own export — not diagnosed
  due to time (loading alone took ~10 minutes given 17 CPEs' full file
  counts, before even reaching this bug).
- **Recommendation if you want to finish this**: inspect one Microgrid
  `System data/*.csv` file's device/field header rows directly (same method
  used earlier in this project for Hall — print `device_row`/`field_row`
  after `.ffill()` and search for "charge"/"Voltage") to find the correct
  column name(s), then retry `load_heed.build_microgrid()`.

Not required for the dissertation's existing claims: "four independent
deployments" in the abstract/RQ already treats HEED as one deployment
(Hall), so this remains a strengthening opportunity, not a gap in what's
currently claimed.

## 9. Where everything lives

- Code: `solar2/code/*.py`
- Trained models: `solar2/code/models/*.pkl`
- Processed data + all result CSVs: `solar2/code/data/`
- Final aggregated tables for the dissertation: `solar2/code/results/table1..6_*.csv`
- Plots: `solar2/code/plots/*.png`
- Tests: `solar2/code/tests/`
- Dissertation LaTeX source: `solar2/nibm-thesis-latex/` (unedited)
