# solarguard-adequacy

**Deprivation-Aware Monitoring for Off-Grid Solar — measuring the household demand a solar-battery
system suppresses.**

BSc final-year dissertation research. Novelty search, adversarial verification, and data-feasibility
testing for a system that detects when an off-grid solar installation has begun failing its users
while every conventional monitoring signal still reads green.

---

## The idea in one sentence

> A system that can tell when an off-grid solar system is quietly failing its users — even though
> every normal check says it is fine — by measuring how much electricity people are silently going
> without, instead of only checking the equipment.

## The problem

When an off-grid battery weakens, households don't wait for a blackout — they quietly use less.
Because the user absorbs the shortfall, generation looks weather-normal, voltages sit at nominal,
the energy balance balances, and lower load reads as a *customer choice*. **Nothing alarms.**
The failure is invisible precisely because the human hides it.

## The gap

The sector has a standard metric for energy a system failed to deliver (*unserved energy* / EENS),
but it can only be **estimated at design time from surveys** — surveys the sector's own reference
admits can be wrong by up to 50%. It has never been measurable on a deployed system, because the
required input (true demand) is exactly what the constraint censors. You only observe
`min(demand, available)`.

## Contribution

A **new task**, not a new method — the only formulation that survives the constraint that novelty
must not be replicable by a human manually or by an existing method:

1. **Task formalisation** — per-household, per-interval *suppression* and *binding rate*
2. **Estimator** — matched-condition counterfactual from routinely-collected telemetry
3. **Working system** — monitor + alert + suppression-minimising controller

---

## Repository contents

| File | Purpose |
|---|---|
| [`00-CONSTRAINTS.md`](00-CONSTRAINTS.md) | Supervisor's rules, formalised as pass/fail tests |
| [`01-DOMAIN-MAP.md`](01-DOMAIN-MAP.md) | Systematic map of the energy research space |
| [`02-CANDIDATES.md`](02-CANDIDATES.md) | Candidate topics generated and scored |
| [`03-ADVERSARIAL-LOG.md`](03-ADVERSARIAL-LOG.md) | Every attack attempted against each candidate, with evidence |
| [`04-RESEARCH-BRIEF.md`](04-RESEARCH-BRIEF.md) | Full research brief: task, prior art, evaluation, risks, 12-week plan |
| [`05-FINAL-ANSWER.md`](05-FINAL-ANSWER.md) | Final recommendation with the empirical data correction |
| [`06-FULL-PUBLICATION-SWEEP.md`](06-FULL-PUBLICATION-SWEEP.md) | 10-channel sweep beyond ranked journals (patents, NGO, industry, theses, OSS) |
| [`07-DATA-SUFFICIENCY.md`](07-DATA-SUFFICIENCY.md) | Measured verdict on all four datasets — opened and tested, not assumed |
| [`08-FORECAST-ADVISORY-CHECK.md`](08-FORECAST-ADVISORY-CHECK.md) | Adversarial check killing the forecast-advisory framing |
| [`09-SUMMARY-SIMPLE.md`](09-SUMMARY-SIMPLE.md) | Plain-English summary for supervisor discussion |

---

## Key empirical findings

All four datasets were opened and tested directly, not taken on documentation.

| Site | Span | Binds? |
|---|---|---|
| **Mafate** (Réunion) | 12 mo, 474k rows | ❌ **Never** — voltage floor 48.50 V (LVD ≈ 44–46), 5.9 days autonomy, PV:load 2.3× |
| **HEED Hall** (Rwanda) | 21 mo | ⚠️ Barely — SOC min 37.4%, **0 full discharges** |
| **HEED Microgrid** (Rwanda) | 21 mo | ⚠️ Barely — SOC min 49.5%, **0 full discharges** |
| **STAR A–D** (Brazil) | 38–187 d/site | ⚠️ Sparse, poor data quality (sentinel values) |
| **Jharkhand** (India) | 230 d | ❓ Unchecked — needs MATLAB calibration |

**Headline:** three of four sites are systematically oversized and essentially never constrained.
This is a structural property of the field — open research deployments are over-engineered by design,
while genuinely constrained systems are commercial and proprietary.

The estimator is **symmetric**, so this becomes a result rather than a limitation:

- counterfactual ≈ observed → never bound → **system oversized** (capital that could have electrified 2–3× more households)
- counterfactual > observed → **demand suppressed** → system failing its users

---

## Status

Research planning and feasibility verification complete. Implementation pending.

**Next step:** run the supplied MATLAB calibration on the Jharkhand channels — the smallest, cheapest
system available and the most likely to actually bind.

## Licence

Research documentation. Datasets referenced are third-party; see individual sources for their
licences (Mafate, HEED, STAR are CC BY 4.0).
