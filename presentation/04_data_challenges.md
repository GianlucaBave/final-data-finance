# Slide 4 — Data Challenges (Bottlenecks)

> The data dictionary warns us upfront: *"You should expect some level of inconsistency of formats, units and categorical encodings, as well as missing values and sparse fields."*

This slide shows the **concrete issues** we found during EDA.

## Issues found

| # | Column(s) | Issue | Evidence |
|---|---|---|---|
| C1 | `date` | Mixed formats (`Apr-2026`, `2022-06-08`, ...) | _TBD: % of each format_ |
| C2 | `prev_default` | Inconsistent encodings (`Yes`, `0`, blank, ...) | _TBD: value counts_ |
| C3 | `ann_income`, `other_income` | String-like numerics | _TBD: example bad rows_ |
| C4 | `cr_scores_fico` / `vantage` / `schufa` | Mostly missing | _TBD: % missing each_ |
| C5 | `external_pd_score` | High missingness | _TBD_ |
| C6 | `age` vs `birth_year` vs `date` | Possible inconsistencies | _TBD: count mismatches_ |
| C7 | `risk_indicator_1/2/3` | Source unknown, consistency unclear | _TBD: pairwise plots_ |
| C8 | `analyst_opinion` | Unstructured free text | _TBD: avg length, examples_ |
| C9 | `religion`, `race` | Synthetic but ethically sensitive | See Slide 7 |

## Headline figures to fill from EDA

- Overall missingness: _TBD %_
- Most affected columns: _TBD_
- Target class balance: _TBD % approved vs rejected_

---

### Speaker notes
This is one of the most important slides. Lead with **"these are the bumps we had to handle to even start modelling"** — frames everything that follows.

### Assets to add
- `assets/figures/04_missing_values_heatmap.png`
- `assets/figures/04_target_balance.png`
- `assets/figures/04_date_format_distribution.png`
