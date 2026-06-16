# Slide 3 — The Data

## Size

| Set | Rows | Period | Has target? |
|---|---|---|---|
| Train | 25,000 | 2022-01 → 2026-04 | ✅ |
| Test | 5,000 | 2026-05 only | ❌ (predict this) |

**Target balance:** 51.98% approved / 48.02% rejected — nearly balanced, so a majority-class guess only scores ~0.52. Every accuracy point is earned.

**Leaderboard split:** public = 40% of test (~2,000 rows), private = 60% (~3,000 rows). Only the private score counts for grading (Slides 9–10).

## Feature groups (24 features + target)

| Group | Columns | Notes |
|---|---|---|
| **Identifiers** | `id`, `date` | Date in mixed formats |
| **Demographics** | `age`, `birth_year`, `status`, `kids`, `highest_ed` | Synthetic codes also present (excluded — see Slide 7) |
| **Employment & income** | `job_category`, `ann_income`, `other_income` | Income string-like |
| **Credit history** | `prev_default`, `external_pd_score`, `cr_scores_fico`, `cr_scores_vantage`, `cr_scores_schufa` | Mostly missing scores |
| **Loan request** | `amount`, `vip` | |
| **Risk aggregators** | `risk_indicator_1`, `risk_indicator_2`, `risk_indicator_3` | Source unknown — `r1` and `r2` turned out identical (Slide 4) |
| **Other** | `internal_code`, `analyst_opinion` (free text) | `internal_code` is a planted leak (Slide 4); `analyst_opinion` = 60 template sentences |
| **Target** | `credit_decision` | Binary |

## Source

> *"The dataset has been assembled from multiple legacy source systems. Do not assume that every column is equally reliable or equally appropriate for modelling."*

This warning turned out to be literal: the data contains **deliberate traps** — a column that looks perfect in train but is scrambled in test, a column that vanishes in test, mixed units, duplicate columns, and protected attributes. Detailed in Slide 4.

---

### Speaker notes
Keep this slide visual: one table, no walls of text. The key takeaway: the organizers explicitly told us not to trust every column equally — so our first phase was **forensics, not modelling**. We verified each column against the test set before trusting it.
