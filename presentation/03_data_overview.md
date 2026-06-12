# Slide 3 — The Data

## Size

| Set | Rows | Has target? |
|---|---|---|
| Train | 25,000 | ✅ |
| Test | 5,000 | ❌ (predict this) |

## Feature groups (24 features + target)

| Group | Columns | Notes |
|---|---|---|
| **Identifiers** | `id`, `date` | Date in mixed formats |
| **Demographics** | `age`, `birth_year`, `status`, `kids`, `highest_ed` | Synthetic codes also present (excluded — see Slide 7) |
| **Employment & income** | `job_category`, `ann_income`, `other_income` | Income string-like |
| **Credit history** | `prev_default`, `external_pd_score`, `cr_scores_fico`, `cr_scores_vantage`, `cr_scores_schufa` | Mostly missing scores |
| **Loan request** | `amount`, `vip` | |
| **Risk aggregators** | `risk_indicator_1`, `risk_indicator_2`, `risk_indicator_3` | Source unknown, check consistency |
| **Other** | `internal_code`, `analyst_opinion` (free text) | |
| **Target** | `credit_decision` | Binary |

## Source

> *"The dataset has been assembled from multiple legacy source systems."*

Expect inconsistencies — covered in detail on Slide 4.

---

### Speaker notes
Keep this slide visual: one table, no walls of text. Mention that some columns are *less reliable than others* — a hint we'll act on later.
