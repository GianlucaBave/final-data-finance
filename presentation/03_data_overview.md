# Slide 3 — The Data

## Size

| Set | Rows | Target |
|---|---|---|
| Train | 25,000 | 52% approved / 48% rejected |
| Test | 5,000 | hidden (predict this) |

Public leaderboard = 40% of test (2,000 rows). **Private leaderboard = 60% (3,000 rows)** — that's the one that counts.

## Time

- Train spans 2022-01 → 2026-04
- **Test is entirely May 2026** → out-of-time evaluation, distribution shift expected

## Feature groups (24 features + target)

| Group | Columns | Notes |
|---|---|---|
| Identifiers | `id`, `date` | Date in 2 different formats |
| Demographics | `age`, `birth_year`, `status`, `kids`, `highest_ed` | + synthetic codes (excluded — see Slide 7) |
| Employment & income | `job_category`, `ann_income`, `other_income` | Income parsed as numeric |
| Credit bureau | `cr_scores_fico`, `cr_scores_vantage`, `cr_scores_schufa` | ~50–80% missing each |
| External risk | `external_pd_score` | 100% missing in test (Slide 5) |
| Loan request | `amount`, `vip` | |
| Risk aggregators | `risk_indicator_1/2/3` | r1/r2 mutually exclusive |
| Other | `internal_code`, `analyst_opinion` (free text), `prev_default` | |
| Target | `credit_decision` | Binary |

---

### Speaker notes
Stress *out-of-time* test. This dictates the choice of rolling-month validation later.
