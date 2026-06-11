# Slide 5 — Data Cleaning & Preprocessing

For each issue in Slide 4, we document the fix here.

## Problem → Solution

| ID | Problem | Our Fix | Notebook ref |
|---|---|---|---|
| C1 | `date` mixed formats | _TBD_ | `02_cleaning.ipynb` |
| C2 | `prev_default` inconsistent | _TBD_ | `02_cleaning.ipynb` |
| C3 | String-like incomes | _TBD_ | `02_cleaning.ipynb` |
| C4 | Missing credit scores | _TBD_ | `02_cleaning.ipynb` |
| C5 | Missing `external_pd_score` | _TBD_ | `02_cleaning.ipynb` |
| C6 | Age / birth_year / date mismatches | _TBD_ | `02_cleaning.ipynb` |
| C7 | Risk indicators inconsistent | _TBD_ | `02_cleaning.ipynb` |
| C8 | `analyst_opinion` text | _TBD_ | `02_cleaning.ipynb` |
| C9 | `religion`, `race` | Dropped (see Slide 7) | `02_cleaning.ipynb` |

## Pipeline overview (draft)

```
raw CSV
  → schema check
  → date normalization
  → numeric parsing (incomes)
  → categorical encoding (prev_default, status, job_category, ...)
  → missing-value strategy (imputation + missingness flags)
  → drop ethically-sensitive columns
  → clean dataset
```

---

### Speaker notes
Pair this slide with Slide 4 visually (same row IDs C1, C2, ...). Audience should map every problem to a fix.

### Assets to add
- `assets/figures/05_pipeline_diagram.png` (optional)
