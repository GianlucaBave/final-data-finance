# Slide 5 — Data Cleaning & Preprocessing

Every fix lives in one shared module — **`src/prep.py`** — so train and test are cleaned identically (no leakage, fully reproducible). Notebook walk-through: `notebooks/credit_decision_pipeline.ipynb` §3–§5.

## Problem → Solution (mirrors Slide 4 IDs)

| ID | Problem | Our Fix | Code |
|---|---|---|---|
| C1 | `internal_code` planted leak | **Dropped** — `DROP_TAMPERED` | `prep.py` |
| C2 | `external_pd_score` absent in test | **Dropped** — `DROP_TEST_MISSING` | `prep.py` |
| C3 | Incomes in mixed units | `np.where(inc < 700, inc*1000, inc)`; same for nonzero `other_income` | `engineer()` |
| C4 | `date` two formats | Parse ISO and `%b-%Y` separately, then combine (`%b-%Y` dated mid-month) | `parse_dates()` |
| C5 | `prev_default` encodings | Map `{0,No}→0`, `{1,Yes}→1`, blank→NaN (kept; trees handle missing) | `engineer()` |
| C6 | `age` missing (~10%) | Impute from `year − birth_year` (verified consistent: median mismatch = 0, max = 1 yr) | `engineer()` |
| C7 | `risk_indicator_1 ≡ risk_indicator_2` | **Coalesce** into `risk12 = r1.fillna(r2)`; keep `risk3` | `engineer()` |
| C8 | Credit scores, 3 scales, ~95% missing | **z-score each bureau** on combined train+test, then coalesce → `credit_z` | `add_scores()` |
| C9 | `analyst_opinion` free text | Keep as a **categorical** (60 fixed templates) — no fragile keyword parsing needed | `build()` |
| C10 | `religion`, `race` | **Dropped** — `DROP_PROTECTED` (Slide 7) | `prep.py` |

## Pipeline (as implemented)

```
raw train.csv / test.csv
  → parse_dates()        # ISO + "Mon-YYYY" → year, month
  → engineer()           # unit fixes, age imputation, prev_default unify,
  │                      # risk coalesce + shape, income ratios
  → add_scores()         # per-bureau z-score → single credit_z
  → categoricals         # job_category, status, analyst_opinion (native, no one-hot)
  → drop {internal_code, external_pd_score, religion, race, id, birth_year}
  → model-ready X (20 features)
```

## Why native categoricals, not one-hot / target encoding

LightGBM and CatBoost handle categories and missing values natively. This avoids one-hot's 60-column blow-up on `analyst_opinion` and sidesteps target-encoding leakage. CatBoost's ordered target statistics handle the high-cardinality opinion column especially well — it became our strongest single model (Slide 8).

---

### Speaker notes
Pair this slide visually with Slide 4 (same IDs). One line that lands well: *"We deleted the two most 'predictive' columns in the dataset on purpose — because their predictiveness was fake."* Emphasize that one `prep.py` is the single source of truth, so the cleaning is identical for train and test and the whole thing reruns end-to-end.

### Assets to add
- `assets/figures/05_pipeline_diagram.png`
- `assets/figures/05_income_before_after.png` (income distribution before/after the ×1000 fix)
