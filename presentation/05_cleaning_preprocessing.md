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
| C8 | Credit scores, 3 scales, ~95% missing | **z-score each bureau on TRAIN-ONLY stats** (no test leak), then coalesce → `credit_z` | `add_scores_fixed()` |
| C11 | Structured missingness | Added explicit flags: `ann_income_missing`, `other_income_missing`, `prev_default_missing` | `prep_v2.engineer()` |
| C9 | `analyst_opinion` free text | Keep as a **categorical** (60 fixed templates) — no fragile keyword parsing needed | `build()` |
| C10 | `religion`, `race` | **Dropped** — `DROP_PROTECTED` (Slide 7) | `prep.py` |

## The actual fixes (excerpts from `src/prep.py`)

```python
def parse_dates(s):                                  # C4: two date formats
    iso = pd.to_datetime(s, format='%Y-%m-%d', errors='coerce')
    mon = pd.to_datetime(s, format='%b-%Y', errors='coerce') + pd.Timedelta(days=14)
    return iso.fillna(mon)

# C3: incomes — half the rows are in thousands (bimodal, empty gap 700..2000)
inc = pd.to_numeric(df['ann_income'], errors='coerce')
out['ann_income'] = np.where(inc < 700, inc * 1000, inc)

# C5/C6: unify prev_default; impute missing age from a verified-consistent identity
out['prev_default'] = df['prev_default'].map({'0':0,'No':0,'1':1,'Yes':1}).astype(float)
out['age'] = df['age'].fillna(out['year'] - df['birth_year'])

# C7: risk_indicator_1 and _2 are the same variable (corr=1.0) → coalesce
out['risk12'] = df['risk_indicator_1'].fillna(df['risk_indicator_2'])
```

```python
# C8: three bureau scores, different scales → z-score each (TRAIN-ONLY stats), then coalesce
for c in ['cr_scores_fico','cr_scores_vantage','cr_scores_schufa']:
    mu, sd = tr[c].mean(), tr[c].std()          # train only — no test peeking
    tr[c+'_z'], te[c+'_z'] = (tr[c]-mu)/sd, (te[c]-mu)/sd
out['credit_z'] = (raw['cr_scores_fico_z']
                   .fillna(raw['cr_scores_vantage_z'])
                   .fillna(raw['cr_scores_schufa_z']))
```

> **Leakage fix (v2):** the earlier version computed these stats on train+test combined — a mild leak. For tree models the numeric effect is ≈ 0 (trees split on rank, and a z-score is monotonic), but train-only is the correct, reportable choice.

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
