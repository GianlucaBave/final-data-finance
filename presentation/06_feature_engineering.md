# Slide 6 — Feature Engineering

After the traps were removed and the legacy formats normalised, we built derived features that capture **credit-domain intuition** and **data-quality signals**.

## Engineered features

| Feature | Definition | Why |
|---|---|---|
| `total_income` | `ann_income + other_income` | Real disposable income |
| `log_income` | `log1p(total_income)` | Compress 0 → $9M range |
| `amt_to_income` | `amount / (total_income + 1)` | Standard loan affordability metric |
| `risk12` | coalesce(`risk_indicator_1`, `risk_indicator_2`) | r1/r2 mutually exclusive (H5) |
| `risk12_dev`, `risk3_dev` | `\|risk − 55\|` | Distance from neutral is more informative than the raw score |
| `risk_max`, `risk_mean` | aggregates of `risk12` & `risk3` | Robust risk signal |
| `credit_z` | z-score per bureau, then coalesce | Compare across FICO / Vantage / SCHUFA |
| `has_score` | any credit score available | Missingness as signal |
| `ann_income_missing`, `other_income_missing`, `prev_default_missing` | binary flags | Missingness as signal |
| `year`, `month` | parsed from `date` | Temporal trend (e.g. tightening in 2026) |
| `age` (imputed) | from `birth_year + year` when missing | Recover ~5% of rows |

## What we did NOT do (and why)

- **No target encoding on `analyst_opinion`** → would leak when fitted on the wrong fold. Native categorical handling (LightGBM / CatBoost) was already strong.
- **No external data** — competition rules.
- **No race/religion proxies** — slide 7.

## Top features by importance (LightGBM, gain)

`analyst_opinion` ≫ `amt_to_income` > `risk12_dev` > `age` > `risk_mean` > `amount` > `risk3_dev` > `risk12` > `total_income` > `risk_max`

`analyst_opinion` alone has nearly **2× the gain** of the next feature — H6 was correct.

---

### Speaker notes
End by surfacing the feature-importance chart: free text dominates. That's the headline.

### Assets to add
- `assets/figures/06_feature_importance_top10.png`
