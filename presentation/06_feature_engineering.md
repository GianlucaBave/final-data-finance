# Slide 6 — Feature Engineering

The final feature set is **20 columns** (down from 24 raw, after dropping the 4 traps). All built in `src/prep.py`; the risk-shape features are added in the model scripts.

## Features we built (and why)

| Feature | Definition | Rationale | Built? |
|---|---|---|---|
| `total_income` | `ann_income + other_income` (both unit-fixed) | Real disposable income | ✅ |
| `amt_to_income` | `amount / (total_income + 1)` | Affordability ratio — standard credit metric | ✅ |
| `log_income` | `log1p(total_income)` | Tame the long tail | ✅ |
| `age` (imputed) | `age.fillna(year − birth_year)` | Recover ~10% missing ages from a verified-consistent identity | ✅ |
| `risk12` | `risk_indicator_1.fillna(risk_indicator_2)` | The two are the same variable (corr = 1.0) | ✅ |
| `risk12_dev`, `risk3_dev` | `|risk − 55|` | Encodes the **inverted-U**: risk is worst at the extremes | ✅ |
| `risk_max`, `risk_mean` | max / mean of `risk12` & `risk3` | Combine the two independent risk sources | ✅ |
| `credit_z` | per-bureau z-scored FICO/Vantage/SCHUFA, coalesced | One comparable credit score from 3 incompatible scales | ✅ |
| `has_score` | any bureau score present? | Missingness as signal | ✅ |
| `prev_default` | unified {0,1,NaN} | 54% approval without prior default vs ~7% with | ✅ |
| `year`, `month` | parsed from `date` | Capture the downward approval trend over 2022→2026 | ✅ |
| `vip` | `{True,False}→{1,0}` | VIPs approved 71% vs 51% baseline | ✅ |
| `analyst_opinion` | native categorical (60 templates) | The analyst's own verdict — strongest legit feature | ✅ |
| `job_category`, `status` | native categoricals | Employment stability drives approval (27%→79% across categories) | ✅ |
| `highest_ed`, `kids` | kept as-is (ordinal / count) | — | ✅ |

## A note on the `analyst_opinion` "NLP" feature

We **did not** need bag-of-words, TF-IDF, or keyword counting. The column is exactly **60 fixed template sentences**, identical in train and test, falling into 3 clear sentiment tiers:

- **Negative** (~0.28–0.36 approval) — "Several credit-risk indicators are unfavourable."
- **Neutral** (~0.74–0.84) — "Overall risk profile appears moderate."
- **Positive** (~0.93–1.00) — "No material credit concerns identified in the review."

Treating it as a single categorical lets the model learn each template's exact approval rate — cleaner and stronger than any hand-built text feature would be.

## Encoding choices

- **Categoricals:** native handling in LightGBM/CatBoost (no one-hot, no target-encoding leakage).
- **Booleans (`vip`):** cast to 0/1.
- **`highest_ed`:** kept ordinal (1–5).
- **Missing values:** left as NaN for the trees to split on (they handle it natively); only `age` is explicitly imputed.

## What we tested and dropped

- **Provenance flags** (recovering which legacy system a row came from, via date/encoding style) — no lift (0.8600 vs 0.8604). The format mess was a *cleaning* test, not a hidden feature.

---

### Speaker notes
The crowd-pleaser here is the inversion of the usual NLP story: *"the smartest move on the free-text column was to realize it wasn't really free text — it's 60 canned verdicts, so we let the model memorize each one's approval rate."* Also call out the **inverted-U risk feature** — linear correlation said the risk indicators were useless (−0.08), but `|risk − 55|` unlocked real signal.

### Assets to add
- `assets/figures/06_opinion_tiers.png` — approval rate across the 60 opinions (3 tiers visible)
- `assets/figures/06_feature_importance.png` — gain importance (risk12, risk3, amt_to_income, analyst_opinion on top)
