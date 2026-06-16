# Slide 9 — Results

## Scoreboard

| Model | OOF Accuracy | OOF AUC | Public LB |
|---|---|---|---|
| Naive Bayes | 0.7597 | — | — |
| Logistic Regression | 0.7835 | — | — |
| Random Forest | 0.8334 | — | — |
| Single LightGBM | 0.8358 | — | — |
| First ensemble (v2) | — | — | 0.8520 |
| Tuned ensemble (v6) | — | — | 0.8565 |
| **Final ensemble (v10 — clean prep + pseudo)** | **0.8590** | **0.9377** | 0.8535 |

The **public-LB plateau around 0.853–0.857** is the rumour-floor of a 2,000-row split: ~5 rows decide each ranking. We stopped chasing it (Slide 10).

## Confusion matrix at the smoothed threshold (t = 0.527)

| | Predicted **0** (rejected) | Predicted **1** (approved) |
|---|---|---|
| **Actual 0** | **10,452** (TN) | 1,554 (FP) |
| **Actual 1** | 1,976 (FN) | **11,018** (TP) |

## Classification report

| Class | Precision | Recall | F1 |
|---|---|---|---|
| rejected | 0.84 | 0.87 | 0.86 |
| approved | 0.88 | 0.85 | 0.86 |

Symmetric performance — the model is **not biased toward either decision**. Errors split roughly 44% / 56% across FP and FN.

## Predicted approval rate

Test: **0.509**, vs train base rate 0.520 — consistent with the 2026 tightening trend visible in train.

## Top 10 feature importance (LightGBM, gain)

`analyst_opinion` (12,204) > `amt_to_income` > `risk12_dev` > `age` > `risk_mean` > `amount` > `risk3_dev` > `risk12` > `total_income` > `risk_max`

---

### Speaker notes
Lead with **OOF AUC = 0.9377** — that's the cleanest single number we can show. Then walk the confusion matrix.

### Assets to add
- `assets/figures/09_confusion_matrix.png`
- `assets/figures/09_feature_importance.png`
- `assets/figures/09_roc_curve.png`
