# Slide 8 — Modeling Approach

## Progression: simple → complex (each step justified)

| Model | OOF Accuracy | Notes |
|---|---|---|
| Naive Bayes | 0.7597 | Distributional baseline |
| LDA | 0.7770 | Linear, Gaussian assumption |
| Logistic Regression | 0.7835 | Interpretable baseline |
| Random Forest | 0.8334 | Non-linear, handles missingness |
| Single LightGBM (tuned) | 0.8358 | Gradient boosting, strongest single model |
| **Final Ensemble (16 members)** | **0.8590** | LGBM + XGB + CatBoost |

**Why ensemble?** Different families make different mistakes. Averaging decorrelates errors.

## Ensemble architecture

- **9 × LightGBM** — top-3 hyperparameter configs × 3 seeds (5-fold CV each)
- **3 × XGBoost** — 3 seeds (5-fold CV)
- **4 × CatBoost** — 2 configs × 2 seeds (5-fold CV)
- Blending: **mean of out-of-fold probabilities** (median tested, tied)
- Semi-supervised: **pseudo-labeling** of test rows at ≥97% confidence, half sample-weight (tested)

## Hyperparameter search

- 18-config **random search** over learning rate, leaves, min child samples, feature/bagging fractions, L2 (LightGBM)
- Top-3 configs kept for the ensemble (`src/best_params.json`)

## Validation strategy — two complementary lenses

1. **5-fold Stratified CV** — preserves class balance, gives an OOF probability for every train row.
2. **Rolling next-month harness** — for each of the last 6 months in train, fit on everything before and predict that month. **Mimics the real task** (test is exactly May 2026, the month after train ends).

The threshold is chosen on **smoothed OOF** (mean of top-1% threshold candidates) to avoid overfitting a single noisy peak.

---

### Speaker notes
Emphasise that this is *not* a single model — the OOF scores tell a story of progressive justification.

### Assets to add
- `assets/figures/08_model_comparison_boxplot.png`
- `assets/figures/08_ensemble_diagram.png`
