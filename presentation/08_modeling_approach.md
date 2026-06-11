# Slide 8 — Modeling Approach

## Models we tried

| Model | Why | Status |
|---|---|---|
| Logistic Regression (baseline) | Simple, interpretable, sanity check | _TBD_ |
| Random Forest | Handles missingness, non-linear | _TBD_ |
| XGBoost / LightGBM | State-of-the-art on tabular | _TBD_ |
| _TBD: stacking / blending_ | | |

## Validation strategy

- **Cross-validation:** 5-fold Stratified K-Fold (preserves class balance)
- **Metric:** _TBD — confirm Kaggle's evaluation metric (Accuracy? AUC? F1?)_
- **Hold-out:** none — we use full train with CV; final score on Kaggle leaderboard

## Hyperparameter tuning

- _TBD: grid search / random search / Optuna_
- Search space: _TBD_

## Final pipeline

```
clean_data → feature_engineering → encoder → model → predict
```

(One pipeline, fit on full training, predicted on `test.csv`.)

---

### Speaker notes
Show the **progression** from baseline to final, with the CV scores. The "why each model" column matters.

### Assets to add
- `assets/figures/08_model_comparison.png`
