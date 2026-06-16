# Slide 8 — Modeling Approach

## Model selection — baseline spot-check (5-fold accuracy)

We didn't just assume boosted trees; we measured. A spot-check across model families confirms the choice and gives a clean progression:

| Model | 5-fold accuracy |
|---|---|
| Naive Bayes | 0.7597 |
| LDA | 0.7770 |
| Logistic Regression | 0.7835 |
| Random Forest | 0.8334 |
| **LightGBM** | **0.8358** |

Boosted trees win decisively → all further effort went into them, ensembling, and validation rigor.

## Final model families

| Model | Why | Result (OOF accuracy) |
|---|---|---|
| **LightGBM** | Fast, native categoricals + missing handling | ~0.858 |
| **XGBoost** | Different tree algorithm → ensemble diversity | 0.8581 |
| **CatBoost** | Ordered target stats — best on the 60-template opinion column | 0.8607 (strongest single) |
| **Ensemble (LGBM+XGB+CatBoost)** | Mean-blended probabilities | 0.8590–0.8615 |
| **v10 = clean prep + pseudo-labeling** | Final model | **0.8590 (AUC 0.9377)** |

## Adversarial validation — is test distributed like train?

We trained a classifier to separate train rows from test rows:

- **All features → AUC = 1.000** — but *only because of the date* (test is entirely May 2026, absent from train).
- **Excluding `year`/`month` → AUC = 0.497 ≈ 0.5** — apart from time, **train and test are identically distributed**. No hidden corruption beyond the traps we removed; our time-aware validation is trustworthy.

## Validation strategy (three independent views)

1. **5-fold Stratified K-Fold OOF** — uses all 25k rows, gives the threshold-tuning curve.
2. **Time split** (train ≤ 2024 → validate 2025–26) — catches temporal drift a random split would hide.
3. **Rolling next-month harness** — for each of the last 6 months, train on everything before it and predict it (2,886 pooled out-of-time predictions). This is the closest possible simulation of the real task: *train ends April 2026, predict May 2026.* **Every experiment was judged here, not on the public leaderboard.**

## Threshold tuning (because the metric is accuracy)

Accuracy needs a hard 0/1 label, so we tune the cutoff on OOF predictions instead of defaulting to 0.5. **We use a *smooth* threshold — the average of the top-1% of candidate cutoffs — not a single argmax.** A single argmax overfits OOF noise (it's what burned submissions v3 and v7); averaging is more robust on the private set. Final cutoff ≈ 0.527, **approval rate ≈ 0.51**, consistent with the declining 2026 trend.

## Hyperparameter tuning

- **LightGBM:** 18-config random search (learning rate, num_leaves, min_child_samples, feature/bagging fractions, L2) → kept top-3 configs (`src/best_params.json`).
- **CatBoost:** 7-config search over depth / learning rate / L2 (`src/best_cat.json`) — depth 6, lr 0.03, l2 3.

## How we train + tune the threshold (core loop)

```python
# 5-fold out-of-fold predictions — every row predicted by a model that never saw it
oof = np.zeros(len(X))
for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    m = lgb.LGBMClassifier(**params)
    m.fit(X.iloc[tr_i], y.iloc[tr_i],
          eval_set=[(X.iloc[va_i], y.iloc[va_i])],
          callbacks=[lgb.early_stopping(150)])
    oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]

# metric is ACCURACY → tune the 0/1 cutoff on OOF probabilities, don't assume 0.5
ths  = np.linspace(0.35, 0.65, 601)
best = max(ths, key=lambda t: ((oof > t) == y).mean())   # ≈ 0.52 → approval rate ≈ 0.51
```

## Final pipeline (v8)

```
prep.build()  →  CatBoost pool: 3 configs × 2 seeds × 3 CV fold-seeds (18 members, 5-fold)
              →  pseudo-label test rows with confidence ≥ 0.97 (half weight)
              →  average probabilities  →  OOF-tuned threshold  →  predict
```

---

### Speaker notes
The story arc: *baseline 0.52 → one feature gets 0.74 → tuned single tree 0.858 → ensemble 0.861.* The differentiator vs other teams is the **rolling next-month validation** — say clearly that we never tuned to the public leaderboard, which is why we trust our private-board result. CatBoost winning makes sense: it handles the 60-category opinion column best.

### Assets to add
- `assets/figures/08_validation_diagram.png` — the 3 validation schemes
- `assets/figures/08_model_progression.png` — accuracy ladder baseline→final
