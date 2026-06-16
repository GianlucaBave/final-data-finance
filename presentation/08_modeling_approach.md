# Slide 8 — Modeling Approach

## Models we used

| Model | Why | Result (OOF accuracy) |
|---|---|---|
| Majority-class | Floor / sanity check | 0.520 |
| `analyst_opinion` tiers only | How far one feature gets you | ~0.74 |
| **LightGBM** | Fast, native categoricals + missing handling | ~0.858 single, 0.8578 in pool |
| **XGBoost** | Different tree algorithm → ensemble diversity | 0.8581 |
| **CatBoost** | Ordered target stats — best on the 60-template opinion column | **0.8607 (strongest single family)** |
| **Blend (LGBM + CatBoost)** | Weighted average of probabilities | **0.8615** |
| Repeated-CV + pseudo-labeling | Variance reduction + semi-supervised | 0.8613–0.8618 (final, v8) |

We deliberately skipped logistic regression / random forest as final models — gradient-boosted trees dominate this kind of tabular data, and our effort went into *validation rigor* and *ensembling* instead.

## Validation strategy (three independent views)

1. **5-fold Stratified K-Fold OOF** — uses all 25k rows, gives the threshold-tuning curve.
2. **Time split** (train ≤ 2024 → validate 2025–26) — catches temporal drift a random split would hide.
3. **Rolling next-month harness** — for each of the last 6 months, train on everything before it and predict it (2,886 pooled out-of-time predictions). This is the closest possible simulation of the real task: *train ends April 2026, predict May 2026.* **Every experiment was judged here, not on the public leaderboard.**

## Threshold tuning (because the metric is accuracy)

Accuracy needs a hard 0/1 label, so we tune the probability cutoff on OOF predictions instead of defaulting to 0.5. Best cutoff ≈ 0.52, giving a **predicted approval rate ≈ 0.51** — consistent with the declining 2026 trend. Submissions with looser cutoffs (higher approval rate) consistently scored worse (Slide 9).

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
