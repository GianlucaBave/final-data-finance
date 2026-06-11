import sys, pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/src")
from prep import build, CATS
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"

X, y, Xte, te_id, tr_raw = build(D)
print("features:", list(X.columns))

params = dict(objective='binary', learning_rate=0.03, num_leaves=63, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              n_estimators=3000, verbose=-1, seed=42)

# ---- 5-fold OOF ----
oof = np.zeros(len(X)); models = []
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for tr_i, va_i in skf.split(X, y):
    m = lgb.LGBMClassifier(**params)
    m.fit(X.iloc[tr_i], y.iloc[tr_i], eval_set=[(X.iloc[va_i], y.iloc[va_i])],
          callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
    models.append(m)

ths = np.linspace(0.3, 0.7, 401)
accs = [( (oof > t).astype(int) == y ).mean() for t in ths]
best_t = ths[int(np.argmax(accs))]
print(f"\nOOF accuracy @0.5: {((oof>0.5)==y).mean():.4f}")
print(f"OOF best threshold {best_t:.3f} -> acc {max(accs):.4f}")

# ---- time-split sanity check (test = May 2026) ----
recent = (X.year >= 2025)
m2 = lgb.LGBMClassifier(**params)
m2.fit(X[~recent], y[~recent], eval_set=[(X[recent], y[recent])],
       callbacks=[lgb.early_stopping(200, verbose=False)])
p2 = m2.predict_proba(X[recent])[:, 1]
acc2 = ((p2 > 0.5) == y[recent]).mean()
accs2 = [((p2 > t) == y[recent]).mean() for t in ths]
print(f"TIME-SPLIT (train<=2024 -> validate 2025/26, n={recent.sum()}): acc@0.5={acc2:.4f}, best={max(accs2):.4f}")

# ---- feature importance ----
imp = pd.Series(np.mean([m.feature_importances_ for m in models], axis=0), index=X.columns).sort_values(ascending=False)
print("\nimportance:\n", imp.round(0))

# ---- fit on full data, predict test ----
n_best = int(np.mean([m.best_iteration_ for m in models]) * 1.1)
final = lgb.LGBMClassifier(**{**params, 'n_estimators': n_best})
final.fit(X, y)
pte = final.predict_proba(Xte)[:, 1]
sub = pd.DataFrame({'id': te_id, 'credit_decision': (pte > best_t).astype(int)})
out = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/submissions/sub_v1_lgbm.csv"
sub.to_csv(out, index=False)
print(f"\nsaved {out}; predicted approval rate: {sub.credit_decision.mean():.3f} (train base {y.mean():.3f})")
