import sys, numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/src")
from prep import build
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
X0, y, Xte0, te_id, tr_raw = build(D)
def shape(X):
    X = X.copy()
    X['risk12_dev']=(X.risk12-55).abs(); X['risk3_dev']=(X.risk3-55).abs()
    X['risk_max']=X[['risk12','risk3']].max(axis=1); X['risk_mean']=X[['risk12','risk3']].mean(axis=1)
    return X
X = shape(X0)
cats = ['job_category','status','analyst_opinion']
for c in cats: X[c] = X[c].astype(str)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
ths = np.linspace(0.35,0.65,601)

CONFIGS = [
    dict(depth=6, learning_rate=0.03, l2_leaf_reg=3),                       # current
    dict(depth=8, learning_rate=0.03, l2_leaf_reg=3),
    dict(depth=6, learning_rate=0.02, l2_leaf_reg=5),
    dict(depth=4, learning_rate=0.04, l2_leaf_reg=3),
    dict(depth=6, learning_rate=0.03, l2_leaf_reg=10, random_strength=2),
    dict(depth=8, learning_rate=0.02, l2_leaf_reg=8),
    dict(depth=6, learning_rate=0.03, l2_leaf_reg=3, bootstrap_type='Bernoulli', subsample=0.8),
]
import json
results = []
for i, cfg in enumerate(CONFIGS):
    oof = np.zeros(len(X))
    for tr_i, va_i in folds:
        m = CatBoostClassifier(iterations=3000, random_seed=42, cat_features=cats, verbose=0,
                               early_stopping_rounds=150, allow_writing_files=False, **cfg)
        m.fit(X.iloc[tr_i], y.iloc[tr_i], eval_set=(X.iloc[va_i], y.iloc[va_i]))
        oof[va_i] = m.predict_proba(X.iloc[va_i])[:,1]
    acc = max(((oof>t)==y).mean() for t in ths)
    results.append((acc, cfg))
    print(f"cat cfg{i}: {acc:.4f}  {cfg}", flush=True)
results.sort(key=lambda r: -r[0])
json.dump(results[0][1], open(D+"../src/best_cat.json","w"))
print("BEST:", results[0])
