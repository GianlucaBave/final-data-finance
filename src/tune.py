import sys, pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/src")
from prep import build
rng = np.random.RandomState(0)
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
X0, y, Xte0, te_id, tr_raw = build(D)

def add_risk_shape(X):
    X = X.copy()
    X['risk12_dev'] = (X.risk12-55).abs(); X['risk3_dev'] = (X.risk3-55).abs()
    X['risk_max'] = X[['risk12','risk3']].max(axis=1); X['risk_mean'] = X[['risk12','risk3']].mean(axis=1)
    return X
X = add_risk_shape(X0); Xte = add_risk_shape(Xte0)

def oof_score(params, seeds=(42,)):
    oof = np.zeros(len(X))
    for sd in seeds:
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        for tr_i, va_i in skf.split(X, y):
            m = lgb.LGBMClassifier(**{**params,'seed':sd})
            m.fit(X.iloc[tr_i], y.iloc[tr_i], eval_set=[(X.iloc[va_i], y.iloc[va_i])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[va_i] += m.predict_proba(X.iloc[va_i])[:,1]/len(seeds)
    ths = np.linspace(0.35,0.65,301)
    accs = [((oof>t)==y).mean() for t in ths]
    return max(accs), ths[int(np.argmax(accs))], oof

results = []
for i in range(18):
    p = dict(objective='binary', verbose=-1, n_estimators=4000,
             learning_rate=10**rng.uniform(-1.9,-1.3),
             num_leaves=int(rng.choice([31,63,95,127])),
             min_child_samples=int(rng.choice([20,40,60,100])),
             feature_fraction=rng.uniform(0.6,0.95),
             bagging_fraction=rng.uniform(0.6,0.95), bagging_freq=1,
             reg_lambda=10**rng.uniform(-2,1.3))
    acc, t, _ = oof_score(p)
    results.append((acc, t, p))
    print(f"cfg{i:02d} acc={acc:.4f} t={t:.2f} lr={p['learning_rate']:.3f} lv={p['num_leaves']} mcs={p['min_child_samples']} ff={p['feature_fraction']:.2f} bf={p['bagging_fraction']:.2f} l2={p['reg_lambda']:.2f}")

results.sort(key=lambda r: -r[0])
import json
best = results[:3]
print("\nTOP3:", [round(r[0],4) for r in best])
with open(D+"../src/best_params.json","w") as f:
    json.dump([r[2] for r in best], f)
