import sys, json, pandas as pd, numpy as np, lightgbm as lgb, xgboost as xgb
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/src")
from prep import build
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
X0, y, Xte0, te_id, tr_raw = build(D)

def add_risk_shape(X):
    X = X.copy()
    X['risk12_dev'] = (X.risk12-55).abs(); X['risk3_dev'] = (X.risk3-55).abs()
    X['risk_max'] = X[['risk12','risk3']].max(axis=1); X['risk_mean'] = X[['risk12','risk3']].mean(axis=1)
    return X
X = add_risk_shape(X0); Xte = add_risk_shape(Xte0)

with open(D+"../src/best_params.json") as f: TOP = json.load(f)
SEEDS = [42, 7, 2026]

oof = np.zeros(len(X)); pte = np.zeros(len(Xte)); nm = 0
skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(X, y))

# LightGBM members
best_iters = []
for cfg in TOP:
    for sd in SEEDS:
        o = np.zeros(len(X)); t = np.zeros(len(Xte)); its = []
        for tr_i, va_i in folds:
            m = lgb.LGBMClassifier(**{**cfg, 'seed': sd})
            m.fit(X.iloc[tr_i], y.iloc[tr_i], eval_set=[(X.iloc[va_i], y.iloc[va_i])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
            o[va_i] = m.predict_proba(X.iloc[va_i])[:,1]
            t += m.predict_proba(Xte)[:,1] / len(folds)
            its.append(m.best_iteration_)
        oof += o; pte += t; nm += 1
        best_iters.append(np.mean(its))

# XGBoost member (3 seeds)
Xx = X.copy(); Xxte = Xte.copy()
for sd in SEEDS:
    o = np.zeros(len(X)); t = np.zeros(len(Xte))
    for tr_i, va_i in folds:
        m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.025, max_depth=6,
                              min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                              reg_lambda=1.0, enable_categorical=True, tree_method='hist',
                              early_stopping_rounds=150, eval_metric='logloss', seed=sd, verbosity=0)
        m.fit(Xx.iloc[tr_i], y.iloc[tr_i], eval_set=[(Xx.iloc[va_i], y.iloc[va_i])], verbose=False)
        o[va_i] = m.predict_proba(Xx.iloc[va_i])[:,1]
        t += m.predict_proba(Xxte)[:,1] / len(folds)
    oof += o; pte += t; nm += 1

oof /= nm; pte /= nm
ths = np.linspace(0.35,0.65,601)
accs = [((oof>t)==y).mean() for t in ths]
bt = ths[int(np.argmax(accs))]
print(f"ENSEMBLE ({nm} members): OOF acc @0.5 = {((oof>0.5)==y).mean():.4f}")
print(f"best threshold {bt:.3f} -> OOF acc {max(accs):.4f}")

pred = (pte > bt).astype(int)
sub = pd.DataFrame({'id': te_id, 'credit_decision': pred})
out = D + "../submissions/sub_v2_ensemble.csv"
sub.to_csv(out, index=False)
print(f"saved {out}")
print(f"test approval rate: {pred.mean():.3f} | train base rate: {y.mean():.3f} | train May rate:",
      tr_raw[tr_raw.date.str.contains('May-2026|2026-05', na=False)].credit_decision.mean().round(3) if tr_raw.date.str.contains('2026', na=False).any() else 'n/a')
np.save(D+"../src/oof_ensemble.npy", oof); np.save(D+"../src/pte_ensemble.npy", pte)
