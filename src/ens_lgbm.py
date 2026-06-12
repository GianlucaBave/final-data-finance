import sys, json, numpy as np, lightgbm as lgb
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
X, Xte = shape(X0), shape(Xte0)
with open(D+"../src/best_params.json") as f: TOP = json.load(f)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
oof = np.zeros(len(X)); pte = np.zeros(len(Xte)); nm = 0
for cfg in TOP:
    for sd in (42, 7, 2026):
        for tr_i, va_i in folds:
            m = lgb.LGBMClassifier(**{**cfg, 'seed': sd})
            m.fit(X.iloc[tr_i], y.iloc[tr_i], eval_set=[(X.iloc[va_i], y.iloc[va_i])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[va_i] += m.predict_proba(X.iloc[va_i])[:,1]
            pte += m.predict_proba(Xte)[:,1]/len(folds)
        nm += 1; print("lgbm member", nm, flush=True)
np.save(D+"../src/oof_lgbm.npy", oof/nm); np.save(D+"../src/pte_lgbm.npy", pte/nm)
print("LGBM DONE")
