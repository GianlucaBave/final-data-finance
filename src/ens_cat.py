import sys, numpy as np, pandas as pd
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
X, Xte = shape(X0), shape(Xte0)
cats = ['job_category','status','analyst_opinion']
for df in (X, Xte):
    for c in cats: df[c] = df[c].astype(str)   # catboost wants str categories
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
oof = np.zeros(len(X)); pte = np.zeros(len(Xte)); nm = 0
for sd in (42, 7, 2026):
    for tr_i, va_i in folds:
        m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                               random_seed=sd, cat_features=cats, verbose=0,
                               early_stopping_rounds=150, allow_writing_files=False)
        m.fit(X.iloc[tr_i], y.iloc[tr_i], eval_set=(X.iloc[va_i], y.iloc[va_i]))
        oof[va_i] += m.predict_proba(X.iloc[va_i])[:,1]
        pte += m.predict_proba(Xte)[:,1]/len(folds)
    nm += 1; print("cat member", nm, flush=True)
np.save(D+"../src/oof_cat.npy", oof/nm); np.save(D+"../src/pte_cat.npy", pte/nm)
print("CAT DONE")
