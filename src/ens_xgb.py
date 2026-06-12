import sys, numpy as np, xgboost as xgb
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
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
oof = np.zeros(len(X)); pte = np.zeros(len(Xte)); nm = 0
for sd in (42, 7, 2026):
    for tr_i, va_i in folds:
        m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.025, max_depth=6,
                              min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                              reg_lambda=1.0, enable_categorical=True, tree_method='hist',
                              early_stopping_rounds=150, eval_metric='logloss', seed=sd, verbosity=0)
        m.fit(X.iloc[tr_i], y.iloc[tr_i], eval_set=[(X.iloc[va_i], y.iloc[va_i])], verbose=False)
        oof[va_i] += m.predict_proba(X.iloc[va_i])[:,1]
        pte += m.predict_proba(Xte)[:,1]/len(folds)
    nm += 1; print("xgb member", nm, flush=True)
np.save(D+"../src/oof_xgb.npy", oof/nm); np.save(D+"../src/pte_xgb.npy", pte/nm)
print("XGB DONE")
