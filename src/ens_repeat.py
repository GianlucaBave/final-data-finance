import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/src")
from prep import build
from sklearn.model_selection import StratifiedKFold
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
X0, y, Xte0, te_id, tr_raw = build(D)
def shape(X):
    X = X.copy()
    X['risk12_dev']=(X.risk12-55).abs(); X['risk3_dev']=(X.risk3-55).abs()
    X['risk_max']=X[['risk12','risk3']].max(axis=1); X['risk_mean']=X[['risk12','risk3']].mean(axis=1)
    return X
X, Xte = shape(X0), shape(Xte0)
with open(D+"../src/best_params.json") as f: LCFG = json.load(f)
CCFG = [dict(depth=6, learning_rate=0.03, l2_leaf_reg=3),
        dict(depth=6, learning_rate=0.02, l2_leaf_reg=5),
        dict(depth=6, learning_rate=0.03, l2_leaf_reg=10, random_strength=2)]
which = sys.argv[1]; fold_seed = int(sys.argv[2])
folds = list(StratifiedKFold(5, shuffle=True, random_state=fold_seed).split(X, y))
oof = np.zeros(len(X)); pte = np.zeros(len(Xte)); nm = 0
if which == 'lgbm':
    import lightgbm as lgb
    for cfg in LCFG:
        for sd in (42, 7):
            for tr_i, va_i in folds:
                m = lgb.LGBMClassifier(**{**cfg, 'seed': sd})
                m.fit(X.iloc[tr_i], y.iloc[tr_i], eval_set=[(X.iloc[va_i], y.iloc[va_i])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
                oof[va_i] += m.predict_proba(X.iloc[va_i])[:,1]; pte += m.predict_proba(Xte)[:,1]/5
            nm += 1; print(f"lgbm fs{fold_seed} member {nm}", flush=True)
else:
    from catboost import CatBoostClassifier
    Xc, Xtec = X.copy(), Xte.copy()
    for df in (Xc, Xtec):
        for c in ['job_category','status','analyst_opinion']: df[c] = df[c].astype(str)
    for cfg in CCFG:
        for sd in (42, 7):
            for tr_i, va_i in folds:
                m = CatBoostClassifier(iterations=3000, random_seed=sd, verbose=0, early_stopping_rounds=150,
                                       cat_features=['job_category','status','analyst_opinion'],
                                       allow_writing_files=False, **cfg)
                m.fit(Xc.iloc[tr_i], y.iloc[tr_i], eval_set=(Xc.iloc[va_i], y.iloc[va_i]))
                oof[va_i] += m.predict_proba(Xc.iloc[va_i])[:,1]; pte += m.predict_proba(Xtec)[:,1]/5
            nm += 1; print(f"cat fs{fold_seed} member {nm}", flush=True)
np.save(D+f"../src/oof_{which}_fs{fold_seed}.npy", oof/nm)
np.save(D+f"../src/pte_{which}_fs{fold_seed}.npy", pte/nm)
print(f"{which} fs{fold_seed} DONE")
