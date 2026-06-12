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
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))

for cfg in TOP:
    for sd in SEEDS:
        for tr_i, va_i in folds:
            m = lgb.LGBMClassifier(**{**cfg, 'seed': sd})
            m.fit(X.iloc[tr_i], y.iloc[tr_i], eval_set=[(X.iloc[va_i], y.iloc[va_i])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[va_i] += m.predict_proba(X.iloc[va_i])[:,1]
            pte += m.predict_proba(Xte)[:,1] / len(folds)
        nm += 1
        print(f"lgbm cfg done ({nm})", flush=True)

for sd in SEEDS:
    for tr_i, va_i in folds:
        m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.025, max_depth=6,
                              min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                              reg_lambda=1.0, enable_categorical=True, tree_method='hist',
                              early_stopping_rounds=150, eval_metric='logloss', seed=sd, verbosity=0)
        m.fit(X.iloc[tr_i], y.iloc[tr_i], eval_set=[(X.iloc[va_i], y.iloc[va_i])], verbose=False)
        oof[va_i] += m.predict_proba(X.iloc[va_i])[:,1]
        pte += m.predict_proba(Xte)[:,1] / len(folds)
    nm += 1
    print(f"xgb seed done ({nm})", flush=True)

oof /= nm; pte /= nm
ths = np.linspace(0.35,0.65,601)
acc_all = [((oof>t)==y).mean() for t in ths]
bt_all = ths[int(np.argmax(acc_all))]
recent = (X.year>=2025).values
acc_rec = [((oof[recent]>t)==y[recent]).mean() for t in ths]
bt_rec = ths[int(np.argmax(acc_rec))]
print(f"ENSEMBLE ({nm} members): OOF acc @0.5 = {((oof>0.5)==y).mean():.4f}")
print(f"global threshold {bt_all:.3f} -> OOF acc {max(acc_all):.4f}")
print(f"recent(2025-26) threshold {bt_rec:.3f} -> acc on recent rows {max(acc_rec):.4f}")
print(f"global-threshold acc on recent rows: {((oof[recent]>bt_all)==y[recent]).mean():.4f}")

for name, t in [("sub_v2_ensemble.csv", bt_all), ("sub_v3_recent_threshold.csv", bt_rec)]:
    pred = (pte > t).astype(int)
    pd.DataFrame({'id': te_id, 'credit_decision': pred}).to_csv(D+"../submissions/"+name, index=False)
    print(f"saved {name}  threshold={t:.3f}  approval_rate={pred.mean():.3f}")
np.save(D+"../src/oof_ensemble.npy", oof); np.save(D+"../src/pte_ensemble.npy", pte)
