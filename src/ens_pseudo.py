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
    for c in cats: df[c] = df[c].astype(str)

# pseudo-labels from v7's pure-cat repeated-CV probabilities
pte_v7 = np.mean([np.load(D+"../src/pte_cat9.npy"), np.load(D+"../src/pte_cat_fs7.npy"),
                  np.load(D+"../src/pte_cat_fs99.npy")], axis=0)
CONF = 0.97
mask = (pte_v7 > CONF) | (pte_v7 < 1-CONF)
yp = (pte_v7[mask] > 0.5).astype(int)
Xp = Xte[mask]
print(f"pseudo-labeling {mask.sum()}/5000 test rows (approval among them: {yp.mean():.3f})")

CONFIGS = [dict(depth=6, learning_rate=0.03, l2_leaf_reg=3),
           dict(depth=6, learning_rate=0.02, l2_leaf_reg=5),
           dict(depth=6, learning_rate=0.03, l2_leaf_reg=10, random_strength=2)]
oof = np.zeros(len(X)); pte = np.zeros(len(Xte)); nm = 0
for fs in (42, 7, 99):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=fs).split(X, y))
    for cfg in CONFIGS:
        for sd in (42, 7):
            for tr_i, va_i in folds:
                Xtr = pd.concat([X.iloc[tr_i], Xp]); ytr = np.concatenate([y.iloc[tr_i], yp])
                sw = np.concatenate([np.ones(len(tr_i)), np.full(len(yp), 0.5)])
                m = CatBoostClassifier(iterations=3000, random_seed=sd, verbose=0, early_stopping_rounds=150,
                                       cat_features=cats, allow_writing_files=False, **cfg)
                m.fit(Xtr, ytr, sample_weight=sw, eval_set=(X.iloc[va_i], y.iloc[va_i]))
                oof[va_i] += m.predict_proba(X.iloc[va_i])[:,1]
                pte += m.predict_proba(Xte)[:,1]/5
            nm += 1; print(f"fs{fs} member {nm}", flush=True)
oof /= (nm/3); pte /= nm   # oof accumulates once per fold-seed triple
oof /= 3
ths = np.linspace(0.35,0.65,601)
accs = [((oof>t)==y.values).mean() for t in ths]
bt = ths[int(np.argmax(accs))]
print(f"v8 OOF acc {max(accs):.4f} at t={bt:.3f} (NOTE: slightly optimistic due to pseudo-label feedback)")
pred = (pte > bt).astype(int)
pd.DataFrame({'id': te_id, 'credit_decision': pred}).to_csv(D+"../submissions/sub_v8_pseudo.csv", index=False)
v7 = pd.read_csv(D+"../submissions/sub_v7_repeatcv.csv")
print(f"saved sub_v8_pseudo.csv approval={pred.mean():.3f} | rows changed vs v7: {(v7.credit_decision.values!=pred).sum()}")
