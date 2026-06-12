import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/src")
from prep import build
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
X0, y, Xte0, te_id, tr_raw = build(D)
yv = y.values
O = {k: np.load(D+f"../src/oof_{k}.npy") for k in ('lgbm','xgb','cat9')}
P = {k: np.load(D+f"../src/pte_{k}.npy") for k in ('lgbm','xgb','cat9')}
ths = np.linspace(0.35,0.65,601)
def best_acc(p):
    accs = [((p>t)==yv).mean() for t in ths]
    return max(accs), ths[int(np.argmax(accs))]
for k in O: print(f"{k:5s}: {best_acc(O[k])[0]:.4f}")
# weight grid
best = (0,)
for wl in np.arange(0.0, 1.01, 0.05):
    for wx in np.arange(0.0, 1.01-wl, 0.05):
        wc = 1-wl-wx
        a, t = best_acc(wl*O['lgbm']+wx*O['xgb']+wc*O['cat9'])
        if a > best[0]: best = (a, (round(wl,2),round(wx,2),round(wc,2)), t)
acc, w, t = best
print(f"blend w={w}: OOF {acc:.4f} t={t:.3f}")
# stacking LR (logit features, OOF-safe via refit folds)
F = np.column_stack([np.log(np.clip(O[k],1e-6,1-1e-6)/np.clip(1-O[k],1e-6,1-1e-6)) for k in ('lgbm','xgb','cat9')])
Fte = np.column_stack([np.log(np.clip(P[k],1e-6,1-1e-6)/np.clip(1-P[k],1e-6,1-1e-6)) for k in ('lgbm','xgb','cat9')])
soof = np.zeros(len(yv)); ste = np.zeros(len(Fte))
folds = list(StratifiedKFold(5, shuffle=True, random_state=99).split(F, yv))
for tr_i, va_i in folds:
    lr = LogisticRegression(C=1.0).fit(F[tr_i], yv[tr_i])
    soof[va_i] = lr.predict_proba(F[va_i])[:,1]
    ste += lr.predict_proba(Fte)[:,1]/len(folds)
sa, st_ = best_acc(soof)
print(f"stack LR: OOF {sa:.4f} t={st_:.3f}")
# pick winner
if sa > acc: pte, t, src, score = ste, st_, "stack", sa
else: pte, t, src, score = w[0]*P['lgbm']+w[1]*P['xgb']+w[2]*P['cat9'], t, f"blend{w}", acc
pred = (pte>t).astype(int)
pd.DataFrame({'id': te_id, 'credit_decision': pred}).to_csv(D+"../submissions/sub_v6_final.csv", index=False)
v4 = pd.read_csv(D+"../submissions/sub_v4_blend.csv")
print(f"saved sub_v6_final.csv [{src}] OOF={score:.4f} t={t:.3f} approval={pred.mean():.3f} | rows changed vs v4: {(v4.credit_decision.values!=pred).sum()}")
