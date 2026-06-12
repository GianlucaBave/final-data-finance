import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/src")
from prep import build
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
X0, y, Xte0, te_id, tr_raw = build(D)
y = y.values
O = {k: np.load(D+f"../src/oof_{k}.npy") for k in ('lgbm','xgb','cat')}
P = {k: np.load(D+f"../src/pte_{k}.npy") for k in ('lgbm','xgb','cat')}
ths = np.linspace(0.35,0.65,601)
def best_acc(p):
    accs = [((p>t)==y).mean() for t in ths]
    return max(accs), ths[int(np.argmax(accs))]
for k in O: print(f"{k:5s} alone: OOF best acc {best_acc(O[k])[0]:.4f}")
# grid-search blend weights (sum=1)
best = (0,None)
for wl in np.arange(0.2, 0.85, 0.05):
    for wx in np.arange(0.0, 1-wl+0.001, 0.05):
        wc = 1-wl-wx
        a, t = best_acc(wl*O['lgbm']+wx*O['xgb']+wc*O['cat'])
        if a > best[0]: best = (a, (round(wl,2),round(wx,2),round(wc,2)), t)
acc, (wl,wx,wc), t = best
print(f"\nbest blend lgbm={wl} xgb={wx} cat={wc}: OOF acc {acc:.4f} at t={t:.3f}")
pte = wl*P['lgbm']+wx*P['xgb']+wc*P['cat']
pred = (pte>t).astype(int)
pd.DataFrame({'id': te_id, 'credit_decision': pred}).to_csv(D+"../submissions/sub_v4_blend.csv", index=False)
print(f"saved sub_v4_blend.csv approval_rate={pred.mean():.3f}")
# how many rows differ from v2?
v2 = pd.read_csv(D+"../submissions/sub_v2_ensemble.csv")
print("rows different from v2:", (v2.credit_decision.values != pred).sum(), "/ 5000")
