import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/src")
from prep import build
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
X0, y, Xte0, te_id, tr_raw = build(D)
ol, og = np.load(D+"../src/oof_lgbm.npy"), np.load(D+"../src/oof_xgb.npy")
pl, pg = np.load(D+"../src/pte_lgbm.npy"), np.load(D+"../src/pte_xgb.npy")
oof = 0.75*ol + 0.25*og; pte = 0.75*pl + 0.25*pg
ths = np.linspace(0.35,0.65,601)
acc = [((oof>t)==y).mean() for t in ths]; bt = ths[int(np.argmax(acc))]
recent = (X0.year>=2025).values
acc_r = [((oof[recent]>t)==y[recent]).mean() for t in ths]; bt_r = ths[int(np.argmax(acc_r))]
print(f"ENSEMBLE OOF acc: @0.5={((oof>0.5)==y).mean():.4f}  global t={bt:.3f} acc={max(acc):.4f}")
print(f"recent rows: global-t acc={((oof[recent]>bt)==y[recent]).mean():.4f} | recent-t={bt_r:.3f} acc={max(acc_r):.4f}")
for name, t in [("sub_v2_ensemble.csv", bt), ("sub_v3_recent_threshold.csv", bt_r)]:
    pred = (pte>t).astype(int)
    pd.DataFrame({'id': te_id, 'credit_decision': pred}).to_csv(D+"../submissions/"+name, index=False)
    print(f"saved {name} t={t:.3f} approval_rate={pred.mean():.3f}")
