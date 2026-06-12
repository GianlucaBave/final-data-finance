import sys, json, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/src")
from prep import build, parse_dates
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
X0, y, Xte0, te_id, tr_raw = build(D)
dates = parse_dates(tr_raw.date); ym = (dates.dt.year*100 + dates.dt.month).values
def shape(X):
    X = X.copy()
    X['risk12_dev']=(X.risk12-55).abs(); X['risk3_dev']=(X.risk3-55).abs()
    X['risk_max']=X[['risk12','risk3']].max(axis=1); X['risk_mean']=X[['risk12','risk3']].mean(axis=1)
    return X
X = shape(X0)
with open(D+"../src/best_params.json") as f: CFG = json.load(f)[0]
MONTHS = [202511, 202512, 202601, 202602, 202603, 202604]
SEEDS = (42, 7)
ths = np.linspace(0.35,0.65,301)

def fit_predict(Xtr, ytr, Xva, sw=None):
    p = np.zeros(len(Xva))
    for sd in SEEDS:
        m = lgb.LGBMClassifier(**{**CFG, 'seed': sd, 'n_estimators': 1200})
        m.fit(Xtr, ytr, sample_weight=sw)
        p += m.predict_proba(Xva)[:,1]/len(SEEDS)
    return p

for conf in (0.90, 0.97):
    base_p, pl_p, ys = [], [], []
    for mth in MONTHS:
        tr_m, va_m = ym < mth, ym == mth
        Xtr, ytr, Xva = X[tr_m], y[tr_m].values, X[va_m]
        pb = fit_predict(Xtr, ytr, Xva)
        # pseudo-label confident rows of the "unseen month"
        mask = (pb > conf) | (pb < 1-conf)
        Xpl = pd.concat([Xtr, Xva[mask]]); ypl = np.concatenate([ytr, (pb[mask]>0.5).astype(int)])
        sw = np.concatenate([np.ones(len(ytr)), np.full(mask.sum(), 0.5)])
        pp = fit_predict(Xpl, ypl, Xva, sw)
        base_p.append(pb); pl_p.append(pp); ys.append(y[va_m].values)
        print(f"month {mth} conf={conf}: pseudo-labeled {mask.sum()}/{va_m.sum()}", flush=True)
    bp, pp_, t_ = np.concatenate(base_p), np.concatenate(pl_p), np.concatenate(ys)
    ab = max(((bp>t)==t_).mean() for t in ths); ap = max(((pp_>t)==t_).mean() for t in ths)
    print(f"== conf={conf}: BASE {ab:.4f}  vs  PSEUDO {ap:.4f}  ({'+' if ap>ab else ''}{(ap-ab)*100:.2f}pt)", flush=True)
