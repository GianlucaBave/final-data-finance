import sys, json, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/src")
from prep import build, parse_dates, provenance
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
X0, y, Xte0, te_id, tr_raw = build(D)
dates = parse_dates(tr_raw.date); ym = dates.dt.year*100 + dates.dt.month
def shape(X):
    X = X.copy()
    X['risk12_dev']=(X.risk12-55).abs(); X['risk3_dev']=(X.risk3-55).abs()
    X['risk_max']=X[['risk12','risk3']].max(axis=1); X['risk_mean']=X[['risk12','risk3']].mean(axis=1)
    return X
XA = shape(X0)
XP = pd.concat([XA, provenance(pd.read_csv(D+"train.csv", dtype=str))], axis=1)
with open(D+"../src/best_params.json") as f: CFG = json.load(f)[0]
MONTHS = [202511, 202512, 202601, 202602, 202603, 202604]
def rolling(X, label, seeds=(42,7)):
    pool_p, pool_y = [], []
    for m in MONTHS:
        tr_m = (ym < m).values; va_m = (ym == m).values
        p = np.zeros(va_m.sum())
        for sd in seeds:
            mdl = lgb.LGBMClassifier(**{**CFG, 'seed': sd})
            mdl.fit(X[tr_m], y[tr_m], eval_set=[(X[va_m], y[va_m])], callbacks=[lgb.early_stopping(150, verbose=False)])
            p += mdl.predict_proba(X[va_m])[:,1]/len(seeds)
        pool_p.append(p); pool_y.append(y[va_m].values)
    p = np.concatenate(pool_p); t_ = np.concatenate(pool_y)
    ths = np.linspace(0.35,0.65,301)
    accs = [((p>th)==t_).mean() for th in ths]
    print(f"{label:30s} acc@0.5={((p>0.5)==t_).mean():.4f}  best={max(accs):.4f}")
rolling(XA, "A: current")
rolling(XP, "P: + provenance flags")
