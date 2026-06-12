import sys, json, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/src")
from prep import build, parse_dates
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
X0, y, Xte0, te_id, tr_raw = build(D)
dates = parse_dates(tr_raw.date)
ym = dates.dt.year*100 + dates.dt.month

def shape(X):
    X = X.copy()
    X['risk12_dev']=(X.risk12-55).abs(); X['risk3_dev']=(X.risk3-55).abs()
    X['risk_max']=X[['risk12','risk3']].max(axis=1); X['risk_mean']=X[['risk12','risk3']].mean(axis=1)
    return X
XA = shape(X0)
with open(D+"../src/best_params.json") as f: CFG = json.load(f)[0]

MONTHS = [202511, 202512, 202601, 202602, 202603, 202604]

def rolling(X, label, weights_halflife=None, drop_time=False):
    Xu = X.drop(columns=['year','month']) if drop_time else X
    pool_p, pool_y = [], []
    for m in MONTHS:
        tr_m = (ym < m).values; va_m = (ym == m).values
        sw = None
        if weights_halflife:
            age_years = (dates[tr_m].max() - dates[tr_m]).dt.days / 365.25
            sw = 0.5 ** (age_years / weights_halflife)
        mdl = lgb.LGBMClassifier(**CFG)
        mdl.fit(Xu[tr_m], y[tr_m], sample_weight=sw,
                eval_set=[(Xu[va_m], y[va_m])], callbacks=[lgb.early_stopping(150, verbose=False)])
        pool_p.append(mdl.predict_proba(Xu[va_m])[:,1]); pool_y.append(y[va_m].values)
    p = np.concatenate(pool_p); t_ = np.concatenate(pool_y)
    ths = np.linspace(0.35,0.65,301)
    accs = [((p>th)==t_).mean() for th in ths]
    print(f"{label:42s} n={len(t_)}  acc@0.5={((p>0.5)==t_).mean():.4f}  best={max(accs):.4f} (t={ths[int(np.argmax(accs))]:.3f})")
    return p, t_

rolling(XA, "A: current features")
rolling(XA, "B: + recency weights (halflife 1.5y)", weights_halflife=1.5)
rolling(XA, "B2: + recency weights (halflife 3y)", weights_halflife=3.0)
rolling(XA, "C: drop year/month", drop_time=True)
rolling(XA, "D: drop time + weights 1.5y", weights_halflife=1.5, drop_time=True)
XE = XA.copy(); XE['religion'] = pd.Categorical(tr_raw.religion); XE['race'] = tr_raw.race
rolling(XE, "E: + race/religion (measurement only)")
