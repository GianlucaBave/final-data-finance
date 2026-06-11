import sys, pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/src")
from prep import build
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
X0, y, Xte0, te_id, tr_raw = build(D)

PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=63, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              n_estimators=3000, verbose=-1, seed=42)

def evaluate(X, label, params=PARAMS, seeds=(42,)):
    oof = np.zeros(len(X))
    for sd in seeds:
        skf = StratifiedKFold(5, shuffle=True, random_state=sd)
        for tr_i, va_i in skf.split(X, y):
            m = lgb.LGBMClassifier(**{**params, 'seed': sd})
            m.fit(X.iloc[tr_i], y.iloc[tr_i], eval_set=[(X.iloc[va_i], y.iloc[va_i])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[va_i] += m.predict_proba(X.iloc[va_i])[:,1] / len(seeds)
    ths = np.linspace(0.35,0.65,301)
    accs = [((oof>t)==y).mean() for t in ths]
    bt = ths[int(np.argmax(accs))]
    # time split
    recent = X.year>=2025 if 'year' in X else None
    m2 = lgb.LGBMClassifier(**params)
    m2.fit(X[~recent], y[~recent], eval_set=[(X[recent], y[recent])], callbacks=[lgb.early_stopping(150, verbose=False)])
    p2 = m2.predict_proba(X[recent])[:,1]
    ts = max(((p2>t)==y[recent]).mean() for t in ths)
    print(f"{label:35s} OOF={max(accs):.4f} (t={bt:.2f})  timesplit={ts:.4f}")
    return oof, bt

def add_risk_shape(X):
    X = X.copy()
    X['risk12_dev'] = (X.risk12 - 55).abs()
    X['risk3_dev']  = (X.risk3 - 55).abs()
    X['risk_max']   = X[['risk12','risk3']].max(axis=1)
    X['risk_mean']  = X[['risk12','risk3']].mean(axis=1)
    return X

evaluate(X0, "A: baseline")
XB = add_risk_shape(X0)
evaluate(XB, "B: + risk shape feats")

# C: opinion rate as numeric (smoothed OOF encoding)
XC = XB.copy()
op = tr_raw.analyst_opinion
rate = np.zeros(len(op))
skf = StratifiedKFold(5, shuffle=True, random_state=7)
for tr_i, va_i in skf.split(op, y):
    m = y.iloc[tr_i].groupby(op.iloc[tr_i]).mean()
    rate[va_i] = op.iloc[va_i].map(m).fillna(y.iloc[tr_i].mean())
XC['op_rate'] = rate
evaluate(XC, "C: B + opinion target-enc")

# D: report-only — add protected attributes
XD = XB.copy()
XD['religion'] = pd.Categorical(tr_raw.religion)
XD['race'] = tr_raw.race
evaluate(XD, "D: B + race/religion (report only)")

# E: tuned params variants on B
evaluate(XB, "E1: B leaves=127 mcs=20", {**PARAMS,'num_leaves':127,'min_child_samples':20})
evaluate(XB, "E2: B lr=0.015 leaves=95", {**PARAMS,'learning_rate':0.015,'num_leaves':95})
evaluate(XB, "E3: B 3-seed avg", PARAMS)
