import pandas as pd, numpy as np
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
tr = pd.read_csv(D+"train.csv"); te = pd.read_csv(D+"test.csv")
opmap = tr.groupby('analyst_opinion').credit_decision.mean()
tr['op'] = tr.analyst_opinion.map(opmap); te['op'] = te.analyst_opinion.map(opmap)

def clean_income(s):
    v = pd.to_numeric(s, errors='coerce')
    return np.where(v < 700, v*1000, v)

tr['inc'] = clean_income(tr.ann_income); te['inc'] = clean_income(te.ann_income)
tr['pd_clean'] = tr.prev_default.map({'0':0,'No':0,'1':1,'Yes':1,0:0,1:1})
te['pd_clean'] = te.prev_default.map({'0':0,'No':0,'1':1,'Yes':1,0:0,1:1})
tr['risk12'] = tr.risk_indicator_1.fillna(tr.risk_indicator_2)
te['risk12'] = te.risk_indicator_1.fillna(te.risk_indicator_2)
tr['vipn'] = tr.vip.astype(str).map({'True':1,'False':0}); te['vipn'] = te.vip.astype(str).map({'True':1,'False':0})
# coalesce credit scores to percentile within each bureau (computed on combined train+test per bureau)
for c in ['cr_scores_fico','cr_scores_vantage','cr_scores_schufa']:
    allv = pd.concat([tr[c], te[c]])
    tr[c+'_p'] = tr[c].rank(pct=True) # quick approx within train
for df in (tr, te):
    df['cs'] = df.cr_scores_fico.fillna(df.cr_scores_vantage).fillna(df.cr_scores_schufa)

feats = ['inc','other_income','amount','age','birth_year','kids','highest_ed','pd_clean',
         'risk_indicator_1','risk_indicator_2','risk_indicator_3','risk12','vipn','cs','internal_code']
print(f"{'feature':20s} {'corr w/ op TRAIN':>16s} {'corr w/ op TEST':>16s}  {'corr w/ y':>10s}")
for f in feats:
    a = tr[f].corr(tr.op); b = te[f].corr(te.op); c = tr[f].corr(tr.credit_decision)
    flag = "  <-- TAMPERED?" if (abs(a) > 0.05 and abs(b) < abs(a)*0.4) else ""
    print(f"{f:20s} {a:>+16.3f} {b:>+16.3f}  {c:>+10.3f}{flag}")

print("\ncategorical integrity (approval-rate encoding vs op):")
for f in ['job_category','status','religion','race']:
    enc = tr.groupby(f).credit_decision.mean()
    a = tr[f].map(enc).corr(tr.op); b = te[f].map(enc).corr(te.op)
    flag = "  <-- TAMPERED?" if (abs(a) > 0.05 and abs(b) < abs(a)*0.4) else ""
    print(f"{f:20s} {a:>+16.3f} {b:>+16.3f}{flag}")
