import pandas as pd, numpy as np
pd.set_option('display.width', 220)
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
tr = pd.read_csv(D+"train.csv"); te = pd.read_csv(D+"test.csv")
y = tr.credit_decision

print("=== RISK INDICATOR MISSINGNESS OVERLAP (train) ===")
r1, r2 = tr.risk_indicator_1.notna(), tr.risk_indicator_2.notna()
print(pd.crosstab(r1, r2, rownames=['r1 present'], colnames=['r2 present']))

print("\n=== POINT-BISERIAL CORR WITH TARGET (train) ===")
for c in ['risk_indicator_1','risk_indicator_2','risk_indicator_3','internal_code','external_pd_score',
          'cr_scores_fico','cr_scores_vantage','cr_scores_schufa','age','ann_income','amount','kids','highest_ed']:
    print(f"{c:20s} corr={tr[c].corr(y):+.4f}  (n={tr[c].notna().sum()})")

print("\n=== r1/r2 coalesced corr ===")
rc = tr.risk_indicator_1.fillna(tr.risk_indicator_2)
print("coalesce(r1,r2) corr:", round(rc.corr(y),4), " n=", rc.notna().sum())

print("\n=== APPROVAL RATE BY YEAR (parsed from date) ===")
yr = tr.date.str.extract(r'(\d{4})')[0]
print(tr.groupby(yr).credit_decision.agg(['mean','size']).round(3))

print("\n=== APPROVAL BY CATEGORY ===")
for c in ['prev_default','vip','job_category','status','religion','race']:
    print(f"\n{c}:"); print(tr.groupby(tr[c].astype(str)).credit_decision.agg(['mean','size']).round(3))

print("\n=== ALL 60 OPINIONS (manual injection check) + tier ===")
ar = tr.groupby('analyst_opinion').credit_decision.agg(['mean','size']).sort_values('mean')
for op, row in ar.iterrows():
    print(f"{row['mean']:.3f} n={int(row['size']):4d}  {op}")
