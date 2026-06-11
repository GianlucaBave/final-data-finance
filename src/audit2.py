import pandas as pd, numpy as np
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 50)
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
tr = pd.read_csv(D+"train.csv"); te = pd.read_csv(D+"test.csv")

print("=== NUMERIC DISTRIBUTIONS (train vs test) ===")
for c in ['ann_income','other_income','amount','age','birth_year','internal_code',
          'risk_indicator_1','risk_indicator_2','risk_indicator_3',
          'cr_scores_fico','cr_scores_vantage','cr_scores_schufa','external_pd_score']:
    a = pd.to_numeric(tr[c], errors='coerce'); b = pd.to_numeric(te[c], errors='coerce')
    print(f"{c:20s} TR q01={a.quantile(.01):>10.2f} q25={a.quantile(.25):>10.2f} med={a.median():>10.2f} q75={a.quantile(.75):>10.2f} q99={a.quantile(.99):>10.2f} | TE med={b.median() if b.notna().any() else float('nan'):>10.2f} q99={b.quantile(.99) if b.notna().any() else float('nan'):>10.2f}")

print("\n=== ann_income: non-numeric strings? ===")
raw = pd.read_csv(D+"train.csv", dtype=str)
nonnum = raw.ann_income[pd.to_numeric(raw.ann_income, errors='coerce').isna() & raw.ann_income.notna()]
print("count:", len(nonnum), "examples:", nonnum.unique()[:10])
nonnum2 = raw.other_income[pd.to_numeric(raw.other_income, errors='coerce').isna() & raw.other_income.notna()]
print("other_income nonnum count:", len(nonnum2), nonnum2.unique()[:10])

print("\n=== ann_income scale mix? (histogram of log10) ===")
ai = pd.to_numeric(tr.ann_income, errors='coerce')
print(np.histogram(np.log10(ai.clip(lower=0.1)), bins=12)[1].round(1))
print(np.histogram(np.log10(ai.clip(lower=0.1)), bins=12)[0])
print("share < 1000:", (ai<1000).mean().round(3), "| share >= 1000:", (ai>=1000).mean().round(3))

print("\n=== age vs birth_year vs date consistency ===")
yr = tr.date.str.extract(r'(\d{4})')[0].astype(float)
chk = tr.assign(year=yr)
both = chk.dropna(subset=['age','birth_year'])
print("rows with both age & birth_year:", len(both))
print("age + birth_year vs decision year: median(year - birth_year - age) =", (both.year - both.birth_year - both.age).median())
print("abs diff distribution:", (both.year - both.birth_year - both.age).abs().describe().round(2).to_dict())

print("\n=== internal_code ===")
print("nunique:", tr.internal_code.nunique(), "sample:", tr.internal_code.dropna().sample(8, random_state=2).tolist())
print("test nunique:", te.internal_code.nunique())

print("\n=== ANALYST_OPINION: unique values + injection scan ===")
uo = tr.analyst_opinion.value_counts()
print("n unique opinions in train:", len(uo))
uo_te = te.analyst_opinion.value_counts()
print("n unique in test:", len(uo_te), "| test values not in train:", len(set(uo_te.index)-set(uo.index)))
print("\nTop opinions w/ approval rate:")
ar = tr.groupby('analyst_opinion').credit_decision.agg(['mean','size']).sort_values('size', ascending=False)
print(ar.head(30).round(3))
import re
susp = [s for s in set(uo.index)|set(uo_te.index) if re.search(r'ignore|instruct|system|prompt|assistant|AI|model|predict|label|always|override|disregard', s, re.I)]
print("\nSuspicious (injection-like) strings:", susp if susp else "NONE FOUND")
