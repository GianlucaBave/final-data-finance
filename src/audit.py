import pandas as pd, numpy as np
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 50)
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
tr = pd.read_csv(D+"train.csv", dtype=str)   # read everything as string first - no assumptions
te = pd.read_csv(D+"test.csv", dtype=str)

print("=== SHAPES ===", tr.shape, te.shape)
print("\n=== TARGET BALANCE ===")
print(tr.credit_decision.value_counts(normalize=True))

print("\n=== MISSINGNESS (train% / test%) ===")
m = pd.DataFrame({'train%': tr.isna().mean().round(3)*100, 'test%': te.isna().mean().round(3)*100})
print(m)

print("\n=== UNIQUE VALUE SAMPLES (categorical-ish cols) ===")
for c in ['prev_default','religion','race','highest_ed','job_category','status','kids','vip']:
    print(f"\n-- {c} -- train:", dict(tr[c].value_counts(dropna=False).head(12)))
    print(f"   {c} -- test :", dict(te[c].value_counts(dropna=False).head(12)))

print("\n=== DATE FORMATS (sample of distinct patterns) ===")
import re
def pat(s):
    if pd.isna(s): return 'NaN'
    return re.sub(r'[A-Za-z]+','A', re.sub(r'\d','9', s))
print("train:", tr.date.map(pat).value_counts().head(10).to_dict())
print("test :", te.date.map(pat).value_counts().head(10).to_dict())
print("examples:", tr.date.dropna().sample(8, random_state=1).tolist())
