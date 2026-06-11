import pandas as pd, numpy as np
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
tr = pd.read_csv(D+"train.csv"); te = pd.read_csv(D+"test.csv")
y = tr.credit_decision

print("=== internal_code distribution by class (train) ===")
print("y=0:", tr[y==0].internal_code.describe().round(2).to_dict())
print("y=1:", tr[y==1].internal_code.describe().round(2).to_dict())

print("\naccuracy of simple rule internal_code>50 on train:", ((tr.internal_code>50)==y).mean().round(4))
# find best threshold
ths = np.linspace(0,100,1001)
accs = [((tr.internal_code>t)==y).mean() for t in ths]
best = ths[int(np.argmax(accs))]
print(f"best threshold={best:.1f} train acc={max(accs):.4f}")

print("\n=== Is test internal_code drawn from the same mixture? ===")
# deciles of distribution
q = np.linspace(0,1,11)
print("train quantiles:", tr.internal_code.quantile(q).round(1).tolist())
print("test  quantiles:", te.internal_code.quantile(q).round(1).tolist())
# train is mixture: 52% from y=1 dist, 48% from y=0. Expected test mixture if leak intact and similar base rate
# Compare histograms
ht, _ = np.histogram(tr.internal_code, bins=20, range=(0,100), density=True)
he, _ = np.histogram(te.internal_code, bins=20, range=(0,100), density=True)
print("\ntrain hist:", ht.round(3))
print("test  hist:", he.round(3))

print("\n=== internal_code vs other strong features (does it correlate with anything else?) ===")
# if it's a pure leak, it correlates with target but ALSO with whatever drives target (opinion tier)
op_rate = tr.groupby('analyst_opinion').credit_decision.transform('mean')
print("corr(internal_code, opinion_approval_rate):", tr.internal_code.corr(op_rate).round(3))
# in TEST: if internal_code is real, it should correlate with opinion tier in test too
opmap = tr.groupby('analyst_opinion').credit_decision.mean()
te_oprate = te.analyst_opinion.map(opmap)
print("TEST corr(internal_code, opinion_rate_from_train):", te.internal_code.corr(te_oprate).round(3))

print("\n=== same stability check by year (train) ===")
yr = tr.date.str.extract(r'(\d{4})')[0]
for g, sub in tr.assign(yr=yr).groupby('yr'):
    print(g, "corr:", round(sub.internal_code.corr(sub.credit_decision),3))
