import pandas as pd, numpy as np
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
tr = pd.read_csv(D+"train.csv"); te = pd.read_csv(D+"test.csv")
y = tr.credit_decision

print("corr(risk12, risk3) where r1 present:", tr.risk_indicator_1.corr(tr.risk_indicator_3).round(3))
print("corr(risk2, risk3):", tr.risk_indicator_2.corr(tr.risk_indicator_3).round(3))
print("corr(risk1, risk2) on overlap:", tr.risk_indicator_1.corr(tr.risk_indicator_2).round(3))

# binned approval by risk3
print("\napproval rate by risk3 decile:")
print(tr.groupby(pd.qcut(tr.risk_indicator_3, 10)).credit_decision.agg(['mean','size']).round(3))
print("\napproval rate by risk12 decile:")
r12 = tr.risk_indicator_1.fillna(tr.risk_indicator_2)
print(tr.groupby(pd.qcut(r12, 10)).credit_decision.agg(['mean','size']).round(3))
# extremes
print("\nrisk3>90:", tr[tr.risk_indicator_3>90].credit_decision.mean().round(3), " n=", (tr.risk_indicator_3>90).sum())
print("risk3<10:", tr[tr.risk_indicator_3<10].credit_decision.mean().round(3), " n=", (tr.risk_indicator_3<10).sum())
