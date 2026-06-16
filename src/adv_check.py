import sys, numpy as np, pandas as pd, warnings, lightgbm as lgb
warnings.filterwarnings('ignore')
sys.path.insert(0, "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/src")
from prep_v2 import build
from sklearn.model_selection import StratifiedKFold, cross_val_score
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
X, y, Xte, te_id, tr = build(D)
NUM = X.select_dtypes(exclude=['category']).columns.tolist()
skf = StratifiedKFold(5, shuffle=True, random_state=42)
def adv(cols, label):
    Xn = X[cols].fillna(X[cols].median()); Xtn = Xte[cols].fillna(X[cols].median())
    Xa = pd.concat([Xn, Xtn]).reset_index(drop=True); ya = np.r_[np.zeros(len(Xn)), np.ones(len(Xtn))]
    s = cross_val_score(lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=42),
                        Xa, ya, cv=skf, scoring='roc_auc', n_jobs=1)
    print(f"  {label:35s} AUC = {s.mean():.4f}")
adv(NUM, "all numeric features")
adv([c for c in NUM if c not in ('year','month')], "excluding year & month")
# feature importance of the adversarial classifier (what separates train/test?)
Xn = X[NUM].fillna(X[NUM].median()); Xtn = Xte[NUM].fillna(X[NUM].median())
Xa = pd.concat([Xn, Xtn]).reset_index(drop=True); ya = np.r_[np.zeros(len(Xn)), np.ones(len(Xtn))]
m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=42).fit(Xa, ya)
imp = pd.Series(m.feature_importances_, index=NUM).sort_values(ascending=False)
print("\ntop separators (train vs test):"); print(imp.head(6).to_string())
