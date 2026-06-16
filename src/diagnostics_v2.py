"""Real numbers for the notebook's new sections: baselines + adversarial validation (prep_v2)."""
import sys, numpy as np, pandas as pd, warnings, lightgbm as lgb
warnings.filterwarnings('ignore')
sys.path.insert(0, "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/src")
from prep_v2 import build
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
X, y, Xte, te_id, tr = build(D)
print("features:", len(X.columns))
NUM = X.select_dtypes(exclude=['category']).columns.tolist()
Xn = X[NUM].fillna(X[NUM].median()); Xtn = Xte[NUM].fillna(X[NUM].median())
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for name, mdl in [
    ('NaiveBayes', Pipeline([('s',StandardScaler()),('m',GaussianNB())])),
    ('LogReg', Pipeline([('s',StandardScaler()),('m',LogisticRegression(max_iter=2000, solver='liblinear', random_state=42))])),
    ('LDA', LinearDiscriminantAnalysis()),
    ('RandomForest', RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=42)),
    ('LightGBM', lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=42)),
]:
    sc = cross_val_score(mdl, Xn, y, cv=skf, scoring='accuracy', n_jobs=-1)
    print(f"  {name:12s} acc = {sc.mean():.4f} +/- {sc.std():.4f}")
# adversarial validation
Xa = pd.concat([Xn, Xtn]).reset_index(drop=True); ya = np.r_[np.zeros(len(Xn)), np.ones(len(Xtn))]
adv = cross_val_score(lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=42),
                      Xa, ya, cv=skf, scoring='roc_auc', n_jobs=-1)
print(f"\nADVERSARIAL train-vs-test AUC = {adv.mean():.4f} +/- {adv.std():.4f}  (0.5 = identical distributions)")
