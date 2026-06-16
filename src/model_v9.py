"""Improved pipeline v9 — audit-driven fixes.

Changes vs the original ensemble:
- prep_v2: z-score uses train-only stats (no leak) + missingness flags
- Baseline spot-check: LR, LDA, Naive Bayes, Random Forest, LightGBM
- Adversarial validation: train vs test classifier AUC
- Ensemble: LightGBM + XGBoost (+ CatBoost if available)
- Median blending across members (more robust than mean)
- Smooth threshold: average of top-1% threshold candidates
- Reports accuracy, ROC AUC, confusion matrix, classification report, feature importance
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prep_v2 import build

D = os.path.normpath(os.path.join(HERE, '..', 'data')) + '/'
SUB_DIR = os.path.normpath(os.path.join(HERE, '..', 'submissions')) + '/'
os.makedirs(SUB_DIR, exist_ok=True)

print("=" * 70)
print("PIPELINE v9 — audit-driven improvements")
print("=" * 70)

X, y, Xte, te_id, tr_raw = build(D)
print(f"\nFeatures used ({len(X.columns)}): {list(X.columns)}")
print(f"Train: {X.shape}  Test: {Xte.shape}  Target balance: {y.mean():.3f}")

# Numeric-only view for baselines that can't handle categoricals
NUM = X.select_dtypes(exclude=['category']).columns.tolist()
Xnum = X[NUM].fillna(X[NUM].median(numeric_only=True))
Xte_num = Xte[NUM].fillna(X[NUM].median(numeric_only=True))

# ---------------- 1) BASELINE SPOT-CHECK ----------------
print("\n" + "=" * 70)
print("1) BASELINE SPOT-CHECK (5-fold stratified CV, scoring=accuracy)")
print("=" * 70)
seed = 42
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

baselines = [
    ('LR  ', Pipeline([('s', StandardScaler()),
                       ('m', LogisticRegression(max_iter=2000, solver='liblinear', random_state=seed))])),
    ('LDA ', LinearDiscriminantAnalysis()),
    ('NB  ', Pipeline([('s', StandardScaler()),
                       ('m', GaussianNB())])),
    ('RF  ', RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=seed)),
    ('LGBM', lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                 verbose=-1, random_state=seed)),
]
baseline_scores = {}
for name, mdl in baselines:
    sc = cross_val_score(mdl, Xnum, y, cv=skf, scoring='accuracy', n_jobs=-1)
    baseline_scores[name.strip()] = (sc.mean(), sc.std())
    print(f"  {name}  acc = {sc.mean():.4f}  ± {sc.std():.4f}")

# ---------------- 2) ADVERSARIAL VALIDATION ----------------
print("\n" + "=" * 70)
print("2) ADVERSARIAL VALIDATION (train vs test classifier)")
print("=" * 70)
Xa = pd.concat([Xnum, Xte_num], axis=0).reset_index(drop=True)
ya = np.r_[np.zeros(len(Xnum)), np.ones(len(Xte_num))]
adv_scores = cross_val_score(
    lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=seed),
    Xa, ya, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring='roc_auc', n_jobs=-1
)
print(f"  AUC (train vs test) = {adv_scores.mean():.4f}  ± {adv_scores.std():.4f}")
print("  Interpretation: AUC ~0.5 → distributions match (good).  AUC > 0.65 → distribution shift.")

# ---------------- 3) MAIN ENSEMBLE ----------------
print("\n" + "=" * 70)
print("3) MAIN ENSEMBLE  (LGBM + XGB" + (" + CAT" if False else "") + ")")
print("=" * 70)

with open(os.path.join(HERE, 'best_params.json')) as f:
    LCFG = json.load(f)

SEEDS = [42, 7, 2026]
folds = list(skf.split(X, y))

# We collect per-member OOF / test probabilities to enable median blending
oof_members = []
pte_members = []

# --- LGBM members
for cfg in LCFG:
    for sd in SEEDS:
        oof = np.zeros(len(X))
        pte = np.zeros(len(Xte))
        for tr_i, va_i in folds:
            m = lgb.LGBMClassifier(**{**cfg, 'seed': sd})
            m.fit(X.iloc[tr_i], y.iloc[tr_i],
                  eval_set=[(X.iloc[va_i], y.iloc[va_i])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
            pte += m.predict_proba(Xte)[:, 1] / len(folds)
        oof_members.append(oof)
        pte_members.append(pte)
        print(f"  lgbm member done ({len(oof_members)})", flush=True)

# --- XGB members
for sd in SEEDS:
    oof = np.zeros(len(X))
    pte = np.zeros(len(Xte))
    for tr_i, va_i in folds:
        m = xgb.XGBClassifier(
            n_estimators=3000, learning_rate=0.025, max_depth=6,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, enable_categorical=True, tree_method='hist',
            early_stopping_rounds=150, eval_metric='logloss',
            seed=sd, verbosity=0
        )
        m.fit(X.iloc[tr_i], y.iloc[tr_i],
              eval_set=[(X.iloc[va_i], y.iloc[va_i])], verbose=False)
        oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
        pte += m.predict_proba(Xte)[:, 1] / len(folds)
    oof_members.append(oof)
    pte_members.append(pte)
    print(f"  xgb member done ({len(oof_members)})", flush=True)

# --- CatBoost members (if installed)
try:
    from catboost import CatBoostClassifier
    Xc = X.copy()
    Xtec = Xte.copy()
    for c in ['job_category', 'status', 'analyst_opinion']:
        Xc[c] = Xc[c].astype(str)
        Xtec[c] = Xtec[c].astype(str)
    for sd in SEEDS:
        oof = np.zeros(len(X))
        pte = np.zeros(len(Xte))
        for tr_i, va_i in folds:
            m = CatBoostClassifier(
                iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                random_seed=sd, cat_features=['job_category', 'status', 'analyst_opinion'],
                verbose=0, early_stopping_rounds=150, allow_writing_files=False
            )
            m.fit(Xc.iloc[tr_i], y.iloc[tr_i],
                  eval_set=(Xc.iloc[va_i], y.iloc[va_i]))
            oof[va_i] = m.predict_proba(Xc.iloc[va_i])[:, 1]
            pte += m.predict_proba(Xtec)[:, 1] / len(folds)
        oof_members.append(oof)
        pte_members.append(pte)
        print(f"  cat member done ({len(oof_members)})", flush=True)
    has_cat = True
except ImportError:
    print("  (catboost not installed — skipping)")
    has_cat = False

oof_arr = np.array(oof_members)
pte_arr = np.array(pte_members)
print(f"\nMembers collected: {len(oof_members)}")

# ---------------- 4) BLENDING: MEAN vs MEDIAN ----------------
print("\n" + "=" * 70)
print("4) BLENDING: MEAN vs MEDIAN")
print("=" * 70)
oof_mean = oof_arr.mean(axis=0)
oof_med = np.median(oof_arr, axis=0)
pte_mean = pte_arr.mean(axis=0)
pte_med = np.median(pte_arr, axis=0)

ths = np.linspace(0.35, 0.65, 601)
acc_mean = np.array([((oof_mean > t) == y).mean() for t in ths])
acc_med = np.array([((oof_med > t) == y).mean() for t in ths])
print(f"  mean    OOF best acc = {acc_mean.max():.4f} at t = {ths[acc_mean.argmax()]:.3f}")
print(f"  median  OOF best acc = {acc_med.max():.4f} at t = {ths[acc_med.argmax()]:.3f}")

# Choose the better blender
if acc_med.max() > acc_mean.max():
    oof_e, pte_e, accs = oof_med, pte_med, acc_med
    blend = "median"
else:
    oof_e, pte_e, accs = oof_mean, pte_mean, acc_mean
    blend = "mean"
print(f"  -> using {blend} blending")

# ---------------- 5) SMOOTH THRESHOLD ----------------
print("\n" + "=" * 70)
print("5) SMOOTH THRESHOLD (top-1% window)")
print("=" * 70)
top_pct = int(0.01 * len(ths))
top_idx = np.argsort(accs)[-top_pct:]
best_t = ths[accs.argmax()]
smooth_t = ths[top_idx].mean()
print(f"  argmax threshold     = {best_t:.4f}  (acc {accs.max():.4f})")
print(f"  smooth (top-1% avg)  = {smooth_t:.4f}  (acc {accs[(np.abs(ths - smooth_t)).argmin()]:.4f})")

# ---------------- 6) FINAL METRICS ----------------
print("\n" + "=" * 70)
print("6) FINAL METRICS at smooth threshold")
print("=" * 70)
pred_oof = (oof_e > smooth_t).astype(int)
auc = roc_auc_score(y, oof_e)
acc = accuracy_score(y, pred_oof)
cm = confusion_matrix(y, pred_oof)
print(f"  OOF accuracy = {acc:.4f}")
print(f"  OOF ROC AUC  = {auc:.4f}")
print(f"  Confusion matrix [[TN FP][FN TP]]:")
print(f"    {cm.tolist()}")
print(f"  Classification report:")
print(classification_report(y, pred_oof, target_names=['rejected', 'approved']))

# ---------------- 7) FEATURE IMPORTANCE (refit one LGBM on full data) ----------------
print("\n" + "=" * 70)
print("7) FEATURE IMPORTANCE (LGBM refit on full data)")
print("=" * 70)
fi_model = lgb.LGBMClassifier(**LCFG[0])
fi_model.fit(X, y)
imp = pd.Series(fi_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(15).to_string())

# ---------------- 8) WRITE SUBMISSION ----------------
print("\n" + "=" * 70)
print("8) SUBMISSION")
print("=" * 70)
pred_test = (pte_e > smooth_t).astype(int)
sub = pd.DataFrame({'id': te_id, 'credit_decision': pred_test})
out_path = os.path.join(SUB_DIR, 'sub_v9_audit.csv')
sub.to_csv(out_path, index=False)
print(f"  Saved {out_path}")
print(f"  Approval rate: {pred_test.mean():.3f}  (train base rate: {y.mean():.3f})")

# ---------------- 9) SUMMARY ----------------
print("\n" + "=" * 70)
print("9) SUMMARY")
print("=" * 70)
print("  Baseline scores:")
for k, (m, s) in baseline_scores.items():
    print(f"    {k:5s}  {m:.4f} ± {s:.4f}")
print(f"  Adversarial AUC = {adv_scores.mean():.4f}")
print(f"  Ensemble OOF accuracy = {acc:.4f}  |  AUC = {auc:.4f}")
print(f"  Blender = {blend}  |  Threshold = {smooth_t:.4f}  |  Approval = {pred_test.mean():.3f}")
print(f"  Submission written: {out_path}")
