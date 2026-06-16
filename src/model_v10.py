"""Pipeline v10 — best of both worlds.

v9 prep (clean, no leak) + v8 architecture (pseudo-labeling).

Stages:
1. Base ensemble: LGBM(9) + XGB(3) + CatBoost(4) = 16 members, mean-blended
2. Pseudo-label test rows with base ensemble confidence >=0.97
3. Refit each member with pseudo-labeled test rows added at sample_weight=0.5
4. Final ensemble = mean of refit-member probabilities

Reports:
- OOF accuracy + AUC + confusion matrix
- Comparison vs v9 and (recorded) v8
- Submission written
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prep_v2 import build

D = os.path.normpath(os.path.join(HERE, '..', 'data')) + '/'
SUB = os.path.normpath(os.path.join(HERE, '..', 'submissions')) + '/'
os.makedirs(SUB, exist_ok=True)

print("=" * 70)
print("PIPELINE v10 — clean prep + pseudo-labeling")
print("=" * 70)

X, y, Xte, te_id, _ = build(D)
print(f"X={X.shape}  Xte={Xte.shape}  base_rate={y.mean():.3f}")

with open(os.path.join(HERE, 'best_params.json')) as f:
    LCFG = json.load(f)

SEEDS_LGBM = [42, 7, 2026]
SEEDS_XGB = [42, 7, 2026]
CAT_CFGS = [
    dict(depth=6, learning_rate=0.03, l2_leaf_reg=3),
    dict(depth=6, learning_rate=0.02, l2_leaf_reg=5),
]
SEEDS_CAT = [42, 7]

skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(X, y))
y_v = y.values
ths = np.linspace(0.35, 0.65, 601)


def cat_data(Xref):
    Xc = Xref.copy()
    for c in ['job_category', 'status', 'analyst_opinion']:
        Xc[c] = Xc[c].astype(str)
    return Xc


def smooth_threshold(oof):
    accs = np.array([((oof > t) == y_v).mean() for t in ths])
    top_idx = np.argsort(accs)[-6:]   # top ~1%
    return float(ths[top_idx].mean()), float(accs.max())


def fit_ensemble(Xpl=None, ypl=None, sw=None, tag="base"):
    """Train all members on (X, y) plus optionally pseudo-labeled (Xpl, ypl) with sw weights.
    Returns: oof_arr (M, N_train), pte_arr (M, N_test).
    """
    oofs = []
    ptes = []

    # ---- LGBM
    for cfg in LCFG:
        for sd in SEEDS_LGBM:
            oof = np.zeros(len(X))
            pte = np.zeros(len(Xte))
            for tr_i, va_i in folds:
                if Xpl is not None:
                    Xtr = pd.concat([X.iloc[tr_i], Xpl])
                    ytr = np.concatenate([y.iloc[tr_i].values, ypl])
                    w = np.concatenate([np.ones(len(tr_i)), sw])
                else:
                    Xtr, ytr, w = X.iloc[tr_i], y.iloc[tr_i].values, None
                m = lgb.LGBMClassifier(**{**cfg, 'seed': sd})
                m.fit(Xtr, ytr, sample_weight=w,
                      eval_set=[(X.iloc[va_i], y.iloc[va_i])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
                oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
                pte += m.predict_proba(Xte)[:, 1] / len(folds)
            oofs.append(oof)
            ptes.append(pte)
            print(f"  [{tag}] lgbm member {len(oofs)}", flush=True)

    # ---- XGB
    for sd in SEEDS_XGB:
        oof = np.zeros(len(X))
        pte = np.zeros(len(Xte))
        for tr_i, va_i in folds:
            if Xpl is not None:
                Xtr = pd.concat([X.iloc[tr_i], Xpl])
                ytr = np.concatenate([y.iloc[tr_i].values, ypl])
                w = np.concatenate([np.ones(len(tr_i)), sw])
            else:
                Xtr, ytr, w = X.iloc[tr_i], y.iloc[tr_i].values, None
            m = xgb.XGBClassifier(
                n_estimators=3000, learning_rate=0.025, max_depth=6,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, enable_categorical=True, tree_method='hist',
                early_stopping_rounds=150, eval_metric='logloss',
                seed=sd, verbosity=0
            )
            m.fit(Xtr, ytr, sample_weight=w,
                  eval_set=[(X.iloc[va_i], y.iloc[va_i])], verbose=False)
            oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
            pte += m.predict_proba(Xte)[:, 1] / len(folds)
        oofs.append(oof)
        ptes.append(pte)
        print(f"  [{tag}] xgb member {len(oofs)}", flush=True)

    # ---- CatBoost
    Xc, Xtec = cat_data(X), cat_data(Xte)
    Xplc = cat_data(Xpl) if Xpl is not None else None
    cat_cols = ['job_category', 'status', 'analyst_opinion']
    for cfg in CAT_CFGS:
        for sd in SEEDS_CAT:
            oof = np.zeros(len(X))
            pte = np.zeros(len(Xte))
            for tr_i, va_i in folds:
                if Xplc is not None:
                    Xtr = pd.concat([Xc.iloc[tr_i], Xplc])
                    ytr = np.concatenate([y.iloc[tr_i].values, ypl])
                    w = np.concatenate([np.ones(len(tr_i)), sw])
                else:
                    Xtr, ytr, w = Xc.iloc[tr_i], y.iloc[tr_i].values, None
                m = CatBoostClassifier(
                    iterations=3000, random_seed=sd, verbose=0,
                    early_stopping_rounds=150, cat_features=cat_cols,
                    allow_writing_files=False, **cfg
                )
                m.fit(Xtr, ytr, sample_weight=w,
                      eval_set=(Xc.iloc[va_i], y.iloc[va_i]))
                oof[va_i] = m.predict_proba(Xc.iloc[va_i])[:, 1]
                pte += m.predict_proba(Xtec)[:, 1] / len(folds)
            oofs.append(oof)
            ptes.append(pte)
            print(f"  [{tag}] cat member {len(oofs)}", flush=True)

    return np.array(oofs), np.array(ptes)


# ========== STAGE 1: base ensemble ==========
print("\n--- STAGE 1: base ensemble (no pseudo-labels) ---")
oof_base_arr, pte_base_arr = fit_ensemble(tag="base")
oof_base = oof_base_arr.mean(axis=0)
pte_base = pte_base_arr.mean(axis=0)
t_base, acc_base = smooth_threshold(oof_base)
auc_base = roc_auc_score(y_v, oof_base)
print(f"\nBASE ENSEMBLE: OOF acc={acc_base:.4f}  AUC={auc_base:.4f}  t={t_base:.3f}")

# ========== STAGE 2: pseudo-labels ==========
CONF = 0.97
mask = (pte_base > CONF) | (pte_base < 1 - CONF)
n_pl = int(mask.sum())
yp = (pte_base[mask] > 0.5).astype(int)
Xp = Xte[mask].copy()
print(f"\n--- STAGE 2: pseudo-labels ({n_pl}/{len(Xte)} test rows, "
      f"approval among them: {yp.mean():.3f}) ---")
sw_pl = np.full(n_pl, 0.5)

# ========== STAGE 3: refit with pseudo-labels ==========
print("\n--- STAGE 3: refit ensemble with pseudo-labeled rows ---")
oof_pl_arr, pte_pl_arr = fit_ensemble(Xpl=Xp, ypl=yp, sw=sw_pl, tag="pseudo")
oof_pl = oof_pl_arr.mean(axis=0)
pte_pl = pte_pl_arr.mean(axis=0)
t_pl, acc_pl = smooth_threshold(oof_pl)
auc_pl = roc_auc_score(y_v, oof_pl)
print(f"\nPSEUDO ENSEMBLE: OOF acc={acc_pl:.4f}  AUC={auc_pl:.4f}  t={t_pl:.3f}")

# ========== FINAL SELECTION ==========
print("\n" + "=" * 70)
print("FINAL SELECTION")
print("=" * 70)
if acc_pl >= acc_base:
    pte_final, t_final, acc_final, auc_final, src = pte_pl, t_pl, acc_pl, auc_pl, "pseudo"
else:
    pte_final, t_final, acc_final, auc_final, src = pte_base, t_base, acc_base, auc_base, "base"
print(f"Using {src} (acc {acc_final:.4f})")

pred_oof_final = ((oof_pl if src == 'pseudo' else oof_base) > t_final).astype(int)
print(f"\nOOF accuracy = {acc_final:.4f}")
print(f"OOF AUC      = {auc_final:.4f}")
print(f"Threshold    = {t_final:.4f}")
print(f"Confusion matrix [[TN FP][FN TP]]:")
print(confusion_matrix(y_v, pred_oof_final).tolist())
print("Classification report:")
print(classification_report(y_v, pred_oof_final, target_names=['rejected', 'approved']))

# ========== SUBMISSION ==========
pred_test = (pte_final > t_final).astype(int)
sub = pd.DataFrame({'id': te_id, 'credit_decision': pred_test})
out = os.path.join(SUB, 'sub_v10_clean_pseudo.csv')
sub.to_csv(out, index=False)
print(f"\nSaved {out}")
print(f"Approval rate: {pred_test.mean():.3f}  (train base rate: {y.mean():.3f})")

# Save probabilities for later threshold/rolling analysis
np.save(os.path.join(HERE, 'oof_v10.npy'), oof_pl if src == 'pseudo' else oof_base)
np.save(os.path.join(HERE, 'pte_v10.npy'), pte_final)
print("Saved oof_v10.npy and pte_v10.npy for later use.")

# ========== COMPARISON ==========
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  v9  (no pseudo)        OOF=0.8588  AUC=0.9376  public=0.8540")
print(f"  v10 BASE (clean prep)  OOF={acc_base:.4f}  AUC={auc_base:.4f}")
print(f"  v10 PSEUDO             OOF={acc_pl:.4f}  AUC={auc_pl:.4f}")
print(f"  v10 selected: {src}  t={t_final:.4f}  approval={pred_test.mean():.3f}")
