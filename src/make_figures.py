"""Generate all presentation figures into presentation/assets/figures/."""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
sys.path.insert(0, "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/src")
from prep import build, parse_dates
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
FIG = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/presentation/assets/figures/"
plt.rcParams.update({'figure.dpi': 130, 'font.size': 11, 'axes.spilnes' if False else 'axes.grid': True, 'grid.alpha': 0.3})
NAVY='#1f3a5f'; RED='#c0392b'; GREEN='#27ae60'; GREY='#7f8c8d'

tr = pd.read_csv(D+"train.csv"); te = pd.read_csv(D+"test.csv")
y = tr.credit_decision
opmap = tr.groupby('analyst_opinion').credit_decision.mean()
tr_op = tr.analyst_opinion.map(opmap); te_op = te.analyst_opinion.map(opmap)

# --- 1. internal_code leak (the smoking gun) ---
fig, ax = plt.subplots(figsize=(6,4))
vals = [tr.internal_code.corr(tr_op), te.internal_code.corr(te_op)]
bars = ax.bar(['Train','Test'], vals, color=[NAVY, RED], width=0.5)
ax.axhline(0, color='k', lw=0.8)
ax.set_ylabel('corr( internal_code , analyst-opinion tier )')
ax.set_title('internal_code: real signal in train, scrambled in test', fontweight='bold')
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2, v+(0.02 if v>0 else -0.04), f'{v:+.2f}', ha='center', fontweight='bold')
ax.set_ylim(-0.1, 0.55)
plt.tight_layout(); plt.savefig(FIG+'04_internal_code_leak.png'); plt.close()

# --- 2. income bimodal histogram (unit gap) ---
fig, ax = plt.subplots(figsize=(6,4))
inc = pd.to_numeric(tr.ann_income, errors='coerce').clip(lower=0.1)
ax.hist(np.log10(inc), bins=40, color=NAVY, alpha=0.85)
ax.axvspan(np.log10(700), np.log10(2000), color=RED, alpha=0.18, label='empty gap (700–2000)')
ax.set_xlabel('log10(ann_income)'); ax.set_ylabel('rows')
ax.set_title('ann_income is bimodal: half the rows are in thousands', fontweight='bold')
ax.legend()
plt.tight_layout(); plt.savefig(FIG+'04_income_bimodal_hist.png'); plt.close()

# --- 3. missingness train vs test ---
cols = ['external_pd_score','cr_scores_schufa','cr_scores_vantage','cr_scores_fico',
        'risk_indicator_2','risk_indicator_1','prev_default','religion','age','birth_year']
mtr = [tr[c].isna().mean()*100 for c in cols]; mte = [te[c].isna().mean()*100 for c in cols]
fig, ax = plt.subplots(figsize=(7,4.5)); yp = np.arange(len(cols))
ax.barh(yp-0.2, mtr, 0.4, label='Train', color=NAVY)
ax.barh(yp+0.2, mte, 0.4, label='Test', color=RED)
ax.set_yticks(yp); ax.set_yticklabels(cols); ax.set_xlabel('% missing')
ax.set_title('Missingness train vs test — external_pd_score vanishes in test', fontweight='bold')
ax.legend(); plt.tight_layout(); plt.savefig(FIG+'04_missing_values_bar.png'); plt.close()

# --- 4. risk inverted-U ---
fig, ax = plt.subplots(figsize=(6,4))
r12 = tr.risk_indicator_1.fillna(tr.risk_indicator_2)
g = tr.assign(r=r12).dropna(subset=['r']).groupby(pd.qcut(r12.dropna(), 10, duplicates='drop'), observed=True).credit_decision.mean()
ax.plot(range(len(g)), g.values, 'o-', color=NAVY, lw=2)
ax.set_xlabel('risk indicator decile (low → high)'); ax.set_ylabel('approval rate')
ax.set_title('Risk indicators: inverted-U, not linear', fontweight='bold')
plt.tight_layout(); plt.savefig(FIG+'04_risk_inverted_u.png'); plt.close()

# --- 5. target balance ---
fig, ax = plt.subplots(figsize=(4.5,4))
vc = y.value_counts().sort_index()
ax.bar(['Rejected (0)','Approved (1)'], vc.values, color=[GREY, GREEN], width=0.6)
for i,v in enumerate(vc.values): ax.text(i, v+200, f'{v/len(y)*100:.1f}%', ha='center', fontweight='bold')
ax.set_ylabel('rows'); ax.set_title('Target is nearly balanced (52/48)', fontweight='bold')
plt.tight_layout(); plt.savefig(FIG+'04_target_balance.png'); plt.close()

# --- 6. opinion tiers ---
fig, ax = plt.subplots(figsize=(7,4))
rates = opmap.sort_values().values
colors = [RED if r<0.5 else (GREY if r<0.88 else GREEN) for r in rates]
ax.bar(range(len(rates)), rates, color=colors, width=1.0)
ax.set_xlabel('60 analyst-opinion templates (sorted)'); ax.set_ylabel('approval rate')
ax.set_title('analyst_opinion = 60 templates in 3 clear tiers', fontweight='bold')
plt.tight_layout(); plt.savefig(FIG+'06_opinion_tiers.png'); plt.close()

# --- model fit for importance + confusion matrix ---
X, yb, Xte, te_id, _ = build(D)
X['risk12_dev']=(X.risk12-55).abs(); X['risk3_dev']=(X.risk3-55).abs()
X['risk_max']=X[['risk12','risk3']].max(axis=1); X['risk_mean']=X[['risk12','risk3']].mean(axis=1)
params = dict(objective='binary', learning_rate=0.03, num_leaves=63, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, n_estimators=600, verbose=-1, seed=42)
oof = np.zeros(len(X)); imps=[]
for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=42).split(X, yb):
    m = lgb.LGBMClassifier(**params); m.fit(X.iloc[tr_i], yb.iloc[tr_i])
    oof[va_i] = m.predict_proba(X.iloc[va_i])[:,1]; imps.append(m.feature_importances_)

# --- 7. feature importance ---
imp = pd.Series(np.mean(imps,axis=0), index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(7,5))
ax.barh(imp.index, imp.values, color=NAVY)
ax.set_xlabel('mean gain importance'); ax.set_title('Feature importance (LightGBM)', fontweight='bold')
plt.tight_layout(); plt.savefig(FIG+'06_feature_importance.png'); plt.savefig(FIG+'09_feature_importance.png'); plt.close()

# --- 8. confusion matrix ---
bt = max(np.linspace(0.4,0.6,201), key=lambda t: ((oof>t)==yb).mean())
cm = confusion_matrix(yb, (oof>bt).astype(int))
fig, ax = plt.subplots(figsize=(4.5,4))
im = ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',fontsize=13,
                color='white' if cm[i,j]>cm.max()/2 else 'black', fontweight='bold')
ax.set_xticks([0,1]); ax.set_xticklabels(['Pred 0','Pred 1']); ax.set_yticks([0,1]); ax.set_yticklabels(['True 0','True 1'])
ax.set_title(f'Confusion matrix (OOF, acc={((oof>bt)==yb).mean():.4f})', fontweight='bold')
plt.tight_layout(); plt.savefig(FIG+'09_confusion_matrix.png'); plt.close()

# --- 9. score progression v1-v8 ---
fig, ax = plt.subplots(figsize=(7,4))
v=['v1','v2','v4','v6','v7','v8']; oofs=[0.8525,0.8584,0.8610,0.8615,0.8618,0.8615]; pub=[0.8515,0.8520,0.8560,0.8565,0.8550,0.8555]
ax.plot(v, oofs, 'o-', color=NAVY, lw=2, label='OOF (validation)')
ax.plot(v, pub, 's--', color=RED, lw=2, label='Public LB (40%)')
ax.axhline(0.8565, color=GREEN, ls=':', label='LB leader (0.8565)')
ax.set_ylabel('accuracy'); ax.set_title('Score progression v1 → v8', fontweight='bold'); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(FIG+'09_score_progression.png'); plt.close()

import os
print("figures written:", sorted(os.listdir(FIG)))
