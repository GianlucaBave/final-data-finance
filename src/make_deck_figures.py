"""Deck-palette figures (transparent bg, no titles) for embedding into the pptx."""
import sys, numpy as np, pandas as pd, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')
sys.path.insert(0, "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/src")
from prep import build
D = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/data/"
FIG = "/Users/riwad/Desktop/ESADE/term 3/data in finance/kagel challenge /final-data-finance/presentation/assets/figures/"

# deck palette
DARK='#14533b'; GREEN='#2e8b62'; MINT='#e3f0ea'; SALMON='#cf7a63'; GREY='#6e7e78'; TEXT='#143a2b'
plt.rcParams.update({'font.family':'sans-serif','font.size':15,'text.color':TEXT,
                     'axes.edgecolor':'#c8d4cf','axes.labelcolor':GREY,
                     'xtick.color':GREY,'ytick.color':GREY,'savefig.transparent':True,
                     'axes.spines.top':False,'axes.spines.right':False})

tr = pd.read_csv(D+"train.csv"); te = pd.read_csv(D+"test.csv"); y = tr.credit_decision
opmap = tr.groupby('analyst_opinion').credit_decision.mean()
tr_op = tr.analyst_opinion.map(opmap); te_op = te.analyst_opinion.map(opmap)

# 1. LEAK (slide 4)
fig,ax=plt.subplots(figsize=(7.2,4.6))
vals=[tr.internal_code.corr(tr_op), te.internal_code.corr(te_op)]
b=ax.bar(['Train','Test'],vals,color=[GREEN,SALMON],width=0.55,zorder=3)
ax.axhline(0,color='#9fb0aa',lw=1)
ax.set_ylabel('correlation with real risk signal',fontsize=14)
ax.set_ylim(-0.12,0.56); ax.grid(axis='y',alpha=0.25,zorder=0)
for bar,v in zip(b,vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+(0.025 if v>0 else -0.05),
            f'{v:+.2f}',ha='center',fontweight='bold',fontsize=20,
            color=(DARK if v>0 else SALMON))
plt.tight_layout(); plt.savefig(FIG+'deck_leak.png',dpi=200); plt.close()

# 2. RISK inverted-U (slide 6 left)
fig,ax=plt.subplots(figsize=(7.6,4.4))
r12=tr.risk_indicator_1.fillna(tr.risk_indicator_2)
g=tr.assign(r=r12).dropna(subset=['r']).groupby(pd.qcut(r12.dropna(),10,duplicates='drop'),observed=True).credit_decision.mean()
ax.plot(range(len(g)),g.values,'o-',color=DARK,lw=3,markersize=9,markerfacecolor=GREEN,markeredgecolor=DARK,zorder=3)
ax.set_xlabel('risk score  (low  →  high)',fontsize=14); ax.set_ylabel('approval rate',fontsize=14)
ax.grid(alpha=0.25,zorder=0); ax.set_xticks([])
plt.tight_layout(); plt.savefig(FIG+'deck_risk.png',dpi=200); plt.close()

# 3. OPINION tiers (slide 6 right)
fig,ax=plt.subplots(figsize=(7.6,4.4))
rates=np.sort(opmap.values)
cols=[SALMON if r<0.5 else (GREY if r<0.88 else GREEN) for r in rates]
ax.bar(range(len(rates)),rates,color=cols,width=1.0,zorder=3)
ax.set_xlabel('60 analyst-opinion templates  (sorted)',fontsize=14); ax.set_ylabel('approval rate',fontsize=14)
ax.set_xticks([]); ax.grid(axis='y',alpha=0.25,zorder=0); ax.set_ylim(0,1.02)
plt.tight_layout(); plt.savefig(FIG+'deck_opinion.png',dpi=200); plt.close()

# 4. FEATURE IMPORTANCE compact (slide 8 box)
X,yb,Xte,_,_=build(D)
X['risk12_dev']=(X.risk12-55).abs();X['risk3_dev']=(X.risk3-55).abs()
X['risk_max']=X[['risk12','risk3']].max(axis=1);X['risk_mean']=X[['risk12','risk3']].mean(axis=1)
params=dict(objective='binary',learning_rate=0.05,num_leaves=63,n_estimators=400,verbose=-1,seed=42)
imps=[]
for ti,vi in StratifiedKFold(5,shuffle=True,random_state=42).split(X,yb):
    m=lgb.LGBMClassifier(**params).fit(X.iloc[ti],yb.iloc[ti]); imps.append(m.feature_importances_)
imp=pd.Series(np.mean(imps,axis=0),index=X.columns).sort_values().tail(8)
fig,ax=plt.subplots(figsize=(8.0,2.6))
ax.barh(imp.index,imp.values,color=DARK,zorder=3)
ax.grid(axis='x',alpha=0.25,zorder=0); ax.tick_params(labelsize=12)
ax.set_xticks([])
plt.tight_layout(); plt.savefig(FIG+'deck_importance.png',dpi=200); plt.close()
print("deck figures written: deck_leak, deck_risk, deck_opinion, deck_importance")
