"""Extra presentation figures: model progression (Slide 8) + ethics cost (Slide 7)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.normpath(os.path.join(HERE, '..', 'presentation', 'assets', 'figures')) + '/'
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'figure.dpi': 130, 'font.size': 11,
                     'axes.grid': True, 'grid.alpha': 0.3})
NAVY = '#1f3a5f'
RED = '#c0392b'
GREEN = '#27ae60'
GREY = '#7f8c8d'

# ---------- 1) Slide 8: model progression ----------
# Source: src/model_v9.py baseline spot-check (5-fold accuracy) + final ensemble OOF
models = ['Naive Bayes', 'LDA', 'LogReg', 'Random Forest', 'LightGBM\n(single)', 'Ensemble\n(LGBM+XGB+CAT)']
scores = [0.7597, 0.7770, 0.7835, 0.8334, 0.8358, 0.8590]
colors = [GREY, GREY, GREY, NAVY, NAVY, GREEN]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(models, scores, color=colors, width=0.65)
for b, v in zip(bars, scores):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.003,
            f'{v:.4f}', ha='center', fontweight='bold', fontsize=10)
ax.axhline(0.52, color='k', lw=0.8, ls='--', alpha=0.5)
ax.text(5.4, 0.525, 'majority baseline 0.52',
        ha='right', fontsize=9, color='k', alpha=0.7)
ax.set_ylim(0.5, 0.88)
ax.set_ylabel('5-fold accuracy')
ax.set_title('Model progression — simple linear → ensemble (+10pt)',
             fontweight='bold')
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(FIG + '08_model_progression.png')
plt.close()
print(f"Saved {FIG}08_model_progression.png")

# ---------- 2) Slide 7: ethics cost ----------
# Source: experiments.py "D: B + race/religion (report only)" — measured but refused
labels = ['Our model\n(no race / religion)', 'Same model\n+ race + religion']
vals = [0.859, 0.866]
colors2 = [GREEN, RED]

fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(labels, vals, color=colors2, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0015,
            f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
ax.annotate('', xy=(1, 0.866), xytext=(1, 0.859),
            arrowprops=dict(arrowstyle='<->', color='k', lw=1.5))
ax.text(1.18, 0.8625, '+0.7 pt\nrefused', fontsize=11,
        fontweight='bold', color=RED, va='center')
ax.set_ylim(0.83, 0.88)
ax.set_ylabel('OOF accuracy')
ax.set_title('Ethics cost — protected attributes refused on principle',
             fontweight='bold')
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(FIG + '07_ethics_cost.png')
plt.close()
print(f"Saved {FIG}07_ethics_cost.png")
