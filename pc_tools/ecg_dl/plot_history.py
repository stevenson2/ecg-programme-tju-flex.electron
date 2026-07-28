#!/usr/bin/env python3
"""生成模型调优历史可视化图表"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

experiments = [
    ("基线\nMIT-BIH+CE", 88.50, 0.9540, 0.75, 0.82, 0.78),
    ("FocalLoss\n修复",     89.66, 0.9549, 0.83, 0.89, 0.86),
    ("+INCART\nα=0.75",    93.98, 0.9716, 0.72, 0.84, 0.78),
]
names = [e[0] for e in experiments]
acc   = [e[1] for e in experiments]
auc   = [e[2] for e in experiments]
recall= [e[3] for e in experiments]
prec  = [e[4] for e in experiments]
f1    = [e[5] for e in experiments]

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'figure.dpi': 150})

# === Figure 1: Acc + AUC ===
fig, ax1 = plt.subplots(figsize=(8, 5))
x = np.arange(len(names)); w = 0.35
bars1 = ax1.bar(x - w/2, acc, w, color='#2196F3', edgecolor='white')
ax1.set_ylabel('Accuracy (%)', color='#2196F3'); ax1.set_ylim(84, 96)
ax1.tick_params(axis='y', labelcolor='#2196F3')
for bar, v in zip(bars1, acc):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f'{v:.2f}%', ha='center', fontsize=10, fontweight='bold', color='#1565C0')
ax2 = ax1.twinx()
ax2.plot(x + w/2, auc, 'o-', color='#FF5722', linewidth=2.5, markersize=10)
ax2.set_ylabel('AUC', color='#FF5722'); ax2.set_ylim(0.94, 0.98)
ax2.tick_params(axis='y', labelcolor='#FF5722')
for i, v in enumerate(auc):
    ax2.annotate(f'{v:.4f}', (x[i]+w/2, v), textcoords="offset points",
                 xytext=(12,-12), fontsize=10, fontweight='bold', color='#BF360C')
ax1.set_xticks(x); ax1.set_xticklabels(names)
ax1.set_title('ECG Model Evolution: Accuracy & AUC', fontweight='bold', pad=15)
ax1.annotate('+4.32%', xy=(2,acc[2]), xytext=(1.5,92),
             arrowprops=dict(arrowstyle='->',color='#4CAF50',lw=2),
             fontsize=10, color='#2E7D32', fontweight='bold')
plt.tight_layout(); fig.savefig('fig_accuracy_evolution.png', bbox_inches='tight')
plt.close(); print("[OK] fig_accuracy_evolution.png")

# === Figure 2: Abnormal P/R/F1 ===
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(names)); w = 0.22
ax.bar(x - w, recall, w, label='Recall', color='#E91E63')
ax.bar(x, prec, w, label='Precision', color='#9C27B0')
ax.bar(x + w, f1, w, label='F1', color='#673AB7')
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel('Score'); ax.set_ylim(0.65, 0.95)
ax.set_title('Abnormal Class Metrics', fontweight='bold', pad=15)
ax.legend(loc='lower right')
ax.annotate('Recall ↓ (0.83→0.72)\nINCART稀释异常比例', xy=(2,recall[2]),
            xytext=(2.3,0.78), fontsize=9, color='#C62828',
            arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.5))
plt.tight_layout(); fig.savefig('fig_abnormal_metrics.png', bbox_inches='tight')
plt.close(); print("[OK] fig_abnormal_metrics.png")

# === Figure 3: Alpha sensitivity ===
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
alphas = ['α=0.75', 'α=0.85']
# Acc subplot
ax = axes[0]
bars = ax.bar(alphas, [93.98, 94.23], color=['#4CAF50','#FF9800'], width=0.5)
for bar, v in zip(bars, [93.98, 94.23]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()-0.5,
            f'{v:.2f}%', ha='center', fontsize=13, fontweight='bold', color='white')
ax.set_ylim(93, 95); ax.set_title('Accuracy by α', fontweight='bold')
ax.annotate('BEST', xy=(0,93.98), xytext=(0,94.5), ha='center',
            fontsize=12, fontweight='bold', color='#2E7D32',
            arrowprops=dict(arrowstyle='->',color='#4CAF50',lw=2))
# Recall subplot
ax = axes[1]
x2 = np.arange(2); w2 = 0.3
ax.bar(x2-w2/2, [0.72,0.69], w2, label='Recall', color='#E91E63')
ax.bar(x2+w2/2, [0.84,0.89], w2, label='Precision', color='#9C27B0')
for i,(r,p) in enumerate(zip([0.72,0.69],[0.84,0.89])):
    ax.text(i-w2/2,r+0.01,f'{r:.2f}',ha='center',fontsize=11,fontweight='bold',color='#C62828')
    ax.text(i+w2/2,p+0.01,f'{p:.2f}',ha='center',fontsize=11,fontweight='bold',color='#6A1B9A')
ax.set_xticks(x2); ax.set_xticklabels(alphas)
ax.set_ylim(0.60,0.95); ax.set_title('Recall/Precision by α', fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout(); fig.savefig('fig_alpha_sensitivity.png', bbox_inches='tight')
plt.close(); print("[OK] fig_alpha_sensitivity.png")

# === Figure 4: Waterfall ===
fig, ax = plt.subplots(figsize=(8, 4.5))
contrib = [("Baseline\n(88.5%)", 88.5, '#78909C'),
           ("FocalLoss Fix", 89.66, '#4CAF50'),
           ("+INCART Data", 93.98, '#2196F3')]
for i, (label, val, color) in enumerate(contrib):
    ax.bar(i, val-84, 0.5, bottom=84, color=color, edgecolor='white')
    ax.text(i, val+0.15, f'{val:.2f}%', ha='center', fontsize=11, fontweight='bold')
    if i > 0:
        prev = contrib[i-1][1]
        ax.annotate(f'+{val-prev:.2f}%', xy=(i-0.5, val),
                    xytext=(i-0.5, val+0.6), ha='center',
                    fontsize=10, fontweight='bold', color='#2E7D32')
ax.set_xticks(range(len(contrib)))
ax.set_xticklabels([c[0] for c in contrib])
ax.set_ylabel('Accuracy (%)'); ax.set_ylim(84, 96)
ax.set_title('Improvement Waterfall: 88.5% → 93.98%', fontweight='bold', pad=15)
plt.tight_layout(); fig.savefig('fig_waterfall.png', bbox_inches='tight')
plt.close(); print("[OK] fig_waterfall.png")

print("\n✅ Done: 4 charts in pc_tools/ecg_dl/")
