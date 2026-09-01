#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_clean_reeval.py — §98 双口径重评可视化
生成:
  models/deploy_match/fig_full_vs_clean_metrics.png  (AUC/事件F1 × 两域 × full/clean)
  models/deploy_match/fig_contamination_and_delta.png (测试记录污染热图 + Δ 对比)
用法: python3 plot_clean_reeval.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = Path(__file__).resolve().parent
SRC = BASE / "models" / "deploy_match" / "clean_test_reeval.json"
OUT1 = BASE / "models" / "deploy_match" / "fig_full_vs_clean_metrics.png"
OUT2 = BASE / "models" / "deploy_match" / "fig_contamination_and_delta.png"

# WSL 无中文字体时退回 DejaVu (英文标签)
_cjk = [f.name for f in font_manager.fontManager.ttflist
        if any(k in f.name for k in ("WenQuanYi", "Noto Sans CJK", "Source Han"))]
if _cjk:
    plt.rcParams["font.sans-serif"] = [_cjk[0]] + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    CN = True
else:
    CN = False

L = {
    "title1": "泄漏前 (full) vs 堵漏后 (clean)：14 模型双口径对比" if CN else
              "Before (full) vs After (clean) leak fix: 14 models",
    "mi": "MIT+INCART" if CN else "MIT+INCART",
    "ptb": "PTB",
    "auc": "AUC", "evf1": "Event F1",
    "full": "full (泄漏口径/legacy)" if CN else "full (leaky)",
    "clean": "clean (干净口径)" if CN else "clean (honest)",
    "na": "无诚实测量 (clean仅剩1条记录)" if CN else "no honest measure (1 record left)",
    "honest": "诚实口径" if CN else "honest",
    "title2": "泄漏规模与指标位移" if CN else "Contamination & metric shift",
    "heat": "各模型泄漏的测试记录数 / 测试集总数" if CN else
            "Leaked test records per model / total test records",
    "delta": "clean − full 差值 (PTB)" if CN else "clean - full delta (PTB)",
}

data = json.loads(SRC.read_text(encoding="utf-8"))
models = data["models"]
names = list(models.keys())
short = {
    "exp7c": "exp7c", "exp7c_v2_hardneg": "v2_hardneg", "exp7c_v3_mild": "v3_mild",
    "exp7c_v4": "v4", "exp7c_ecgfounder": "ecgfdr", "exp7c_ecgfounder_v2": "ecgfdr_v2",
    "exp7c_ecgfounder_v3": "ecgfdr_v3", "exp7c_ecgfounder_v4": "ecgfdr_v4",
    "DEPLOYED_exp7c_int8": "DEPLOYED_int8", "exp7c_qat_int8": "qat_int8",
    "ecgfounder_v3_qat_int8": "v3_qat", "ecgfounder_v3b_qat_int8": "v3b_qat",
    "ecgfounder_v4_qat_int8": "v4_qat", "ecgfounder_v5_qat_int8": "v5_qat",
}
labels = [short[n] for n in names]

TOT_MI, TOT_PTB = 23, 95


def get(name, dom, scope, what):
    m = models[name][dom][scope]
    if what == "auc":
        return m["beat"]["auc"]
    return m["event"]["event_f1"]


def clean_degenerate_mi(name):
    """MIT+INCART clean 子集是否退化到无测量价值 (<=2 条记录)。"""
    dropped = len(models[name]["mit_incart"]["clean"].get("dropped_records", []))
    return (TOT_MI - dropped) <= 2


# ── 图1: 2x2 条形对比 ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle(L["title1"], fontsize=15, fontweight="bold")
x = np.arange(len(names))
w = 0.38
C_FULL, C_CLEAN = "#9ecae1", "#de2d26"

for r, metric in enumerate(("auc", "evf1")):
    for c, dom in enumerate(("mit_incart", "ptb")):
        ax = axes[r][c]
        vfull = [get(n, dom, "full", metric) for n in names]
        ax.bar(x - w / 2, vfull, w, color=C_FULL, edgecolor="k",
               linewidth=0.4, label=L["full"])
        for i, n in enumerate(names):
            if dom == "mit_incart" and clean_degenerate_mi(n):
                ax.text(x[i] + w / 2, 0.02, "N/A", ha="center", va="bottom",
                        fontsize=7.5, color="#666", rotation=90)
                continue
            v = get(n, dom, "clean", metric)
            ax.bar(x[i] + w / 2, v, w, color=C_CLEAN, edgecolor="k",
                   linewidth=0.4, label=L["clean"] if i == 0 else None)
            ax.annotate(f"{v:.2f}", (x[i] + w / 2, v), textcoords="offset points",
                        xytext=(0, 2), ha="center", fontsize=6.5, color="#8b0000")
        ax.set_title(f"{L['mi'] if dom == 'mit_incart' else L['ptb']} — {L[metric]}",
                     fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        if r == 0 and c == 0:
            ax.legend(fontsize=9, loc="lower right")

axes[0][0].annotate(
    L["na"] + "\n(12/14 models: 22-23/23 test records leaked)" if not CN else
    L["na"] + "\n(12/14 模型: 泄漏 22-23/23 条测试记录)",
    xy=(0.5, 0.5), xycoords="axes fraction", ha="center", fontsize=9,
    color="#444", style="italic",
    bbox=dict(boxstyle="round", fc="#fff3cd", ec="#b8860b", alpha=0.9))
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT1, dpi=150)
print(f"saved {OUT1}")

# ── 图2: 污染热图 + PTB Δ 对比 ────────────────────────────────────
fig2, (axA, axB) = plt.subplots(1, 2, figsize=(17, 6.5),
                                gridspec_kw={"width_ratios": (1, 1.25)})
fig2.suptitle(L["title2"], fontsize=14, fontweight="bold")

heat = np.array([[len(models[n]["leaked_test_records"]["mit_bih"]),
                  len(models[n]["leaked_test_records"]["incart"]),
                  len(models[n]["leaked_test_records"]["ptb"])] for n in names])
totals = np.array([10, 13, 95])
im = axA.imshow(heat / totals, cmap="Reds", vmin=0, vmax=1, aspect="auto")
axA.set_xticks(range(3), ["MIT-BIH /10", "INCART /13", "PTB /95"])
axA.set_yticks(range(len(names)), labels, fontsize=8)
for i in range(len(names)):
    for j in range(3):
        axA.text(j, i, f"{heat[i, j]}", ha="center", va="center", fontsize=7,
                 color="white" if heat[i, j] / totals[j] > 0.6 else "black")
axA.set_title(L["heat"], fontsize=11)
fig2.colorbar(im, ax=axA, label="fraction leaked")

# PTB Δ: clean - full, AUC 与 Event F1
d_auc = [get(n, "ptb", "clean", "auc") - get(n, "ptb", "full", "auc") for n in names]
d_f1 = [get(n, "ptb", "clean", "evf1") - get(n, "ptb", "full", "evf1") for n in names]
axB.bar(x - w / 2, d_auc, w, color="#31a354", label="ΔAUC")
axB.bar(x + w / 2, d_f1, w, color="#756bb1", label="ΔEvent F1")
for i in range(len(names)):
    axB.annotate(f"{d_auc[i]:+.2f}", (x[i] - w / 2, d_auc[i]),
                 textcoords="offset points", xytext=(0, 1 if d_auc[i] >= 0 else -9),
                 ha="center", fontsize=6)
    axB.annotate(f"{d_f1[i]:+.2f}", (x[i] + w / 2, d_f1[i]),
                 textcoords="offset points", xytext=(0, 1 if d_f1[i] >= 0 else -9),
                 ha="center", fontsize=6)
axB.axhline(0, color="k", lw=0.8)
axB.set_xticks(x)
axB.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
axB.set_title(L["delta"], fontsize=11)
axB.legend(fontsize=9)
axB.grid(axis="y", alpha=0.3)
note = ("clean > full: 训练见过的记录反而拖低成绩 = 过拟合失真签名;\n"
        "ecgfdr* clean AUC 暴跌是小子集构成偏差 (clean 子集异常拍占比 ~95%)") if CN else \
       ("clean > full: seen-in-training records drag scores down = overfitting;\n"
        "ecgfdr* clean-AUC collapse = composition bias (clean subset ~95% abnormal)")
axB.annotate(note, xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8,
             color="#444", style="italic")
fig2.tight_layout(rect=(0, 0, 1, 0.95))
fig2.savefig(OUT2, dpi=150)
print(f"saved {OUT2}")
