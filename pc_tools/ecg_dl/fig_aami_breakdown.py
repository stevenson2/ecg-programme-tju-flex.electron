#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_aami_breakdown.py — AAMI-category recall breakdown figure
=============================================================
Two panels:
  (a) Abnormal-class composition (share of abnormal beats by AAMI superclass,
      beat-level full set) — shows WHICH classes dominate the abnormal class.
  (b) Recall@0.5 by AAMI superclass, beat-level vs patient-level bars — shows
      WHICH classes are hard (SVEB/F low recall, V/Q high).

Input : models/aami_breakdown_exp6_deploy_beatlevel.json + _exp6_deploy.json
Output: models/figures/patient/aami_breakdown.png  (English, 150 dpi)
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
FIG_DIR = MODEL_DIR / "figures"

BEAT_JSON = MODEL_DIR / "aami_breakdown_exp6_deploy_beatlevel.json"
PAT_JSON = MODEL_DIR / "aami_breakdown_exp6_deploy.json"

THRESH = "thr_0.5"
CLASSES = ["S", "V", "F", "Q", "ALL"]
LABELS = {"S": "SVEB", "V": "VEB", "F": "Fusion",
          "Q": "Paced/Uncl.", "ALL": "ALL"}

# ---- Load ----
d_beat = json.load(open(BEAT_JSON))
d_pat = json.load(open(PAT_JSON))
pc_b = d_beat["per_class"]
pc_p = d_pat["per_class"]

# Abnormal composition (beat-level full set)
comp = {c: (pc_b[c]["n_abn"] if c in pc_b else 0) for c in ["S", "V", "F", "Q"]}
tot_abn = sum(comp.values())
comp_frac = {c: v / tot_abn * 100 for c, v in comp.items()}

# Recall @0.5 per class
rec_b = {c: (pc_b[c]["thr"][THRESH]["recall"] if c in pc_b else None)
         for c in CLASSES}
rec_p = {c: (pc_p[c]["thr"][THRESH]["recall"] if c in pc_p else None)
         for c in CLASSES}
# aggregate recall
rec_b["ALL"] = d_beat["aggregate_recall"][THRESH]["recall"]
rec_p["ALL"] = d_pat["aggregate_recall"][THRESH]["recall"]

# ---- Figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# Panel (a): abnormal composition (horizontal bar)
colors = {"S": "#d62728", "V": "#1f77b4", "F": "#ff7f0e", "Q": "#2ca02c"}
order = ["V", "Q", "S", "F"]
ys = np.arange(len(order))[::-1]
vals = [comp_frac[c] for c in order]
bars = ax1.barh(ys, vals, color=[colors[c] for c in order],
                edgecolor="black", linewidth=0.5, height=0.62)
ax1.set_yticks(ys)
ax1.set_yticklabels([LABELS[c].split("\n")[0] for c in order], fontsize=10)
ax1.invert_yaxis()
for bar, v in zip(bars, vals):
    ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
             f"{v:.1f}%", va="center", fontsize=10, fontweight="bold")
ax1.set_xlabel("Share of abnormal beats (%)", fontsize=11)
ax1.set_xlim(0, 50)
ax1.set_title("(a) Abnormal-class composition\n(beat-level, 114,566 abnormal)",
              fontsize=12, fontweight="bold")
ax1.grid(True, axis="x", alpha=0.3)

# Panel (b): recall comparison
x = np.arange(len(CLASSES))
w = 0.34
rec_b_vals = [rec_b[c] if rec_b[c] is not None else 0 for c in CLASSES]
rec_p_vals = [rec_p[c] if rec_p[c] is not None else 0 for c in CLASSES]
has_p = [rec_p[c] is not None for c in CLASSES]

b1 = ax2.bar(x - w / 2, rec_b_vals, w, label="Beat-level", color="#9467bd",
             edgecolor="black", linewidth=0.5)
b2 = ax2.bar(x + w / 2, rec_p_vals, w, label="Patient-level",
             color="#17becf", edgecolor="black", linewidth=0.5,
             alpha=0.9 if any(has_p) else 0.0)

for i, c in enumerate(CLASSES):
    if rec_b[c] is not None:
        ax2.text(i - w / 2, rec_b_vals[i] + 0.015, f"{rec_b_vals[i]:.2f}",
                 ha="center", fontsize=9, fontweight="bold", color="#5b2c8f")
    if rec_p[c] is not None:
        ax2.text(i + w / 2, rec_p_vals[i] + 0.015, f"{rec_p_vals[i]:.2f}",
                 ha="center", fontsize=9, fontweight="bold", color="#0e7c8a")
    if rec_p[c] is None:
        ax2.text(i + w / 2, 0.03, "n/a", ha="center", fontsize=8,
                 color="gray", style="italic")

ax2.set_xticks(x)
ax2.set_xticklabels([LABELS[c] for c in CLASSES], fontsize=10)
ax2.set_ylabel("Recall @ threshold 0.5", fontsize=11)
ax2.set_ylim(0, 1.12)
ax2.axhline(rec_b["ALL"], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax2.text(4.35, rec_b["ALL"] + 0.02, f"beat ALL={rec_b['ALL']:.2f}",
         fontsize=8, color="gray")
ax2.set_title("(b) Recall@0.5 by AAMI superclass\n(exp6 deploy-chain retrained)",
              fontsize=12, fontweight="bold")
ax2.legend(loc="lower right", fontsize=9)
ax2.grid(True, axis="y", alpha=0.3)

# Annotate the bottleneck
ax2.annotate("SVEB/F: morphology\nclose to normal\nin 1-beat window",
             xy=(0, 0.442), xytext=(0.6, 0.15), fontsize=8.5, color="red",
             arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
fig.suptitle("AAMI-category recall breakdown — single-beat 1s window classifier",
             fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.94])

out = FIG_DIR / "patient" / "aami_breakdown.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
