#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate exp6 deploy figures: training curves + eval bar chart.
English labels, 150 dpi.

Usage:
  python3 fig_exp6_deploy.py                                       # default exp6_deploy (AdamW)
  python3 fig_exp6_deploy.py --tag exp6_sgd \
      --history train_history_exp6_sgd.csv --eval retrain_exp6_sgd_eval.json
"""

import argparse
import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
FIG_DIR = MODEL_DIR / "figures"
CACHE_DIR = MODEL_DIR / "deploy_match"

# ---- Figure 1: Training curves ----
def fig_training(history_csv, tag):
    rows = []
    with open(history_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})

    epochs = [r["epoch"] for r in rows]
    val_auc = [r["val_auc"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    train_auc = [r["auc"] for r in rows]
    train_loss = [r["loss"] for r in rows]
    lr = [r["learning_rate"] for r in rows]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # AUC
    ax1.plot(epochs, train_auc, "b-", alpha=0.6, label="Train AUC")
    ax1.plot(epochs, val_auc, "r-o", markersize=4, label="Val AUC")
    ax1.set_ylabel("AUC")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    best_idx = np.argmax(val_auc)
    ax1.axvline(epochs[best_idx], color="green", linestyle="--", alpha=0.5)
    ax1.annotate(f"Best: {val_auc[best_idx]:.4f} (epoch {int(epochs[best_idx])})",
                 xy=(epochs[best_idx], val_auc[best_idx]),
                 xytext=(epochs[best_idx] + 1, val_auc[best_idx] - 0.05),
                 fontsize=9, color="green")

    # Loss
    ax2.plot(epochs, train_loss, "b-", alpha=0.6, label="Train Loss")
    ax2.plot(epochs, val_loss, "r-o", markersize=4, label="Val Loss")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Learning rate
    ax3.semilogy(epochs, lr, "k-o", markersize=4)
    ax3.set_ylabel("Learning Rate")
    ax3.set_xlabel("Epoch")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f"exp6 {tag.upper().replace('_', ' ')} — Training History", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "train" / f"{tag}_train_history.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves: {out}")


# ---- Figure 2: Eval bar chart ----
def fig_eval(eval_json, tag):
    with open(eval_json, "r") as f:
        data = json.load(f)

    results = data["results"]
    anchors = data["meta"]["anchors"]["exp6c_d3_floor"]

    domains = ["MIT", "PTB"]
    d0_vals = [results["mit"]["d0"]["auc"], results["ptb"]["d0"]["auc"]]
    d3_vals = [results["mit"]["d3"]["auc"], results["ptb"]["d3"]["auc"]]
    d3_floors = [anchors["mit"]["d3_floor"], anchors["ptb"]["d3_floor"]]
    d0_targets = [anchors["mit"]["d0_target"], anchors["ptb"]["d0_target"]]

    x = np.arange(len(domains))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_d0 = ax.bar(x - width, d0_vals, width, label="D0 (Baseline)",
                     color="steelblue", edgecolor="black", linewidth=0.5)
    bars_d3 = ax.bar(x, d3_vals, width, label="D3 (Deploy)",
                     color="darkorange", edgecolor="black", linewidth=0.5)
    bars_floor = ax.bar(x + width, d3_floors, width, label="D3 Floor (old exp6c)",
                        color="lightgray", edgecolor="black", linewidth=0.5,
                        hatch="//", alpha=0.7)

    # Target markers (D0)
    for i, target in enumerate(d0_targets):
        ax.plot(x[i] - width, target, "r_", markersize=15, markeredgewidth=2,
                label="D0 Target" if i == 0 else "")

    # Value labels
    for bar in bars_d0:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_d3:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"exp6 {tag.upper().replace('_', ' ')} — D0 vs D3 AUC with Anchors",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    out = FIG_DIR / "patient" / f"{tag}_eval.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Eval chart: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate exp6 deploy-chain retrain figures")
    parser.add_argument("--tag", type=str, default="exp6_deploy",
                        help="output filename tag (default: exp6_deploy)")
    parser.add_argument("--history", type=str,
                        default="train_history_exp6_deploy.csv",
                        help="history csv filename in models/")
    parser.add_argument("--eval", type=str, default="retrain_exp6_eval.json",
                        help="eval json filename in models/deploy_match/")
    args = parser.parse_args()

    history_csv = MODEL_DIR / args.history
    eval_json = CACHE_DIR / args.eval

    print(f"Generating {args.tag} figures...")
    fig_training(history_csv, args.tag)
    fig_eval(eval_json, args.tag)
    print("Done.")


if __name__ == "__main__":
    main()
