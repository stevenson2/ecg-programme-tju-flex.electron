#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_kd_pilot.py — KD 试点图表生成器
产出 (到 reports/kd_pilot/):
 1. screen_heatmap.png   — α×T 网格 → 均值 D3 AUC 热力图
 2. kd_vs_baseline.png   — KD_BEST vs 基线 D0/D3 AUC 对比 (MIT/PTB)
 3. train_curves.png     — KD_BEST 与基线 val_auc 训练曲线对比
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports" / "kd_pilot"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_deploy_eval(out_name: str) -> dict:
    """Load models/deploy_match/<out_name>.json -> {mit: {d0,d3}, ptb: {d0,d3}}."""
    p = MODEL_DIR / "deploy_match" / out_name
    if not p.exists():
        print(f"  !! missing {p}")
        return None
    with open(p) as f:
        d = json.load(f)
    return {
        "mit": {"d0": d["results"]["mit"]["d0"]["auc"],
                "d3": d["results"]["mit"]["d3"]["auc"]},
        "ptb": {"d0": d["results"]["ptb"]["d0"]["auc"],
                "d3": d["results"]["ptb"]["d3"]["auc"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="kd_screen_summary.json",
                    help="screen summary json in models/")
    ap.add_argument("--kd-best", default="",
                    help="KD_BEST eval json in models/deploy_match/ (e.g. kd_p40_a050_t3_eval.json)")
    ap.add_argument("--baseline", default="retrain_exp6_hp005_eval.json",
                    help="baseline eval json in models/deploy_match/")
    ap.add_argument("--kd-history", default="",
                    help="KD_BEST train history csv in models/")
    ap.add_argument("--baseline-history", default="train_history_exp6_hp005.csv",
                    help="baseline train history csv in models/")
    args = ap.parse_args()

    # ---- 1. Screen heatmap ----
    screen_path = MODEL_DIR / args.screen
    if screen_path.exists():
        with open(screen_path) as f:
            screen = json.load(f)
        alphas = sorted({r["alpha"] for r in screen["runs"]})
        temps = sorted({r["temperature"] for r in screen["runs"]})
        Z = np.full((len(alphas), len(temps)), np.nan)
        for r in screen["runs"]:
            ai = alphas.index(r["alpha"])
            ti = temps.index(r["temperature"])
            Z[ai, ti] = r.get("mean_d3", np.nan)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(Z, cmap="YlGnBu", vmin=0.7, vmax=0.95)
        ax.set_xticks(range(len(temps)))
        ax.set_xticklabels([f"T={t}" for t in temps])
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels([f"α={a}" for a in alphas])
        for i in range(len(alphas)):
            for j in range(len(temps)):
                v = Z[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="black" if v > 0.83 else "white")
        ax.set_title("KD Screen: mean(MIT,PTB) D3 AUC  (α × T)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "screen_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"  saved screen_heatmap.png")
    else:
        print(f"  !! screen summary missing: {screen_path}")

    # ---- 2. KD vs baseline bar ----
    bl = load_deploy_eval(args.baseline)
    if args.kd_best:
        kd = load_deploy_eval(args.kd_best)
    else:
        # fallback: any kd_p40 eval json found
        cands = sorted((MODEL_DIR / "deploy_match").glob("kd_p40_*_eval.json"))
        kd = load_deploy_eval(cands[0].name) if cands else None
    if bl and kd:
        domains = ["MIT", "PTB"]
        chains = ["D0", "D3"]
        x = np.arange(len(domains) * len(chains))
        width = 0.35
        bl_vals = [bl["mit"]["d0"], bl["mit"]["d3"], bl["ptb"]["d0"], bl["ptb"]["d3"]]
        kd_vals = [kd["mit"]["d0"], kd["mit"]["d3"], kd["ptb"]["d0"], kd["ptb"]["d3"]]
        fig, ax = plt.subplots(figsize=(8, 5))
        b1 = ax.bar(x - width/2, bl_vals, width, label="Baseline (SGD hp005)", color="#8ecae6")
        b2 = ax.bar(x + width/2, kd_vals, width, label="KD Best", color="#fb8500")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d} {c}" for d in domains for c in chains])
        ax.set_ylabel("AUC")
        ax.set_ylim(0.5, 1.0)
        ax.axhline(0.5, color="grey", ls="--", lw=0.5)
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.008,
                        f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        ax.legend()
        ax.set_title("KD Best vs Baseline — Deploy-Chain D0/D3 AUC")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "kd_vs_baseline.png", dpi=150)
        plt.close(fig)
        print(f"  saved kd_vs_baseline.png")
        # print delta summary
        for d in domains:
            for c in chains:
                key = d.lower()
                delta = kd[key][c.lower()] - bl[key][c.lower()]
                print(f"  Δ{d} {c}: {bl[key][c.lower()]:.4f} -> {kd[key][c.lower()]:.4f} = {delta:+.4f}")
    else:
        print("  !! baseline or kd-best eval missing — skip bar chart")

    # ---- 3. Training curves ----
    def load_history(csv_name):
        p = MODEL_DIR / csv_name
        if not p.exists():
            print(f"  !! history missing: {p}")
            return None
        import csv
        rows = []
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append(r)
        return rows

    kh = load_history(args.kd_history) if args.kd_history else None
    bh = load_history(args.baseline_history)
    if bh and (kh or True):
        fig, ax = plt.subplots(figsize=(8, 5))
        if bh:
            ax.plot([float(r["epoch"]) for r in bh],
                    [float(r["val_auc"]) for r in bh],
                    label="Baseline val_auc", color="#8ecae6")
        if kh:
            ax.plot([float(r["epoch"]) for r in kh],
                    [float(r["val_auc"]) for r in kh],
                    label="KD Best val_auc", color="#fb8500")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val_auc")
        ax.legend()
        ax.set_title("KD Best vs Baseline — Training val_auc")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "train_curves.png", dpi=150)
        plt.close(fig)
        print(f"  saved train_curves.png")
    else:
        print("  !! no history — skip curves")

    print(f"\nReport dir: {REPORT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
