#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_exp6_deploy.py — exp6 deploy-chain retrain evaluation
==========================================================
Evaluates a deploy-chain retrained ResNet-Large on cached deployment-chain test beats
(mit_deploy_match.npz, ptb_deploy_match.npz). Produces retrain_exp6_eval.json.

Usage:
  python3 eval_exp6_deploy.py                                  # default: best_resnet_large_exp6_deploy.h5
  python3 eval_exp6_deploy.py --model best_resnet_large_exp6_sgd.h5 \
      --out retrain_exp6_sgd_eval.json                          # SGD A/B arm
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# ---- Paths ----
BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
CACHE_DIR = MODEL_DIR / "deploy_match"
MODEL_PATH = MODEL_DIR / "best_resnet_large_exp6_deploy.h5"
OUT_JSON = CACHE_DIR / "retrain_exp6_eval.json"
# ---- Anchors from old exp6c ----
ANCHORS = {
    "mit": {"d3_floor": 0.8990, "d0_target": 0.8942},
    "ptb": {"d3_floor": 0.7184, "d0_target": 0.8232},
}

THRESHOLDS = [0.35, 0.50]


def add_channel_dim(x):
    """(N, 250) -> (N, 250, 1) float32."""
    return x.astype(np.float32)[..., np.newaxis]


def compute_metrics(y_true, prob):
    """Return dict: auc, rec/prec/f1 at each theta."""
    result = {"auc": float(roc_auc_score(y_true, prob))}
    for thr in THRESHOLDS:
        pred = (prob >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0)
        result[f"thr_{thr}"] = {
            "recall": float(rec),
            "precision": float(prec),
            "f1": float(f1),
        }
    return result


def evaluate_domain(model, cache, domain_name):
    """Evaluate model on D0 (baseline) and D3 (deploy) for one domain."""
    beats_baseline = cache["beats_baseline"]  # (N, 250)
    beats_deploy = cache["beats_deploy"]       # (N, 250)
    labels = cache["labels"]                    # (N,)

    print(f"  {domain_name}: {len(labels)} beats "
          f"({int(np.sum(labels))} abnormal)")

    # D0 (baseline)
    x0 = add_channel_dim(beats_baseline)
    prob0 = model.predict(x0, batch_size=512, verbose=0)[:, 1]
    m0 = compute_metrics(labels, prob0)
    print(f"    D0 (baseline): AUC={m0['auc']:.4f}")

    # D3 (deploy)
    x3 = add_channel_dim(beats_deploy)
    prob3 = model.predict(x3, batch_size=512, verbose=0)[:, 1]
    m3 = compute_metrics(labels, prob3)
    print(f"    D3 (deploy):   AUC={m3['auc']:.4f}")

    return {"d0": m0, "d3": m3, "n_beats": int(len(labels)),
            "n_abnormal": int(np.sum(labels))}


def verdict(domain, d3_auc, d0_auc):
    """Generate verdict lines vs old exp6c anchors."""
    anchors = ANCHORS[domain]
    dom_label = domain.upper()
    lines = []

    # Recovery floor check
    floor = anchors["d3_floor"]
    if d3_auc >= floor:
        lines.append(f"{dom_label} D3 AUC >= {floor:.4f}: PASS")
    else:
        lines.append(f"{dom_label} D3 AUC >= {floor:.4f}: FAIL "
                     f"(shortfall {floor - d3_auc:.4f})")

    # Target comparison
    target = anchors["d0_target"]
    delta = d3_auc - target
    if delta >= 0:
        lines.append(f"{dom_label} D3 AUC vs {target:.4f}: recovered by {delta:+.4f}")
    else:
        lines.append(f"{dom_label} D3 AUC vs {target:.4f}: shortfall {delta:+.4f}")

    # D0 check
    d0_delta = d0_auc - target
    if d0_delta >= 0:
        lines.append(f"{dom_label} D0 AUC vs target {target:.4f}: above by {d0_delta:+.4f}")
    else:
        lines.append(f"{dom_label} D0 AUC vs target {target:.4f}: shortfall {d0_delta:+.4f}")

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate exp6 deploy-chain retrained model on cached test beats")
    parser.add_argument("--model", type=str, default="best_resnet_large_exp6_deploy.h5",
                        help="model filename in models/ (default: best_resnet_large_exp6_deploy.h5)")
    parser.add_argument("--out", type=str, default="retrain_exp6_eval.json",
                        help="output json filename in models/deploy_match/ (default: retrain_exp6_eval.json)")
    args = parser.parse_args()

    model_path = MODEL_DIR / args.model
    out_json = CACHE_DIR / args.out

    print("=" * 60)
    print("eval_exp6_deploy.py — exp6 deploy-chain retrain evaluation")
    print("=" * 60)

    # Load model
    import tensorflow as tf
    print(f"\nLoading model: {model_path.name}")
    model = tf.keras.models.load_model(str(model_path), compile=False)
    print("  Model loaded (compile=False).")

    # Load cached data
    mit_cache = np.load(CACHE_DIR / "mit_deploy_match.npz")
    ptb_cache = np.load(CACHE_DIR / "ptb_deploy_match.npz")

    results = {}
    all_verdicts = {}

    # MIT
    print("\n[MIT]")
    results["mit"] = evaluate_domain(model, mit_cache, "MIT")
    all_verdicts["mit"] = verdict(
        "mit", results["mit"]["d3"]["auc"], results["mit"]["d0"]["auc"])

    # PTB
    print("\n[PTB]")
    results["ptb"] = evaluate_domain(model, ptb_cache, "PTB")
    all_verdicts["ptb"] = verdict(
        "ptb", results["ptb"]["d3"]["auc"], results["ptb"]["d0"]["auc"])

    # Assemble output
    output = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_path.name,
            "purpose": "exp6 deploy-chain retrain evaluation on cached test beats",
            "anchors": {
                "exp6c_d3_floor": ANCHORS,
                "description": "Old exp6c D3 = recovery floor; D0 = target"
            },
        },
        "results": results,
        "verdicts": all_verdicts,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for domain in ["mit", "ptb"]:
        r = results[domain]
        print(f"\n{domain.upper()}:")
        print(f"  D0 AUC: {r['d0']['auc']:.4f}  "
              f"D3 AUC: {r['d3']['auc']:.4f}  "
              f"Δ = {r['d3']['auc'] - r['d0']['auc']:+.4f}")
        for thr in THRESHOLDS:
            k = f"thr_{thr}"
            d0 = r["d0"][k]
            d3 = r["d3"][k]
            print(f"  θ={thr}:  D0 R={d0['recall']:.4f} P={d0['precision']:.4f} "
                  f"F1={d0['f1']:.4f}  |  "
                  f"D3 R={d3['recall']:.4f} P={d3['precision']:.4f} "
                  f"F1={d3['f1']:.4f}")
    print("\nVERDICTS:")
    for domain in ["mit", "ptb"]:
        print(f"  {domain}:")
        for line in all_verdicts[domain]:
            print(f"    {line}")

    print(f"\nOutput: {out_json}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
