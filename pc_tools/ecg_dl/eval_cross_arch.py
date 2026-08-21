#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_cross_arch.py — 跨架构部署链失配对照评估
================================================================
读取 train_cross_arch.py 训练好的 3×2 模型，在 deploy_match 缓存的
MIT/PTB 测试 beat 上计算：
  A: 训练链模型 + 训练链测试
  B: 训练链模型 + 部署链测试（失配）
  C: 部署链模型 + 部署链测试（修复）
输出每个域 A/B/C AUC、Δ(B−A)、Δ(C−A)、患者级 bootstrap CI。

用法:
  python3 eval_cross_arch.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODELS_DIR
from eval_deploy_match import CACHE_DIR
from data.patient_split import (
    build_mit_patient_map,
    build_incart_patient_map,
    build_ptb_patient_map,
)

N_REPS = 500
SEED = 123


def add_channel(x):
    return x.astype(np.float32)[..., np.newaxis]


def bootstrap_ci_for_delta(a_scores, b_scores, labels, rec_ids, pmap,
                           n_reps=N_REPS, seed=SEED):
    """Return CI of Δ = AUC(A) - AUC(B), plus n_patients."""
    rid_to_pat = {}
    for rid in np.unique(rec_ids):
        rid_i = int(rid)
        rid_to_pat[rid_i] = pmap.get(rid_i, f"unknown_{rid_i}")
    pat_of_beat = np.array([rid_to_pat[int(r)] for r in rec_ids])
    unique_pats = np.unique(pat_of_beat)
    pat_indices = {p: np.where(pat_of_beat == p)[0] for p in unique_pats}
    pat_list = list(pat_indices.keys())
    n_pats = len(pat_list)
    rng = np.random.default_rng(seed)
    deltas = np.zeros(n_reps)
    for rep in range(n_reps):
        idx = np.concatenate([pat_indices[p] for p in rng.choice(pat_list, n_pats, replace=True)])
        y = labels[idx]
        if len(np.unique(y)) < 2:
            deltas[rep] = 0.0
            continue
        auc_a = roc_auc_score(y, a_scores[idx])
        auc_b = roc_auc_score(y, b_scores[idx])
        deltas[rep] = auc_b - auc_a  # mismatch Δ = deploy - baseline
    return {
        "ci_lo": round(float(np.percentile(deltas, 2.5)), 4),
        "ci_hi": round(float(np.percentile(deltas, 97.5)), 4),
        "ci_width": round(float(np.percentile(deltas, 97.5) - np.percentile(deltas, 2.5)), 4),
        "mean": round(float(np.mean(deltas)), 4),
        "std": round(float(np.std(deltas)), 4),
        "n_patients": n_pats,
        "reps": n_reps,
    }


def load_domain_cache(domain: str):
    if domain == "mit":
        path = CACHE_DIR / "mit_deploy_match.npz"
        pmap0 = build_mit_patient_map()
        pmap1 = build_incart_patient_map()
        pmap = dict(pmap0)
        pmap.update({rid + 100000: "inc_" + pat for rid, pat in pmap1.items()})
    elif domain == "ptb":
        path = CACHE_DIR / "ptb_deploy_match.npz"
        pmap = build_ptb_patient_map()
    else:
        raise ValueError(domain)
    z = np.load(path)
    return {
        "beats_baseline": z["beats_baseline"],
        "beats_deploy": z["beats_deploy"],
        "labels": z["labels"],
        "record_ids": z["record_ids"],
        "pmap": pmap,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=str, default=str(MODELS_DIR / "cross_arch"))
    ap.add_argument("--out", type=str, default=str(MODELS_DIR / "cross_arch_eval.json"))
    ap.add_argument("--archs", nargs="+", default=["lstm_cnn", "cnn_standard", "resnet1d"])
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    results = {}
    t0 = time.time()

    for domain in ("mit", "ptb"):
        cache = load_domain_cache(domain)
        xb = add_channel(cache["beats_baseline"])
        xd = add_channel(cache["beats_deploy"])
        labels = cache["labels"]
        rec_ids = cache["record_ids"]
        pmap = cache["pmap"]
        print("=" * 70, flush=True)
        print(f"[{domain}] beats={len(labels)} "
              f"norm={(labels==0).sum()} abn={(labels==1).sum()}", flush=True)
        print("=" * 70, flush=True)

        for arch in args.archs:
            model_b_path = models_dir / f"{arch}_baseline.h5"
            model_d_path = models_dir / f"{arch}_deploy.h5"
            if not model_b_path.exists() or not model_d_path.exists():
                print(f"  [{arch}] missing model, skip")
                continue
            mb = tf.keras.models.load_model(str(model_b_path), compile=False)
            md = tf.keras.models.load_model(str(model_d_path), compile=False)

            pb_A = mb.predict(xb, batch_size=512, verbose=0)[:, 1]
            pb_B = mb.predict(xd, batch_size=512, verbose=0)[:, 1]
            pd_C = md.predict(xd, batch_size=512, verbose=0)[:, 1]

            auc_A = float(roc_auc_score(labels, pb_A))
            auc_B = float(roc_auc_score(labels, pb_B))
            auc_C = float(roc_auc_score(labels, pd_C))
            delta_BA = auc_B - auc_A
            delta_CA = auc_C - auc_A
            delta_CB = auc_C - auc_B
            ci_BA = bootstrap_ci_for_delta(pb_A, pb_B, labels, rec_ids, pmap)
            ci_CA = bootstrap_ci_for_delta(pb_A, pd_C, labels, rec_ids, pmap)
            results.setdefault(arch, {})[domain] = {
                "n_beats": int(len(labels)),
                "auc_A_baseline_train_test": round(auc_A, 4),
                "auc_B_baseline_train_deploy_test": round(auc_B, 4),
                "auc_C_deploy_train_deploy_test": round(auc_C, 4),
                "delta_B_minus_A": round(delta_BA, 4),
                "delta_C_minus_A": round(delta_CA, 4),
                "delta_C_minus_B": round(delta_CB, 4),
                "ci_delta_B_minus_A": ci_BA,
                "ci_delta_C_minus_A": ci_CA,
                "n_records": int(len(np.unique(rec_ids))),
            }
            print(f"  [{arch}] A={auc_A:.4f} B={auc_B:.4f} C={auc_C:.4f} "
                  f"ΔBA={delta_BA:+.4f} CI=[{ci_BA['ci_lo']:.4f},{ci_BA['ci_hi']:.4f}]",
                  flush=True)
            # free GPU/CPU memory
            del mb, md
            tf.keras.backend.clear_session()

    output = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "跨架构部署链失配对照（A/B/C）",
            "architectures": args.archs,
            "training_protocol": "MIT+INCART+PTB, patient-split seed42, "
                                 "ptb_abn_max=10000, domain_balanced, FocalLoss+AdamW",
            "deploy_chain": "eval_deploy_match.deployment_chain (D3)",
            "cache": "mit_deploy_match.npz / ptb_deploy_match.npz",
            "bootstrap": f"患者级重采样 {N_REPS} reps, seed={SEED}, 95% CI",
        },
        "results": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()