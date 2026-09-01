#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecgfounder_hardmine.py — ECGFounder 1-lead 域距离硬负样本挖掘
================================================================================
基于 ecgfounder_embed_1lead.py 生成的特征，计算：
  - 真实 AFE 正常 10s 段（anchor）
  - PTB-XL 公共异常/正常 10s 段（candidate）
之间的距离，输出“最像真实 AFE 正常域”的公共异常记录候选，
用于后续离线硬负样本列表，而不是直接作为真实 AFE 负样本。

输出：
  pc_tools/ecg_dl/models/ecgfounder/hard_negative_candidates.json
  pc_tools/ecg_dl/models/ecgfounder/hard_negative_candidates.csv
  pc_tools/ecg_dl/models/ecgfounder/ecgfounder_distance_summary.json
"""
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "models" / "ecgfounder"
REAL_FEAT = OUT_DIR / "real_afe_1lead_features.npy"
PTB_FEAT = OUT_DIR / "ptbxl_1lead_features.npy"
REAL_META = OUT_DIR / "real_afe_1lead_meta.json"
PTB_META = OUT_DIR / "ptbxl_1lead_meta.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=60, help="输出前 N 个异常候选")
    ap.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    args = ap.parse_args()

    real_x = np.load(REAL_FEAT).astype(np.float64)
    ptb_x = np.load(PTB_FEAT).astype(np.float64)
    real_m = load_json(REAL_META)
    ptb_m = load_json(PTB_META)

    print(f"[DATA] real={real_x.shape}, ptbxl={ptb_x.shape}")

    # 归一化用于 cosine
    def norm_rows(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, 1e-12)

    if args.metric == "cosine":
        real_n = norm_rows(real_x)
        ptb_n = norm_rows(ptb_x)
        sim = ptb_n @ real_n.T          # (n_ptb, n_real)
        dist = 1.0 - sim                # cosine distance
        dist_min = dist.min(axis=1)
        dist_mean = dist.mean(axis=1)
        dist_median = np.median(dist, axis=1)
    else:
        # Euclidean: dist to each real anchor
        diff = ptb_x[:, None, :] - real_x[None, :, :]   # (n_ptb, n_real, d)
        dist = np.linalg.norm(diff, axis=2)
        dist_min = dist.min(axis=1)
        dist_mean = dist.mean(axis=1)
        dist_median = np.median(dist, axis=1)

    records = ptb_m["records"]
    tasks = ptb_m["tasks"]
    ptb_logits = np.load(OUT_DIR / "ptbxl_1lead_logits.npy")  # (n,150)
    probs = 1.0 / (1.0 + np.exp(-ptb_logits))

    rows = []
    for i, rec in enumerate(records):
        top_idx = np.argsort(probs[i])[::-1][:5]
        rows.append({
            "index": i,
            "ecg_id": rec["ecg_id"],
            "patient_id": rec["patient_id"],
            "filename_hr": rec["filename_hr"],
            "label": rec["label"],
            "scp_codes": rec["scp_codes"],
            "dist_min": float(dist_min[i]),
            "dist_mean": float(dist_mean[i]),
            "dist_median": float(dist_median[i]),
            "ecgfounder_top5": [
                {"task": tasks[int(k)], "prob": float(probs[i, int(k)])}
                for k in top_idx
            ],
        })

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "hard_negative_candidates.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {csv_path} ({len(df)} rows)")

    # 异常候选按 dist_min 升序
    abn = [r for r in rows if r["label"] == "abnormal"]
    abn.sort(key=lambda r: r["dist_min"])
    top = abn[:args.topk]
    top_json = {
        "method": f"ECGFounder 1-lead 10s segment embedding, {args.metric} distance to real AFE normal segments",
        "metric": args.metric,
        "n_real_afe_segments": int(len(real_x)),
        "n_ptbxl_records": int(len(ptb_x)),
        "n_abnormal_candidates": len(abn),
        "n_normal_controls": sum(1 for r in rows if r["label"] == "normal"),
        "hard_negative_top": top,
        "note": "这些是公共异常记录，仅作为代理硬负样本候选；禁止直接当作真实 AFE 负样本。",
    }
    out_path = OUT_DIR / "hard_negative_candidates.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(top_json, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {out_path}")

    # 摘要：真实正常 vs 公共正常 vs 公共异常的距离分布
    if args.metric == "cosine":
        real_n_self = norm_rows(real_x)
        dists_real_self = 1.0 - (real_n_self @ real_n_self.T)
        dist_real_self = dists_real_self[~np.eye(len(real_x), dtype=bool)]
    else:
        dists_real_self = np.linalg.norm(real_x[:, None, :] - real_x[None, :, :], axis=2)
        dist_real_self = dists_real_self[~np.eye(len(real_x), dtype=bool)]
    abn_min = np.array([r["dist_min"] for r in rows if r["label"] == "abnormal"])
    norm_min = np.array([r["dist_min"] for r in rows if r["label"] == "normal"])
    summary = {
        "metric": args.metric,
        "real_self_pairwise_distance": {
            "mean": float(dist_real_self.mean()),
            "median": float(np.median(dist_real_self)),
            "p10": float(np.percentile(dist_real_self, 10)),
            "p90": float(np.percentile(dist_real_self, 90)),
        },
        "pub_abnormal_to_real": {
            "min": float(abn_min.min()),
            "mean": float(abn_min.mean()),
            "median": float(np.median(abn_min)),
            "p10": float(np.percentile(abn_min, 10)),
            "p90": float(np.percentile(abn_min, 90)),
        },
        "pub_normal_to_real": {
            "min": float(norm_min.min()),
            "mean": float(norm_min.mean()),
            "median": float(np.median(norm_min)),
            "p10": float(np.percentile(norm_min, 10)),
            "p90": float(np.percentile(norm_min, 90)),
        },
    }
    with open(OUT_DIR / "ecgfounder_distance_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {OUT_DIR / 'ecgfounder_distance_summary.json'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
