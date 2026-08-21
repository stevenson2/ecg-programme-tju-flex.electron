#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_ptbxl_mi_subtypes.py — PTB-XL MI 亚类分层（离线复用已存记录级分数）

输入: 由 eval_ptbxl_record_level.py --save-scores 生成的 npz
      （默认 KD a070_t1, Lead II, aggregate=mean, MI vs 正常）
输出: models/ptbxl_record_level_mi_subtype.json
     含每个 MI SCP 码（任意包含，允许重叠）和 3 个解剖互斥亚组的
     记录级 AUC + 患者级 bootstrap 95% CI。

用法:
  python3 pc_tools/ecg_dl/eval_ptbxl_mi_subtypes.py
"""
import argparse
import ast
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PTBXL_DIR = ROOT / "PTB-XL_ECG"
PTBXL_CSV = PTBXL_DIR / "ptbxl_database.csv"
MODELS = Path(__file__).resolve().parent / "models"
DEFAULT_NPZ = MODELS / "ptbxl_record_level_eval_kd_mi_mean_scores.npz"
OUT_JSON = MODELS / "ptbxl_record_level_mi_subtype.json"

MI_CODES = [
    "IMI", "ASMI", "ILMI", "AMI", "ALMI", "INJAS", "LMI", "INJAL",
    "IPLMI", "IPMI", "INJIN", "INJLA", "PMI", "INJIL",
]
# 解剖亚组（按 SCP diagnostic_subclass 归类）
INFERIOR_CODES = {"IMI", "ILMI", "IPLMI", "IPMI", "INJIN", "INJIL"}
ANTERIOR_CODES = {"AMI", "ALMI", "INJAL", "INJLA"}
NORMAL_KEYS = {"NORM", "SR"}

N_REPS = 500
SEED = 123


def load_rows():
    with open(PTBXL_CSV, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["validated_by_human"] == "True"]


def norm_filename(fn):
    return fn.replace("records500/", "").replace("_hr", "")


def is_normal_record(scp):
    return set(scp.keys()) <= NORMAL_KEYS


def bootstrap_auc_ci(scores, labels, patient_ids, n_reps=N_REPS, seed=SEED):
    """患者级重采样 bootstrap AUC 95% CI。"""
    scores = np.asarray(scores); labels = np.asarray(labels)
    patient_ids = np.asarray(patient_ids, dtype=str)
    unique_pats = np.unique(patient_ids)
    pat_indices = {p: np.where(patient_ids == p)[0] for p in unique_pats}
    pat_list = list(unique_pats)
    rng = np.random.default_rng(seed)
    aucs = np.zeros(n_reps)
    n_pats = len(pat_list)
    for rep in range(n_reps):
        idx = np.concatenate([pat_indices[p] for p in rng.choice(pat_list, n_pats, replace=True)])
        if len(np.unique(labels[idx])) < 2:
            aucs[rep] = 0.5
            continue
        aucs[rep] = roc_auc_score(labels[idx], scores[idx])
    return {
        "ci_lo": round(float(np.percentile(aucs, 2.5)), 4),
        "ci_hi": round(float(np.percentile(aucs, 97.5)), 4),
        "ci_width": round(float(np.percentile(aucs, 97.5) - np.percentile(aucs, 2.5)), 4),
        "bootstrap_mean": round(float(aucs.mean()), 4),
        "bootstrap_std": round(float(aucs.std()), 4),
        "n_patients": n_pats,
        "reps": n_reps,
    }


def main():
    ap = argparse.ArgumentParser(description="PTB-XL MI 亚类分层（复用已存分数）")
    ap.add_argument("--npz", type=str, default=str(DEFAULT_NPZ))
    ap.add_argument("--out", type=str, default=str(OUT_JSON))
    ap.add_argument("--reps", type=int, default=N_REPS)
    args = ap.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise SystemExit(f"[ERROR] 分数文件不存在: {npz_path}")
    z = np.load(npz_path)
    rows = load_rows()
    meta = {r["filename_hr"]: r for r in rows}
    records = []
    for fn, s, t, pid in zip(z["filenames"], z["scores"], z["y_true"], z["patient_ids"]):
        if fn not in meta:
            continue
        scp = ast.literal_eval(meta[fn]["scp_codes"])  # PTB-XL csv 是 Python literal dict
        records.append({
            "fn": fn, "score": float(s), "label": int(t), "pid": str(pid),
            "scp": scp,
        })
    normal_scores = [r["score"] for r in records if r["label"] == 0]
    normal_pids = [r["pid"] for r in records if r["label"] == 0]
    normal_labels = [0] * len(normal_scores)
    print(f"载入 {len(records)} 条记录（MI {sum(r['label'] for r in records)} / 正常 {len(normal_scores)}）")

    results = {}
    t0 = time.time()

    # 1) 单个 SCP 码：任意包含（可重叠）
    for code in MI_CODES:
        pos = [r for r in records if r["label"] == 1 and code in r["scp"]]
        if len(pos) < 20:
            results[code] = {"n": len(pos), "n_normal": len(normal_scores),
                             "note": "样本<20，不计算 AUC"}
            continue
        scores = normal_scores + [r["score"] for r in pos]
        labels = normal_labels + [1] * len(pos)
        pids = normal_pids + [r["pid"] for r in pos]
        auc = float(roc_auc_score(labels, scores))
        ci = bootstrap_auc_ci(scores, labels, pids, n_reps=args.reps)
        results[code] = {"n": len(pos), "n_normal": len(normal_scores),
                         "auc": round(auc, 4), **ci}
        print(f"  {code:6s} n={len(pos):5d} AUC={auc:.4f} 95%CI=[{ci['ci_lo']:.4f},{ci['ci_hi']:.4f}]")

    # 2) 解剖互斥亚组（同一记录只落入一组；只统计 MI 标签）
    groups = {
        "IMI_inferior_exclusive": {
            "codes": INFERIOR_CODES, "desc": "下壁 MI 亚类（IMI/ILMI/IPLMI/IPMI/INJIN/INJIL），不含其他 MI 码",
        },
        "ASMI_exclusive": {
            "codes": {"ASMI"}, "desc": "前间壁 MI（ASMI），不含其他 MI 码",
        },
        "AMI_anterior_exclusive": {
            "codes": ANTERIOR_CODES, "desc": "前壁/前侧壁 MI（AMI/ALMI/INJAL/INJLA），不含其他 MI 码",
        },
    }
    for name, spec in groups.items():
        code_set = spec["codes"]
        pos = []
        for r in records:
            if r["label"] != 1:
                continue
            mi_present = set(r["scp"].keys()) & set(MI_CODES)
            if mi_present & code_set and not (mi_present - code_set):
                pos.append(r)
        if len(pos) < 20:
            results[name] = {"n": len(pos), "n_normal": len(normal_scores),
                             "desc": spec["desc"], "note": "样本<20，不计算 AUC"}
            continue
        scores = normal_scores + [r["score"] for r in pos]
        labels = normal_labels + [1] * len(pos)
        pids = normal_pids + [r["pid"] for r in pos]
        auc = float(roc_auc_score(labels, scores))
        ci = bootstrap_auc_ci(scores, labels, pids, n_reps=args.reps)
        results[name] = {"n": len(pos), "n_normal": len(normal_scores),
                         "desc": spec["desc"], "auc": round(auc, 4), **ci}
        print(f"  {name:24s} n={len(pos):5d} AUC={auc:.4f} 95%CI=[{ci['ci_lo']:.4f},{ci['ci_hi']:.4f}]")

    output = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_npz": str(npz_path),
            "label": "MI 各亚类 vs PTB-XL 正常记录（NORM/SR）",
            "deploy_chain": "见源 eval_ptbxl_record_level.py 输出（完整板上部署链）",
            "aggregate": "mean（源 npz 由 kd_mi_mean_scores 生成）",
            "bootstrap": f"患者级重采样 {args.reps} reps, seed={SEED}, 95% CI 百分位",
        },
        "mi_code_counts": {c: sum(1 for r in records if r["label"] == 1 and c in r["scp"])
                           for c in MI_CODES},
        "results": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()