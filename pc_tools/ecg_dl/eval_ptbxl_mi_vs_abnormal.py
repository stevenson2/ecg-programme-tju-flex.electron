#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_ptbxl_mi_vs_abnormal.py — 阴性集扩充：MI vs 其他异常（STTC/CD/HYP）

输入: eval_ptbxl_record_level.py --negative abnormal --save-scores 生成 npz
      （KD a070_t1, Lead II, aggregate=mean, MI vs 排除 MI 后的其他异常）
输出: models/ptbxl_mi_vs_abnormal.json
      含整体 AUC + 按 STTC/CD/HYP 分组的 AUC，均带患者级 bootstrap 95%CI。

用法:
  python3 pc_tools/ecg_dl/eval_ptbxl_mi_vs_abnormal.py
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
DEFAULT_NPZ = MODELS / "ptbxl_record_level_eval_kd_mi_vs_abnormal.npz"
OUT_JSON = MODELS / "ptbxl_mi_vs_abnormal.json"
N_REPS = 500
SEED = 123


def load_scp_classes():
    mapping = {}
    with open(PTBXL_DIR / "scp_statements.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code = r[""].strip()
            cls = r.get("diagnostic_class", "").strip()
            if code and cls:
                mapping[code] = cls
    return mapping


def bootstrap_auc_ci(scores, labels, patient_ids, n_reps=N_REPS, seed=SEED):
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


def compute_auc_ci(pos, neg, n_reps):
    scores = [r["score"] for r in pos] + [r["score"] for r in neg]
    labels = [1] * len(pos) + [0] * len(neg)
    pids = [r["pid"] for r in pos] + [r["pid"] for r in neg]
    if len(pos) < 2 or len(neg) < 2 or len(np.unique(scores)) < 2:
        return {"n_positive": len(pos), "n_negative": len(neg),
                "note": "样本不足或分数无变化，不计算 AUC"}
    auc = float(roc_auc_score(labels, scores))
    ci = bootstrap_auc_ci(scores, labels, pids, n_reps=n_reps)
    return {"n_positive": len(pos), "n_negative": len(neg),
            "auc": round(auc, 4), **ci}


def main():
    ap = argparse.ArgumentParser(description="MI vs 其他异常（STTC/CD/HYP）")
    ap.add_argument("--npz", type=str, default=str(DEFAULT_NPZ))
    ap.add_argument("--out", type=str, default=str(OUT_JSON))
    ap.add_argument("--reps", type=int, default=N_REPS)
    args = ap.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise SystemExit(f"[ERROR] 分数文件不存在: {npz_path}")
    z = np.load(npz_path)
    mapping = load_scp_classes()
    rows = [r for r in csv.DictReader(open(PTBXL_CSV, encoding="utf-8"))
            if r["validated_by_human"] == "True"]
    meta = {r["filename_hr"]: r for r in rows}
    records = []
    for fn, s, t, pid in zip(z["filenames"], z["scores"], z["y_true"], z["patient_ids"]):
        if fn not in meta:
            continue
        scp = ast.literal_eval(meta[fn]["scp_codes"])
        classes = {mapping.get(c) for c in scp.keys() if mapping.get(c) and mapping.get(c) != "NORM"}
        records.append({"fn": fn, "score": float(s), "label": int(t),
                        "pid": str(pid), "scp": scp, "classes": classes})
    pos = [r for r in records if r["label"] == 1]
    neg_all = [r for r in records if r["label"] == 0]
    print(f"载入 {len(records)} 条记录（MI {len(pos)} / 其他异常 {len(neg_all)}）")

    results = {"all_abnormal": compute_auc_ci(pos, neg_all, args.reps)}
    print(f"  all_abnormal  n_pos={len(pos)} n_neg={len(neg_all)} "
          f"AUC={results['all_abnormal'].get('auc')}")

    for cls in ("STTC", "CD", "HYP"):
        neg_cls = [r for r in neg_all if cls in r["classes"]]
        res = compute_auc_ci(pos, neg_cls, args.reps)
        results[cls] = res
        print(f"  {cls:6s} n_pos={len(pos)} n_neg={len(neg_cls)} AUC={res.get('auc')}")

    # 补充：三类（STTC/CD/HYP）都不含的其他异常（多为节律/形态描述码，如 SBRAD/LVOLT）
    hard_neg = [r for r in neg_all if not (r["classes"] & {"STTC", "CD", "HYP"})]
    results["other_unclassified"] = {
        "n_positive": len(pos), "n_negative": len(hard_neg),
        "classes_found": sorted({c for r in hard_neg for c in r["classes"]}),
        "note": "这些记录不含 STTC/CD/HYP 诊断类，通常只含 NORM/SR + 节律/描述性 SCP 码",
    }
    print(f"  其他未分类负例（无 STTC/CD/HYP）: {len(hard_neg)}")

    output = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_npz": str(npz_path),
            "positive": "MI",
            "negative": "排除 MI 后的 PTB-XL 异常记录（STTC/CD/HYP 可重叠）",
            "deploy_chain": "完整板上部署链（源 eval_ptbxl_record_level.py）",
            "bootstrap": f"患者级重采样 {args.reps} reps, seed={SEED}, 95% CI 百分位",
        },
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()