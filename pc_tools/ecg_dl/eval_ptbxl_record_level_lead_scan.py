#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_ptbxl_record_level_lead_scan.py — 12 导联逐导扫描汇总（离线）

输入: eval_ptbxl_record_level.py --lead 0..11 输出 JSON 和 save-scores npz
      （KD a070_t1, MI vs 正常, aggregate=mean）
输出: models/ptbxl_record_level_lead_scan.json
      各导联 AUC + 患者级 bootstrap 95%CI + Youden/F1 最优操作点

用法:
  python3 pc_tools/ecg_dl/eval_ptbxl_record_level_lead_scan.py
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(__file__).resolve().parent / "models"
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
OUT_JSON = MODELS / "ptbxl_record_level_lead_scan.json"
N_REPS = 500
SEED = 123


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


def main():
    ap = argparse.ArgumentParser(description="12 导联逐导扫描汇总")
    ap.add_argument("--models-dir", type=str, default=str(MODELS))
    ap.add_argument("--out", type=str, default=str(OUT_JSON))
    ap.add_argument("--reps", type=int, default=N_REPS)
    ap.add_argument("--tag-prefix", type=str, default="ptbxl_record_level_eval_kd_mi_mean_lead")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    results = {}
    print("导联扫描汇总（KD MI mean, 部署链）")
    t0 = time.time()
    for lead in range(12):
        tag = f"{args.tag_prefix}{lead}"
        json_path = models_dir / f"{tag}.json"
        npz_path = models_dir / f"{tag}.npz"
        if not json_path.exists():
            results[LEAD_NAMES[lead]] = {"lead_index": lead, "note": "missing json"}
            print(f"  {LEAD_NAMES[lead]:3s} missing {json_path.name}")
            continue
        rep = json.load(open(json_path, encoding="utf-8"))
        auc = rep.get("auc")
        if npz_path.exists():
            z = np.load(npz_path)
            ci = bootstrap_auc_ci(z["scores"], z["y_true"], z["patient_ids"], n_reps=args.reps)
        else:
            ci = None
        results[LEAD_NAMES[lead]] = {
            "lead_index": lead,
            "lead_name": LEAD_NAMES[lead],
            "auc": auc,
            "n_records": rep.get("n_records"),
            "best_youden_point": rep.get("best_youden_point"),
            "best_f1_point": rep.get("best_f1_point"),
            "bootstrap_ci": ci,
        }
        if auc is not None:
            ci_txt = f"[{ci['ci_lo']:.4f},{ci['ci_hi']:.4f}]" if ci else "N/A"
            print(f"  {LEAD_NAMES[lead]:3s} AUC={auc:.4f} 95%CI={ci_txt}")

    # 排序显示
    order = sorted(((k, v["auc"]) for k, v in results.items() if v.get("auc") is not None),
                   key=lambda x: x[1], reverse=True)
    print("\n导联 AUC 排序：")
    for name, auc in order:
        print(f"  {name:3s} {auc:.4f}")

    output = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "PTB-XL 12 导联逐导扫描（KD a070_t1, MI vs NORM, aggregate=mean, 完整部署链）",
            "bootstrap": f"患者级重采样 {args.reps} reps, seed={SEED}, 95% CI 百分位",
        },
        "results": results,
        "rank_by_auc": [(name, results[name]["auc"]) for name, _ in order],
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()