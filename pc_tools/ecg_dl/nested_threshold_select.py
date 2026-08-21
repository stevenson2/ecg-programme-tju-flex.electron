#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nested_threshold_select.py — 患者级嵌套阈值选择（修复测试集偷看漏洞）
====================================================================
输入: eval_ptbxl_record_level.py --save-scores 产生的 npz
方法:
  1. 按 patient_id 做 50/50 划分（同一患者的所有记录同侧，seed=42）
  2. 在选择侧扫描阈值，取 Youden 最优点
  3. 将该阈值冻结，只在评估侧报告 Se/Sp/PPV/F1
  4. 对比"在全集上选阈值"的乐观数字，量化偷看偏倚
用法: python3 nested_threshold_select.py <scores.npz> [--seed 42]
"""
import sys
import json
import argparse
import numpy as np


def metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    se = tp / max(1, tp + fn)
    sp = tn / max(1, tn + fp)
    ppv = tp / max(1, tp + fp)
    f1 = 2 * se * ppv / (se + ppv) if (se + ppv) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity": round(se, 4), "specificity": round(sp, 4),
            "precision": round(ppv, 4), "f1": round(f1, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step", type=float, default=0.005)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    y = d["y_true"]; s = d["scores"]; pids = d["patient_ids"]
    print(f"载入 {args.npz}: {len(y)} 条记录, 正类 {int(y.sum())}")

    uniq_pids = np.unique(pids)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(uniq_pids)
    sel_pids = set(uniq_pids[: len(uniq_pids) // 2])
    sel_mask = np.asarray([p in sel_pids for p in pids])
    ev_mask = ~sel_mask
    print(f"患者级 50/50: 选择侧 {len(sel_pids)} 患者/{int(sel_mask.sum())} 条, "
          f"评估侧 {len(uniq_pids)-len(sel_pids)} 患者/{int(ev_mask.sum())} 条")

    # AUC（两侧分别）
    from sklearn.metrics import roc_auc_score
    auc_all = roc_auc_score(y, s)
    auc_sel = roc_auc_score(y[sel_mask], s[sel_mask])
    auc_ev = roc_auc_score(y[ev_mask], s[ev_mask])
    print(f"AUC: 全集 {auc_all:.4f} | 选择侧 {auc_sel:.4f} | 评估侧 {auc_ev:.4f}")

    # 选择侧扫描 → 冻结阈值
    thrs = np.arange(0.01, 0.99, args.step)
    best_thr, best_youden = None, -1
    for t in thrs:
        m = metrics(y[sel_mask], (s[sel_mask] >= t).astype(int))
        youden = m["sensitivity"] + m["specificity"]
        if youden > best_youden:
            best_youden, best_thr = youden, float(t)
    print(f"\n选择侧 Youden 最优阈值: {best_thr:.3f} (Youden={best_youden:.4f})")

    # 冻结阈值 → 评估侧
    nested = metrics(y[ev_mask], (s[ev_mask] >= best_thr).astype(int))
    print(f"\n== 嵌套评估（阈值 {best_thr:.3f}, 仅评估侧 {int(ev_mask.sum())} 条）==")
    print(json.dumps(nested, indent=2, ensure_ascii=False))

    # 对照：全集选阈值的乐观数字
    best_thr_naive, best_y_naive = None, -1
    for t in thrs:
        m = metrics(y, (s >= t).astype(int))
        youden = m["sensitivity"] + m["specificity"]
        if youden > best_y_naive:
            best_y_naive, best_thr_naive = youden, float(t)
    naive_on_eval = metrics(y[ev_mask], (s[ev_mask] >= best_thr_naive).astype(int))
    print(f"\n== 对照: 全集选阈值 {best_thr_naive:.3f} 再套到评估侧（偷看偏倚演示）==")
    print(json.dumps(naive_on_eval, indent=2, ensure_ascii=False))

    d_se = naive_on_eval["sensitivity"] - nested["sensitivity"]
    d_sp = naive_on_eval["specificity"] - nested["specificity"]
    print(f"\n偷看偏倚: ΔSe={d_se:+.4f}, ΔSp={d_sp:+.4f}（正值=全集选阈值虚高）")

    out = {
        "npz": args.npz, "seed": args.seed,
        "auc": {"all": auc_all, "select": auc_sel, "eval": auc_ev},
        "n_patients": {"select": len(sel_pids), "eval": len(uniq_pids)-len(sel_pids)},
        "n_records": {"select": int(sel_mask.sum()), "eval": int(ev_mask.sum())},
        "nested": {"threshold": best_thr, **nested},
        "naive_on_eval": {"threshold": best_thr_naive, **naive_on_eval},
        "peek_bias": {"delta_se": d_se, "delta_sp": d_sp},
    }
    out_path = args.npz.replace(".npz", "_nested.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
