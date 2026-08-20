#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_ptbxl_record_level.py — PTB-XL 记录级验证（当前拍级/窗口级模型聚合）

用途：
  读取 PTB-XL（records500, 500Hz, 10s, 12 导联），用当前拍级 AI 模型
  （默认 exp7c INT8）对每条记录做滑窗推理，聚合成"记录级异常分数"，
  然后输出 AUC / 混淆矩阵 / 分类报告。

与合成向量的区别：
  - PTB-XL 是真实临床记录，贴合真实场景；
  - 记录级标签来自 SCP 诊断码，不用合成波形。

标签定义：
  负类（正常）：scp_codes 键集 ⊆ {NORM, SR}
  正类（异常）：其他（可通过 --positive 指定具体 SCP 码，如 MI / AFIB / PVC）

用法（WSL）：
  python3 pc_tools/ecg_dl/eval_ptbxl_record_level.py
  python3 pc_tools/ecg_dl/eval_ptbxl_record_level.py --n-max 200 --lead 1
  python3 pc_tools/ecg_dl/eval_ptbxl_record_level.py --positive MI --aggregate p95

输出：
  pc_tools/ecg_dl/models/ptbxl_record_level_eval.json
"""

import argparse
import ast
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]      # 项目根
REPO = Path(__file__).resolve().parent          # pc_tools/ecg_dl
PTBXL_DIR = ROOT / "PTB-XL_ECG"
PTBXL_CSV = PTBXL_DIR / "ptbxl_database.csv"
MODELS = REPO / "models"
OUT_JSON = MODELS / "ptbxl_record_level_eval.json"

FS = 500
REC_LEN = 5000          # 10s @500Hz
N_LEADS = 12

# 正常记录允许出现的 SCP 键
NORMAL_KEYS = {"NORM", "SR"}


def load_lead(path, lead_idx=1):
    """读 records500 .dat（12 导联交错 int16, 500Hz）单导联，返回 mV 数组。"""
    raw = np.fromfile(path, dtype='<i2')
    sig = raw.reshape(-1, N_LEADS)[:, lead_idx].astype(np.float64) / 1000.0
    return sig[:REC_LEN]


def parse_scp(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


def is_normal_record(scp: dict) -> bool:
    """负类定义：scp_codes 键集 ⊆ {NORM, SR}。"""
    return set(scp.keys()) <= NORMAL_KEYS


def classify_record(scp: dict, positive: str) -> bool:
    """按 --positive 参数决定记录是否为阳性。
    positive='abnormal': 任意非 NORM/SR 诊断码 → 阳性
    positive=<SCP码>  : 记录包含该 SCP 码 → 阳性
    """
    if positive == "abnormal":
        return not is_normal_record(scp)
    return positive in scp


def window_abnormal_probs(engine, x: np.ndarray, win: int = 250, stride: int = 125):
    """对一段信号滑窗推理，返回每个窗口的异常概率（0~1）。"""
    probs = []
    n = len(x)
    for start in range(0, n - win + 1, stride):
        window = x[start:start + win]
        result = engine.predict(window)
        # 二分类 softmax：abnormal_prob = confidence（判异常）或 1-confidence（判正常）
        p_abn = result["confidence"] if result["abnormal"] else 1.0 - result["confidence"]
        probs.append(p_abn)
    return np.asarray(probs, dtype=np.float64)


def aggregate(probs: np.ndarray, method: str) -> float:
    if method == "max":
        return float(probs.max())
    if method == "mean":
        return float(probs.mean())
    if method == "p95":
        return float(np.percentile(probs, 95))
    if method == "abn_ratio":
        return float((probs >= 0.5).mean())
    raise ValueError(f"unknown aggregate: {method}")


def confusion(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def main():
    ap = argparse.ArgumentParser(description="PTB-XL 记录级 AI 验证")
    ap.add_argument("--model", type=str,
                    default=str(MODELS / "ecg_model_exp7c_int8.tflite"),
                    help="TFLite 模型路径（默认 exp7c INT8，与固件一致）")
    ap.add_argument("--lead", type=int, default=1,
                    help="导联索引 0-11（0=I, 1=II, 2=III...）")
    ap.add_argument("--n-max", type=int, default=0,
                    help="每类最多记录数（调试用，0=全部）")
    ap.add_argument("--aggregate", choices=["max", "mean", "p95", "abn_ratio"],
                    default="max", help="窗口概率→记录级分数的方法")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="记录级判定阈值（默认 0.5）")
    ap.add_argument("--positive", type=str, default="abnormal",
                    help="阳性定义：abnormal=任意异常诊断，或指定 SCP 码如 MI/AFIB/PVC")
    ap.add_argument("--validated-only", action="store_true",
                    help="只用 validated_by_human=True 的记录（默认 True 为正式评估）")
    ap.add_argument("--skip-validated-filter", action="store_true",
                    help="不按 validated_by_human 过滤（调试用）")
    ap.add_argument("--tag", default="", help="输出 JSON 文件名后缀")
    args = ap.parse_args()

    if not PTBXL_CSV.exists():
        print(f"[ERROR] PTB-XL 数据库不存在: {PTBXL_CSV}")
        sys.exit(1)

    # 模型
    # 复用 07_pc_inference.py 的 ECGInferenceEngine（避免重复实现）
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pc_inference_script", REPO / "07_pc_inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    engine = mod.ECGInferenceEngine(model_path=args.model)

    # 读取数据库
    rows = list(csv.DictReader(open(PTBXL_CSV)))
    if not args.skip_validated_filter:
        rows = [r for r in rows if r["validated_by_human"] == "True"]
    print(f"记录总数（过滤后）: {len(rows)}")

    pos = [r for r in rows if classify_record(parse_scp(r["scp_codes"]), args.positive)]
    neg = [r for r in rows if not classify_record(parse_scp(r["scp_codes"]), args.positive)]
    print(f"正类({args.positive}): {len(pos)}  负类(正常): {len(neg)}")

    n_pos = len(pos) if args.n_max == 0 else min(args.n_max, len(pos))
    n_neg = len(neg) if args.n_max == 0 else min(args.n_max, len(neg))
    print(f"实际采样: 正 {n_pos} / 负 {n_neg}（Lead {args.lead+1}）")

    t0 = time.time()
    y_true, scores = [], []

    def process(rec):
        fname = PTBXL_DIR / (rec["filename_hr"] + ".dat")
        x = load_lead(fname, args.lead)
        probs = window_abnormal_probs(engine, x)
        return aggregate(probs, args.aggregate)

    for i, rec in enumerate(pos[:n_pos]):
        scores.append(process(rec))
        y_true.append(1)
        if (i + 1) % 200 == 0:
            print(f"  正类 {i+1}/{n_pos}  ({time.time()-t0:.0f}s)")
    for i, rec in enumerate(neg[:n_neg]):
        scores.append(process(rec))
        y_true.append(0)
        if (i + 1) % 200 == 0:
            print(f"  负类 {i+1}/{n_neg}  ({time.time()-t0:.0f}s)")

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    # AUC
    from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
    auc = float(roc_auc_score(y_true, scores))
    fpr, tpr, thr = roc_curve(y_true, scores)

    # 混淆矩阵（在给定阈值下）
    y_pred = (scores >= args.threshold).astype(int)
    cm = confusion(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                       average="binary")

    # 保存 JSON
    out_json = OUT_JSON if not args.tag else OUT_JSON.with_name(
        f"{OUT_JSON.stem}_{args.tag}.json")
    report = {
        "config": vars(args),
        "model": args.model or str(MODELS / "ecg_model_exp7c_int8.tflite"),
        "label_definition": {
            "positive": args.positive,
            "normal_keys": sorted(NORMAL_KEYS),
        },
        "n_records": {"positive": n_pos, "negative": n_neg},
        "aggregate": args.aggregate,
        "auc": auc,
        "threshold": args.threshold,
        "confusion_matrix": cm,
        "metrics": {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "specificity": float(cm["TN"] / max(1, cm["TN"] + cm["FP"])),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印
    print("\n" + "=" * 60)
    print(f"PTB-XL 记录级验证结果（aggregate={args.aggregate}, thr={args.threshold}）")
    print("=" * 60)
    print(f"样本: 正 {n_pos} / 负 {n_neg}")
    print(f"AUC: {auc:.4f}")
    print(f"混淆矩阵: {cm}")
    print(f"阈值 {args.threshold} 下: Precision={prec:.3f} Recall={rec:.3f} "
          f"F1={f1:.3f} Specificity={report['metrics']['specificity']:.3f}")
    print(f"\n报告已保存: {out_json}")


if __name__ == "__main__":
    main()
