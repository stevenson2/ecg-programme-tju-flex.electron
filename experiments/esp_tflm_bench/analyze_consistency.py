#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_consistency.py — 解析板上 RESULT 日志并计算 PC/ESP32 输出一致性
================================================================================
输入:
  - experiments/esp_tflm_bench/consistency_pc.json (PC BUILTIN_REF 每拍概率)
  - 板上串口日志 (默认 C:/esp/esp_tflm_bench/monitor_consistency.log)
输出:
  - experiments/esp_tflm_bench/consistency_result.json
验收:
  - |ΔAUC| <= 0.01
  - mean|Δp| 记录，目标 <=0.01
  - max|Δp| 记录
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
PC_JSON = Path(__file__).resolve().parent / "consistency_pc.json"
BOARD_LOG = Path(__file__).resolve().parent / "logs" / "monitor_consistency.log"
OUT_JSON = Path(__file__).resolve().parent / "consistency_result.json"

THRESHOLDS = [0.35, 0.50, 0.60]


def parse_board_log(path):
    board = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"RESULT,(\d+),(-?\d+),(-?\d+),([-0-9.eE+]+),([-0-9.eE+]+)", line)
            if m:
                idx = int(m.group(1))
                board[idx] = {
                    "q0": int(m.group(2)),
                    "q1": int(m.group(3)),
                    "p1": float(m.group(4)),
                    "p1_double": float(m.group(5)),
                }
    return board


def main():
    pc = json.loads(PC_JSON.read_text(encoding="utf-8"))
    n = len(pc["labels"])
    board = parse_board_log(BOARD_LOG)
    if len(board) != n:
        print(f"[warn] board lines {len(board)} != PC {n}")
    pc_p = np.array(pc["p_single"], dtype=np.float64)
    pc_pd = np.array(pc["p_double"], dtype=np.float64)
    labels = np.array(pc["labels"], dtype=np.int32)
    idxs = list(range(n))
    bd_p = np.array([board[i]["p1"] for i in idxs if i in board], dtype=np.float64)
    bd_pd = np.array([board[i]["p1_double"] for i in idxs if i in board], dtype=np.float64)
    bd_q = np.array([[board[i]["q0"], board[i]["q1"]] for i in idxs if i in board], dtype=np.int32)
    pc_raw = np.array(pc["raw_output"], dtype=np.int32)

    def auc(y, p):
        if len(np.unique(y)) < 2:
            return None
        return float(roc_auc_score(y, p))

    metrics = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pc_json": str(PC_JSON.relative_to(REPO)),
            "board_log": str(BOARD_LOG.relative_to(REPO)),
            "n_beats": n,
            "n_normal": int((labels == 0).sum()),
            "n_abnormal": int((labels == 1).sum()),
            "pc_semantics": "single dequantized output (no second softmax)",
            "pc_op_resolver": pc["meta"].get("pc_op_resolver", "UNKNOWN"),
            "note": "PC reference resolver is used to match MCU TFLM integer kernel family; "
                    "default XNNPACK is not used for acceptance comparison.",
        },
        "raw_q_exact_matches": int((bd_q == pc_raw[:len(bd_q)]).all(axis=1).sum()),
        "raw_q_mismatches": int((bd_q != pc_raw[:len(bd_q)]).any(axis=1).sum()),
        "p_single": {
            "mean_abs_delta": float(np.mean(np.abs(bd_p - pc_p))),
            "max_abs_delta": float(np.max(np.abs(bd_p - pc_p))),
            "pearson": float(pearsonr(pc_p, bd_p)[0]),
            "spearman": float(spearmanr(pc_p, bd_p)[0]),
            "auc_pc": auc(labels, pc_p),
            "auc_board": auc(labels, bd_p),
            "delta_auc": auc(labels, bd_p) - auc(labels, pc_p),
        },
        "p_double": {
            "mean_abs_delta": float(np.mean(np.abs(bd_pd - pc_pd))),
            "max_abs_delta": float(np.max(np.abs(bd_pd - pc_pd))),
            "spearman": float(spearmanr(pc_pd, bd_pd)[0]),
        },
        "agreement": {},
    }
    for thr in THRESHOLDS:
        agree = float(np.mean((pc_p >= thr) == (bd_p >= thr)))
        metrics["agreement"][f"thr_{thr:.2f}"] = agree

    metrics["accept"] = {
        "delta_auc_ok": abs(metrics["p_single"]["delta_auc"]) <= 0.01,
        "mean_dp_recorded": metrics["p_single"]["mean_abs_delta"],
        "mean_dp_target": 0.01,
        "mean_dp_ok": metrics["p_single"]["mean_abs_delta"] <= 0.01,
        "max_dp_recorded": metrics["p_single"]["max_abs_delta"],
        "verdict": "PASS" if abs(metrics["p_single"]["delta_auc"]) <= 0.01
                   and metrics["p_single"]["mean_abs_delta"] <= 0.01 else "CHECK",
    }
    OUT_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
