#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_v5_peaks.py — v5 检测峰级诊断 (为 LUDB v6 参数扫描做准备)
===================================================================
对全部 200 条 LUDB 记录运行 v5 状态机, 收集每个检测峰:
  - 峰位置/幅度 (MWI 能量域)
  - getQRSWidth() 半高宽 / rise-fall 比 (形态学验证一旦开启就会用的特征)
  - 与前一检测峰的 RR、与最近金标准的距离 (带符号)
  - 金标准匹配结果 (TP/FP, 贪婪互斥, ±150ms)

输出:
  - models/ludb_hr_v5_peaks.csv  (每检测峰一行)
  - models/ludb_hr_v5_fp_summary.json (FP 分类统计: 近峰双计数 / T 波 / 其他)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_heartrate_ludb_v5 import (
    FS, TS, TOLERANCE_SAMP, DEFAULT_DATA_DIR,
    HRDetectorV5, HRParamsV5, chain_filter_v5, load_ludb_record,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "pc_tools" / "ecg_dl" / "models"


def match_greedy(gold, det_idx):
    gold_arr = np.asarray(gold, dtype=np.int64)
    det_arr = np.asarray(det_idx, dtype=np.int64)
    used = set()
    matched = {}
    for gi, g in enumerate(sorted(gold_arr)):
        cands = np.where(np.abs(det_arr - g) <= TOLERANCE_SAMP)[0]
        if len(cands) == 0:
            continue
        best = min(cands, key=lambda c: abs(int(det_arr[c]) - int(g)))
        if best in used:
            continue
        used.add(best)
        matched[int(best)] = gi
    return matched, used


def run_record_peaks(rid, sig, gold, gain):
    y = chain_filter_v5(sig, gain)
    det = HRDetectorV5(HRParamsV5())
    det_idx = []
    rows = []
    prev_det_idx = None

    for i in range(len(y)):
        res = det.process(y[i])
        if res["beatDetected"]:
            idx = i
            rr_prev = (idx - prev_det_idx) / FS if prev_det_idx is not None else 0.0
            w = det.get_qrs_width()
            rf = det.get_rise_fall_ratio()
            rows.append({
                "record": rid,
                "idx": idx,
                "peak_val": float(det.mwi_prev),
                "width": w,
                "rise_fall": float(rf),
                "rr_prev_ms": round(rr_prev * 1000.0, 1),
                "beat_count": det.beat_count,
                "sqi": float(det.sqi),
                "motion": int(det.motion_confirmed),
                "bpm": res["bpm"],
            })
            det_idx.append(idx)
            prev_det_idx = idx

    matched, used = match_greedy(gold, det_idx)
    gold_arr = np.asarray(gold, dtype=np.int64)
    for k, row in enumerate(rows):
        row["tp"] = k in used
        if k in matched:
            row["nearest_gold_ms"] = round((row["idx"] - gold_arr[matched[k]]) * 1000.0 / FS, 1)
            row["nearest_gold_abs_ms"] = abs(row["nearest_gold_ms"])
        else:
            # 最近金标准 (带符号)
            d = row["idx"] - gold_arr
            j = int(np.argmin(np.abs(d)))
            row["nearest_gold_ms"] = round(d[j] * 1000.0 / FS, 1)
            row["nearest_gold_abs_ms"] = round(abs(d[j]) * 1000.0 / FS, 1)

    return rows, det_idx, gold_arr


def classify(fp_row, tp_rows):
    """把 FP 峰粗分类: near-gold / T-wave-after-QRS / wide-QRS-split / other"""
    d = fp_row["nearest_gold_ms"]
    rr = fp_row["rr_prev_ms"]
    if fp_row["nearest_gold_abs_ms"] <= 150:
        return "near_gold_extra"
    if 150 < d <= 700 and 150 <= rr < 900:
        return "after_gold_Twave"
    if -700 <= d < -150:
        return "before_gold"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--records", type=str, default=None)
    ap.add_argument("--gain", type=float, default=1000.0)
    ap.add_argument("--tolerance-ms", type=float, default=150)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rec_file = data_dir.parent / "RECORDS"
    all_recs = ([l.strip().split("/")[-1] for l in rec_file.read_text().splitlines() if l.strip()]
                if rec_file.exists() else sorted({f.stem for f in data_dir.glob("*.hea")}))
    if args.records:
        all_recs = [r for r in all_recs if r in args.records.split(",")]

    peak_rows = []
    rec_stats = []
    for rid in all_recs:
        sig, gold = load_ludb_record(data_dir, rid, "ii")
        rows, det_idx, gold_arr = run_record_peaks(rid, sig, gold, args.gain)
        peak_rows.extend(rows)
        tp = sum(1 for r in rows if r["tp"])
        fp = len(rows) - tp
        fn = len(gold_arr) - tp
        rec_stats.append({"record": rid, "gold": len(gold_arr), "det": len(rows),
                          "tp": tp, "fp": fp, "fn": fn})

    Path(MODELS).mkdir(parents=True, exist_ok=True)
    csv_path = MODELS / "ludb_hr_v5_peaks.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(peak_rows[0].keys()))
        w.writeheader()
        for r in peak_rows:
            w.writerow(r)

    fps = [r for r in peak_rows if not r["tp"]]
    cls = {}
    for r in fps:
        c = classify(r, None)
        cls.setdefault(c, []).append(r)

    # FP 特征分布
    widths_fp = np.array([r["width"] for r in fps])
    widths_tp = np.array([r["width"] for r in peak_rows if r["tp"]])
    amps_fp = np.array([r["peak_val"] for r in fps])
    amps_tp = np.array([r["peak_val"] for r in peak_rows if r["tp"]])

    summary = {
        "n_records": len(all_recs),
        "n_det": len(peak_rows),
        "n_tp": int(np.sum([r["tp"] for r in peak_rows])),
        "n_fp": len(fps),
        "fp_by_class": {c: len(v) for c, v in sorted(cls.items())},
        "fp_width_pct": {q: float(np.percentile(widths_fp, q)) for q in (0, 10, 25, 50, 75, 90, 100)},
        "tp_width_pct": {q: float(np.percentile(widths_tp, q)) for q in (0, 10, 25, 50, 75, 90, 100)},
        "fp_amp_pct": {q: float(np.percentile(amps_fp, q)) for q in (0, 10, 25, 50, 75, 90, 100)},
        "tp_amp_pct": {q: float(np.percentile(amps_tp, q)) for q in (0, 10, 25, 50, 75, 90, 100)},
        "fp_rr_prev_ms_pct": {q: float(np.percentile([r["rr_prev_ms"] for r in fps], q)) for q in (0, 25, 50, 75, 100)},
        "records_fp_ge_5": [s["record"] for s in rec_stats if s["fp"] >= 5],
    }
    json_path = MODELS / "ludb_hr_v5_fp_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"peaks CSV -> {csv_path}")
    print(f"FP summary -> {json_path}")
    print(f"det={summary['n_det']} tp={summary['n_tp']} fp={summary['n_fp']}")
    print("FP 分类:", summary["fp_by_class"])
    print("FP 宽度分位:", summary["fp_width_pct"])
    print("TP 宽度分位:", summary["tp_width_pct"])
    print("FP RR_prev ms 分位:", summary["fp_rr_prev_ms_pct"])
    print("records fp>=5:", summary["records_fp_ge_5"])


if __name__ == "__main__":
    main()
