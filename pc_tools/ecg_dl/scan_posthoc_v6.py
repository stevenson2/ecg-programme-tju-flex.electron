#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_posthoc_v6.py — v5 检测峰事后滤波粗扫描 (排序候选门限组合)
================================================================
注意: 事后滤波不是完整状态机仿真 (被拒峰会改变 rr/refractory/阈值学习),
仅用于快速排序候选参数, 最终数字必须以 verify_heartrate_ludb_v6.py 全仿真为准。
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_heartrate_ludb_v5 import FS, TOLERANCE_SAMP

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "pc_tools" / "ecg_dl" / "models"


def load_peaks():
    rows = list(csv.DictReader(open(MODELS / "ludb_hr_v5_peaks.csv", encoding="utf-8")))
    for r in rows:
        r["idx"] = int(r["idx"])
        r["peak_val"] = float(r["peak_val"])
        r["width"] = int(r["width"])
        r["rise_fall"] = float(r["rise_fall"])
        r["rr_prev_ms"] = float(r["rr_prev_ms"])
        r["beat_count"] = int(r["beat_count"])
        r["bpm"] = int(r["bpm"])
        r["sqi"] = float(r["sqi"])
        r["tp"] = (r["tp"] == "True")
    recs = {}
    for r in rows:
        recs.setdefault(r["record"], []).append(r)
    return recs


def match_greedy(gold, det_idx):
    gold_arr = np.asarray(gold, dtype=np.int64)
    det_arr = np.asarray(det_idx, dtype=np.int64)
    used = set()
    matched = set()
    for g in gold_arr:
        cands = np.where(np.abs(det_arr - g) <= TOLERANCE_SAMP)[0]
        if len(cands) == 0:
            continue
        best = min(cands, key=lambda c: abs(int(det_arr[c]) - int(g)))
        if best in used:
            continue
        used.add(int(best))
        matched.add(int(best))
    return matched


def evaluate(recs, golds, rule):
    tot = {"gold": 0, "det": 0, "tp": 0, "fp": 0, "fn": 0}
    errs = []
    details = []
    for rid, peaks in recs.items():
        # 逐拍应用事后门限
        valid_peaks = []
        valid_widths = []
        kept = []
        for p in peaks:
            accept = True
            bc = p["beat_count"]
            w = p["width"]
            amp = p["peak_val"]
            rr = p["rr_prev_ms"] / 1000.0
            if rule.get("min_rr") is not None and 0 < rr < rule["min_rr"]:
                accept = False
            if rule["min_conf"] is not None and bc >= rule["min_conf"]:
                if w < rule.get("wmin", 0):
                    accept = False
                if w > rule.get("wmax", 10**9):
                    accept = False
                if rule.get("rf_min") is not None and p["rise_fall"] < rule["rf_min"]:
                    accept = False
                if rule.get("rf_max") is not None and p["rise_fall"] > rule["rf_max"]:
                    accept = False
                if rule.get("amp_consistency") is not None and len(valid_peaks) >= 3:
                    mean = float(np.mean(valid_peaks[-8:]))
                    if mean > 1e-7 and abs(amp - mean) / mean > rule["amp_consistency"]:
                        accept = False
                if rule.get("width_ratio") is not None and len(valid_widths) >= 3:
                    med_w = float(np.median(valid_widths[-8:]))
                    if med_w >= 1 and w > rule["width_ratio"] * med_w:
                        accept = False
            if rule.get("amp_frac_prev") is not None and len(valid_peaks) >= 1:
                prev = valid_peaks[-1]
                if prev > 1e-7 and amp < rule["amp_frac_prev"] * prev and rr < 0.9:
                    accept = False
            if accept:
                kept.append(p)
                valid_peaks.append(amp)
                valid_widths.append(w)

        gold = golds[rid]
        matched = match_greedy(gold, [p["idx"] for p in kept])
        tp = len(matched)
        det = len(kept)
        fp = det - tp
        fn = len(gold) - tp
        tot["gold"] += len(gold)
        tot["det"] += det
        tot["tp"] += tp
        tot["fp"] += fp
        tot["fn"] += fn
        # BPM 近似: 保留检测峰的 RR 中位数 (v5 bpm 列在 beat 帧恒 0, 不能用)
        idx_arr = np.asarray([p["idx"] for p in kept], dtype=np.float64)
        if len(idx_arr) >= 2:
            rr = np.diff(idx_arr) / FS
            rr = rr[(rr >= 0.3) & (rr <= 2.0)]
            db = 60.0 / float(np.median(rr)) if len(rr) else 0.0
        else:
            db = 0.0
        if gold is not None:
            g = np.asarray(gold, dtype=np.float64)
            if len(g) >= 2:
                rr = np.diff(g) / FS
                rr = rr[(rr >= 0.3) & (rr <= 2.0)]
                gb = 60.0 / float(np.median(rr)) if len(rr) else 0.0
            else:
                gb = 0.0
            if gb > 0 and db > 0:
                errs.append(abs(gb - db))
        details.append({"record": rid, "gold": len(gold), "det": det, "tp": tp,
                        "fp": fp, "fn": fn, "det_bpm": db})
    se = tot["tp"] / max(1, tot["tp"] + tot["fn"])
    ppv = tot["tp"] / max(1, tot["tp"] + tot["fp"])
    f1 = 2 * se * ppv / (se + ppv) if (se + ppv) else 0.0
    mae = float(np.mean(errs)) if errs else 0.0
    return {"se": se, "ppv": ppv, "f1": f1, "mae": mae,
            "tp": tot["tp"], "fp": tot["fp"], "fn": tot["fn"], "det": tot["det"],
            "n_err": len(errs), "details": details}


def main():
    recs = load_peaks()
    # gold 位置从 v5 明细拿不到, 重新载入 LUDB (轻量)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from verify_heartrate_ludb_v5 import DEFAULT_DATA_DIR, load_ludb_record
    golds = {}
    for rid in recs:
        _, gold = load_ludb_record(DEFAULT_DATA_DIR, rid, "ii")
        golds[rid] = gold

    base = {"min_conf": None, "wmin": 0, "wmax": 10**9}
    b = evaluate(recs, golds, base)
    print(f"v5 baseline (post-hoc no-gate)  Se={b['se']:.4f} PPV={b['ppv']:.4f} "
          f"F1={b['f1']:.4f} MAE={b['mae']:.2f} TP={b['tp']} FP={b['fp']} FN={b['fn']}")

    rules = []
    # 固定宽度门 (激活点扫描)
    for min_conf in (5, 8, 12):
        for wmax in (40, 45, 50, 55, 60):
            for wmin in (10, 15, 20):
                rules.append({"name": f"fixW conf{min_conf} w{wmin}-{wmax}",
                              "min_conf": min_conf, "wmin": wmin, "wmax": wmax})
    # 自适应宽度比
    for min_conf in (5, 8, 12):
        for wr in (1.8, 2.0, 2.5, 3.0):
            rules.append({"name": f"adaW conf{min_conf} ratio{wr}",
                          "min_conf": min_conf, "wmin": 0, "wmax": 10**9,
                          "width_ratio": wr})
    # 固定宽度 + 幅度一致性
    for wmax in (45, 50, 60):
        for ac in (0.25, 0.30, 0.35):
            rules.append({"name": f"fixW+amp conf8 w10-{wmax} ac{ac}",
                          "min_conf": 8, "wmin": 10, "wmax": wmax, "amp_consistency": ac})
    # 幅度前拍比
    for frac in (0.35, 0.45, 0.55):
        rules.append({"name": f"ampPrev{frac}", "min_conf": None,
                      "amp_frac_prev": frac})
    # 固定宽度 + 前拍幅度比
    for wmax in (50, 60):
        for frac in (0.45, 0.55):
            rules.append({"name": f"fixW conf8 10-{wmax} +prevAmp{frac}",
                          "min_conf": 8, "wmin": 10, "wmax": wmax,
                          "amp_frac_prev": frac})
    # rf 比 (T 波 rf~65-70, QRS rf~11-33)
    for min_conf in (5, 8):
        for rfmax in (35, 40, 45, 50):
            rules.append({"name": f"rfmax conf{min_conf} {rfmax}",
                          "min_conf": min_conf, "wmin": 0, "wmax": 10**9,
                          "rf_max": rfmax})
    # rf + 幅度一致性
    for rfmax in (35, 40, 45):
        for ac in (0.25, 0.30, 0.35):
            rules.append({"name": f"rf{rfmax}+ac{ac} conf8",
                          "min_conf": 8, "wmin": 0, "wmax": 10**9,
                          "rf_max": rfmax, "amp_consistency": ac})
    # rf + 自适应宽度
    for rfmax in (35, 40, 45):
        for wr in (2.0, 2.5):
            rules.append({"name": f"rf{rfmax}+adaW{wr} conf8",
                          "min_conf": 8, "wmin": 0, "wmax": 10**9,
                          "rf_max": rfmax, "width_ratio": wr})
    # rf + 前拍幅度比
    for rfmax in (40, 45):
        for frac in (0.45, 0.55):
            rules.append({"name": f"rf{rfmax}+prevAmp{frac} conf8",
                          "min_conf": 8, "wmin": 0, "wmax": 10**9,
                          "rf_max": rfmax, "amp_frac_prev": frac})
    # 幅度一致性 (无宽度/rf)
    for min_conf in (5, 8, 12):
        for ac in (0.20, 0.25, 0.30):
            rules.append({"name": f"ac{ac} conf{min_conf}",
                          "min_conf": min_conf, "wmin": 0, "wmax": 10**9,
                          "amp_consistency": ac})
    # 前拍幅度比 (无其他)
    for min_conf in (5, 8):
        for frac in (0.35, 0.45, 0.55, 0.65):
            rules.append({"name": f"prevAmp{frac} conf{min_conf}",
                          "min_conf": min_conf, "wmin": 0, "wmax": 10**9,
                          "amp_frac_prev": frac})
    # MIN_RR 提升 (事后近似: rr<min_rr 的峰整体剔除)
    for min_rr in (0.50, 0.52, 0.55):
        rules.append({"name": f"minRR{min_rr}",
                      "min_conf": None, "wmin": 0, "wmax": 10**9,
                      "min_rr": min_rr})

    out = []
    for rule in rules:
        r = evaluate(recs, golds, rule)
        if r["se"] >= 0.95:
            score = r["ppv"] + r["se"] - 0.95 - 0.02 * r["mae"]  # 粗排序分
        else:
            score = -10.0 + r["se"]
        out.append((score, r, rule))

    out.sort(key=lambda x: -x[0])
    print(f"\ntop 40 candidate rules (post-hoc, Se>=0.95, sorted by heuristic score):")
    for score, r, rule in out[:40]:
        print(f"  {rule['name']:<42} Se={r['se']*100:6.2f} PPV={r['ppv']*100:6.2f} "
              f"F1={r['f1']:.4f} MAE={r['mae']:6.2f} TP={r['tp']:4d} FP={r['fp']:4d} FN={r['fn']:3d}")

    json.dump([{"rule": x[2], "metrics": {k: v for k, v in x[1].items() if k != "details"}}
               for x in out[:60]],
              open(MODELS / "ludb_hr_v6_posthoc_top.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\n(posthoc top60 -> models/ludb_hr_v6_posthoc_top.json)")


if __name__ == "__main__":
    main()
