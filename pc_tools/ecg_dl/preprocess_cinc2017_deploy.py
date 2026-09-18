#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preprocess_cinc2017_deploy.py — cinc2017 N 类正常拍部署链预处理 (R18 H4)
================================================================================
预注册 runs/R18_PREREG.md §1-H4 冻结:
  - 只取 N 类 (REFERENCE.csv label=N) 记录 —— A/O/~ 一律不入库
    (A=节律级标签与拍形态语义冲突; O 异质; ~ 保留后续 valid 头素材候选)。
  - 链: corrected_deployment_chain(sig, 300) (与 INCART/PTB 同链, 原生 300Hz
    内部精确有理重采样 500Hz → comb×2 → HP/LP → 2:1 → 因果 HP0.5@250)。
  - R 波: cinc2017 无标注 → 简化 Pan-Tompkins (5-15Hz 带通 + 0.15s 平方
    积分 + 自适应阈值 + 距离约束 + ±50ms 局部极值精化); 每记录 QC 断言
    (检出>0, 中位 RR ∈ [0.33, 1.5]s)。
  - 窗: strict-skip R 居中 250 样本 (extract_beats_deploy 同公式, 固件
    z-score: 总体标准差, std<1e-6 → std=1)。
  - rid = 200000 + 记录号 (独占区间 [200001,208528], split_guard 断言)。
  - 每记录限抽 max_per_rec 拍 (rng seed 42, 患者多样性优先)。

输出 (ECG_DATA): cinc2017_processed_deploy_causal_{beats,labels,record_ids}.npy
审计: models/deploy_match/cinc2017_preprocess_audit.json (逐记录统计 + 与既有
三域 test 交集断言 + QRS 检出 QC)。

泄漏纪律: cinc2017 患者与 MIT/INCART/PTB 结构性不相交 (不同采集设备/人群);
交集断言显式执行 (防御性)。划分由 split_guard.compute_cinc2017_split 权威重算。
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_deploy_match import corrected_deployment_chain

BASE = Path(__file__).resolve().parent
ECG_DATA = Path("/home/devcontainers/ecg_data")
CINC_DIR = ECG_DATA / "cinc2017" / "training2017"
REF_CSV = CINC_DIR / "REFERENCE.csv"
OUT_PREFIX = ECG_DATA / "cinc2017_processed_deploy_causal"
AUDIT = BASE / "models" / "deploy_match" / "cinc2017_preprocess_audit.json"

RID_OFFSET = 200000
WIN = 250
HALF = WIN // 2
FS = 250
MAX_PER_REC = 8
SEED = 42


def detect_qrs(sig250: np.ndarray):
    """简化 Pan-Tompkins: 5-15Hz 带通 → 平方 → 0.15s 移动积分 → 自适应阈值。"""
    b, a = scipy_signal.butter(2, [5.0, 15.0], btype="band", fs=FS)
    xf = scipy_signal.filtfilt(b, a, np.asarray(sig250, dtype=np.float64))
    integ = np.convolve(xf * xf, np.ones(int(0.15 * FS)) / int(0.15 * FS),
                        mode="same")
    thr = 0.3 * float(np.percentile(integ, 99))
    if thr <= 0:
        return np.array([], dtype=np.int64)
    peaks, _ = scipy_signal.find_peaks(integ, height=thr,
                                       distance=int(0.33 * FS))
    refined = []
    for p in peaks:
        lo = max(0, p - int(0.05 * FS))
        hi = min(len(xf), p + int(0.05 * FS) + 1)
        if hi <= lo:
            continue
        refined.append(lo + int(np.argmax(np.abs(xf[lo:hi]))))
    return np.unique(np.asarray(refined, dtype=np.int64))


def extract_windows(sig250, r_idx):
    """strict-skip R 居中 250 窗 + 固件 z-score (extract_beats_deploy 同公式)。"""
    beats = []
    n = len(sig250)
    for ri in r_idx:
        lo, hi = ri - HALF, ri - HALF + WIN
        if lo < 0 or hi > n:
            continue
        w = sig250[lo:hi].copy()
        mu = np.mean(w)
        std = np.sqrt(np.var(w))
        if std < 1e-6:
            std = 1.0
        beats.append((w - mu) / std)
    return np.asarray(beats, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 条 N 类记录 (0=全部; 冒烟用)")
    ap.add_argument("--max-per-rec", type=int, default=MAX_PER_REC)
    args = ap.parse_args()

    t0 = time.time()
    import wfdb

    labels = {}
    with open(REF_CSV, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                labels[row[0].strip()] = row[1].strip()
    n_records = {k: sum(1 for v in labels.values() if v == k)
                 for k in ("N", "A", "O", "~")}
    print(f"[CINC] REFERENCE: {n_records}, total={len(labels)}", flush=True)
    assert sum(n_records.values()) == 8528, "记录数与挑战赛官方 8528 不符"

    n_recs = sorted([r for r, v in labels.items() if v == "N"])
    if args.limit:
        n_recs = n_recs[: args.limit]

    rng = np.random.default_rng(SEED)
    all_beats, all_rids = [], []
    per_rec, qc_fail = [], []
    for i, rec in enumerate(n_recs):
        try:
            sig = wfdb.rdrecord(str(CINC_DIR / rec)).p_signal[:, 0].astype(np.float64)
            fs_in = int(wfdb.rdheader(str(CINC_DIR / rec)).fs)
        except Exception as e:  # noqa: BLE001
            qc_fail.append({"rec": rec, "stage": "read", "error": str(e)})
            continue
        if fs_in != 300:
            qc_fail.append({"rec": rec, "stage": "fs", "fs": fs_in})
            continue
        s250 = corrected_deployment_chain(sig, fs_in)
        r_idx = detect_qrs(s250)
        beats = extract_windows(s250, r_idx)
        if len(beats) == 0:
            qc_fail.append({"rec": rec, "stage": "no_beats"})
            continue
        rr = np.diff(r_idx) / FS if len(r_idx) > 1 else np.array([1.0])
        med_rr = float(np.median(rr))
        if not (0.33 <= med_rr <= 1.5):
            qc_fail.append({"rec": rec, "stage": "rr_qc", "median_rr_s": med_rr})
            continue
        if len(beats) > args.max_per_rec:
            sel = rng.choice(len(beats), args.max_per_rec, replace=False)
            beats = beats[sel]
        rid = RID_OFFSET + int(rec[1:])
        all_beats.append(beats)
        all_rids.append(np.full(len(beats), rid, dtype=np.int64))
        per_rec.append({"rec": rec, "rid": rid, "n_beats_total": len(r_idx),
                        "n_beats_kept": len(beats),
                        "median_rr_s": round(med_rr, 3),
                        "hr_bpm": round(60.0 / med_rr, 1)})
        if (i + 1) % 500 == 0:
            print(f"[CINC] {i + 1}/{len(n_recs)} ...", flush=True)

    beats_arr = np.concatenate(all_beats)
    rids_arr = np.concatenate(all_rids)
    labels_arr = np.zeros(len(beats_arr), dtype=np.int32)
    np.save(str(OUT_PREFIX) + "_beats.npy", beats_arr)
    np.save(str(OUT_PREFIX) + "_labels.npy", labels_arr)
    np.save(str(OUT_PREFIX) + "_record_ids.npy", rids_arr)
    print(f"[CINC] saved {len(beats_arr)} beats from "
          f"{len(np.unique(rids_arr))} N-class records "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---------- 守卫划分 + 防御性交集断言 ----------
    from data.split_guard import get_guard, compute_cinc2017_split
    sp = compute_cinc2017_split()
    st = sp["stats"]
    # 与既有三域 test 记录的结构性交集 (rid 空间不同, 防御性断言)
    overlaps = {}
    from data.split_guard import load_arrays as _la
    for t in ("mit_bih", "incart", "ptb"):
        r_other = np.asarray(_la(t)[2])
        inter = np.intersect1d(np.unique(rids_arr), np.unique(r_other))
        overlaps[t] = int(len(inter))
        assert len(inter) == 0, f"cinc2017 rid 与 {t} 数组重叠: {inter[:10]}"
    g = get_guard("cinc2017")
    g.assert_train_only(g.train_record_ids(), context="self-check(train)")

    audit = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "preprocess_cinc2017_deploy.py",
        "reference_counts": n_records,
        "policy": {"classes_used": ["N"], "max_per_rec": args.max_per_rec,
                   "chain": "corrected_deployment_chain (D3 + HP0.5@250)",
                   "window": "strict-skip R-centered 250, firmware z-score",
                   "qrs_detector": "simplified Pan-Tompkins 5-15Hz band"},
        "records_processed": len(n_recs),
        "records_ok": len(per_rec),
        "records_qc_fail": len(qc_fail),
        "qc_fail_sample": qc_fail[:20],
        "beats_total": int(len(beats_arr)),
        "split_stats": st,
        "overlap_with_existing_arrays": overlaps,
        "per_record": per_rec,
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print(f"[CINC] audit -> {AUDIT} (split {st['n_train']}/{st['n_val']}/"
          f"{st['n_test']} patients)", flush=True)


if __name__ == "__main__":
    main()
