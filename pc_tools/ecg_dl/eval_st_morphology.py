#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_st_morphology.py — 下一步待办#6: ST 形态学预研（模块 4 地基, 不入主结果）
=============================================================================
任务: 可行性统计报告 —— ①LUDB 波形边界验证 J 点定位精度（固定偏移近似 vs
     人工标注 QRS 终点）; ②PTB-XL 子类标签（IMI/ASMI/STTC/NORM）+ J+80ms 测量
     统计 ST 偏移与标签关系。

设计:
  J 点近似 (移动设备零训练方案): R 峰 + 固定 50ms (QRS ~100ms, 峰在中点前)
  ST80 = mean(sig[J+60ms : J+100ms]) − 基线 (PQ 段 = R 峰前 80-40ms 均值)
  LUDB 金标准: .ii 标注 ')' = QRS 终点 (J 点), 'N' = QRS 峰
  PTB-XL 子类: IMI(下壁) 用 III/aVF, ASMI(前间壁) 用 V2/V3, STTC 用全体, NORM 对照

数据:
  LUDB: ECG-Database/lobachevsky-.../data/ (200 条, 500Hz, 12 导联)
  PTB-XL: PTB-XL_ECG/ (records500, 21837 条, 500Hz, 12 导联)
输出: models/st_morphology_eval.json
用法 (WSL): python3 eval_st_morphology.py [--ludb-only] [--ptbxl-only] [--n-max N]
"""
import argparse
import ast
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

ROOT = Path(__file__).resolve().parents[2]     # 项目根
REPO = Path(__file__).resolve().parent
MODELS = REPO / "models"
OUT_JSON = MODELS / "st_morphology_eval.json"

LUDB_DIR = (ROOT / "ECG-Database" / "lobachevsky-university-electrocardiography-database-1.0.1"
            / "lobachevsky-university-electrocardiography-database-1.0.1" / "data")
PTBXL_DIR = ROOT / "PTB-XL_ECG"
PTBXL_CSV = PTBXL_DIR / "ptbxl_database.csv"

FS = 500
J_OFFSET_MS = 50          # J 点近似: R 峰后 50ms
ST_WIN = (60, 100)        # ST80 窗口: J+60..J+100ms (均值)
BASE_WIN = (-80, -40)     # PQ 基线: R 峰前 80-40ms

# PTB-XL 子类代码 (scp_codes 键)
IMI_CODES = {'IMI', 'ILMI', 'IPLMI', 'LMI', 'PMI'}
ASMI_CODES = {'ASMI'}
STTC_CODES = {'ISC_', 'ISCAL', 'ISCIN', 'ISCIL', 'ISCAS', 'ISCLA',
              'NST_', 'STD_', 'STE_', 'INVT', 'TAB_', 'LOWT'}
LEAD_LUT = {'I': 0, 'II': 1, 'III': 2, 'AVR': 3, 'AVL': 4, 'AVF': 5,
            'V1': 6, 'V2': 7, 'V3': 8, 'V4': 9, 'V5': 10, 'V6': 11}


def parse_scp(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


def bandpass5_15(x, fs=FS):
    b, a = scipy_signal.butter(2, [5.0, 15.0], btype='band', fs=fs)
    return scipy_signal.filtfilt(b, a, x)


def detect_r_peaks(x, fs=FS):
    """简化 Pan-Tompkins (与 eval_rhythm_af_ptbxl.py 一致)."""
    y = bandpass5_15(x, fs)
    d = np.diff(y, prepend=y[0])
    sq = d * d
    win = int(0.15 * fs)
    m = np.convolve(sq, np.ones(win) / win, mode='same')
    thr = 0.30 * np.max(m)
    peaks, _ = scipy_signal.find_peaks(m, height=thr, distance=int(0.2 * fs))
    return peaks


def measure_st80(sig, r_peaks, fs=FS):
    """逐拍 ST80 (mV) 相对 PQ 基线; 返回 (拍级数组, 无效拍数)."""
    j_off = int(J_OFFSET_MS / 1000 * fs)
    w0, w1 = int(ST_WIN[0] / 1000 * fs), int(ST_WIN[1] / 1000 * fs)
    b0, b1 = int(BASE_WIN[0] / 1000 * fs), int(BASE_WIN[1] / 1000 * fs)
    vals, bad = [], 0
    for r in r_peaks:
        j = r + j_off
        if j + w1 >= len(sig) or r + b0 < 0:
            bad += 1
            continue
        base = np.mean(sig[r + b0:r + b1])
        st = np.mean(sig[j + w0:j + w1]) - base
        vals.append(st)
    return np.array(vals), bad


def load_ptbxl_lead(rec, lead_name):
    fname = PTBXL_DIR / (rec['filename_hr'] + '.dat')
    raw = np.fromfile(fname, dtype='<i2')
    idx = LEAD_LUT[lead_name]
    sig = raw.reshape(-1, 12)[:, idx].astype(np.float64) / 1000.0
    return sig[:5000]


def ludb_jpoint_validation():
    """子任务 A: 固定 50ms J 点近似 vs LUDB 人工标注 QRS 终点."""
    import wfdb
    print("=" * 60)
    print(f"LUDB J 点定位精度验证 ({LUDB_DIR})")
    print("=" * 60)
    recs = sorted(f.stem for f in LUDB_DIR.glob('*.dat'))
    print(f"记录数: {len(recs)}")
    errs, rel_errs, n_pairs = [], [], 0
    per_rec = []
    t0 = time.time()
    for rid in recs:
        try:
            hdr = wfdb.rdheader(str(LUDB_DIR / rid))
            lead_idx = hdr.sig_name.index('ii')
            rec = wfdb.rdrecord(str(LUDB_DIR / rid), channels=[lead_idx])
            sig = rec.p_signal[:, 0].astype(np.float64)
            ann = wfdb.rdann(str(LUDB_DIR / rid), 'ii')
        except Exception:
            continue
        # 金标准 J 点: ')' 结束且其段内含 'N' (QRS 峰) 的波形段 = QRS 终点
        # LUDB 符号: '(' 开段, 段内 p/N/t 分别 = P 峰/QRS 峰/T 终点, ')' 关段
        gold_j = []
        in_seg, seg_has_qrs = False, False
        for s, sym in zip(ann.sample, ann.symbol):
            if sym == '(':
                in_seg, seg_has_qrs = True, False
            elif sym == 'N':
                seg_has_qrs = True
            elif sym == ')' and in_seg:
                if seg_has_qrs:
                    gold_j.append(int(s))
                in_seg = False
        # 自动: R 峰 (N 或检测) + 固定 50ms
        auto_r = detect_r_peaks(sig)
        auto_j = auto_r + int(0.050 * FS)
        # 配对: 金标准 J 与自动 J 最近匹配 (≤100ms)
        gj = np.asarray(gold_j, dtype=np.int64)
        aj = np.asarray(auto_j, dtype=np.int64)
        for a in aj:
            d = np.abs(gj - a)
            i = int(np.argmin(d)) if len(d) else -1
            if len(d) and d[i] <= 50:  # 100ms 容差
                errs.append(float(a - gj[i]))
                rel_errs.append(float(a - gj[i]) / 250.0)
                n_pairs += 1
        if len(gold_j) > 0:
            per_rec.append(len(auto_r) / len(gold_j))
    errs = np.array(errs)
    print(f"  配对: {n_pairs} (耗时 {time.time()-t0:.0f}s)")
    print(f"  自动/金标准峰数比: 中位 {np.median(per_rec):.2f}")
    print(f"  J 点误差 (样本, 中位): {np.median(errs):.1f} (IQR "
          f"{np.percentile(errs,25):.1f}~{np.percentile(errs,75):.1f})")
    print(f"  J 点误差 MAE: {np.mean(np.abs(errs)):.1f} 样本 = "
          f"{np.mean(np.abs(errs))/5:.1f} ms")
    print(f"  J 点误差 |err|≤25ms 占比: {np.mean(np.abs(errs)<=12.5)*100:.1f}% "
          f"(≤50ms: {np.mean(np.abs(errs)<=25)*100:.1f}%)")
    return {
        "n_records": len(recs), "n_pairs": int(n_pairs),
        "median_err_samples": float(np.median(errs)),
        "iqr_err_samples": [float(np.percentile(errs, 25)),
                            float(np.percentile(errs, 75))],
        "mae_samples": float(np.mean(np.abs(errs))),
        "mae_ms": float(np.mean(np.abs(errs)) / 5.0),
        "pct_within_25ms": float(np.mean(np.abs(errs) <= 12.5)),
        "pct_within_50ms": float(np.mean(np.abs(errs) <= 25)),
        "median_peak_ratio": float(np.median(per_rec)),
    }


def ptbxl_st80_analysis(n_max=0):
    """子任务 B: PTB-XL 子类标签 + J+80ms 测量."""
    from sklearn.metrics import roc_auc_score
    print("=" * 60)
    print("PTB-XL ST80 子类判别分析")
    print("=" * 60)
    if not PTBXL_CSV.exists():
        print(f"PTB-XL 数据库不存在: {PTBXL_CSV}")
        sys.exit(1)
    rows = list(csv.DictReader(open(PTBXL_CSV)))
    val = [r for r in rows if r['validated_by_human'] == 'True']

    # 互斥分组: IMI(下壁)/ASMI(前间壁)/STTC/NORM
    # 注: IMI 与 ASMI 解剖互斥; STTC (缺血/ST 改变) 常为 MI 伴随表现, 仅排除 MI 解剖类
    groups = {}
    for g, codes, lead in [("IMI", IMI_CODES, 'III'),
                           ("ASMI", ASMI_CODES, 'V2')]:
        other = (IMI_CODES | ASMI_CODES) - codes
        groups[g] = [r for r in val
                     if any(c in parse_scp(r['scp_codes']) for c in codes)
                     and not any(c in parse_scp(r['scp_codes']) for c in other)]
    groups["STTC"] = [r for r in val
                      if any(c in parse_scp(r['scp_codes']) for c in STTC_CODES)
                      and not any(c in parse_scp(r['scp_codes'])
                                  for c in IMI_CODES | ASMI_CODES)]
    groups["NORM"] = [r for r in val
                      if 'NORM' in parse_scp(r['scp_codes'])
                      and not any(c in parse_scp(r['scp_codes'])
                                  for c in IMI_CODES | ASMI_CODES | STTC_CODES)]
    for g, l in groups.items():
        print(f"  {g}: {len(l)} (导联 { {'IMI':'III','ASMI':'V2','STTC':'II','NORM':'II'}[g] })")

    t0 = time.time()
    results = {}
    raw_st80 = {}   # 组 → 记录级 ST80 数组 (判别阶段复用)
    for g, recs in groups.items():
        lead = {'IMI': 'III', 'ASMI': 'V2', 'STTC': 'II', 'NORM': 'II'}[g]
        n = len(recs) if n_max == 0 else min(n_max, len(recs))
        st80s, n_bad = [], 0
        for rec in recs[:n]:
            try:
                sig = load_ptbxl_lead(rec, lead)
            except Exception:
                continue
            peaks = detect_r_peaks(sig)
            vals, bad = measure_st80(sig, peaks)
            n_bad += bad
            if len(vals) >= 3:           # 至少 3 拍有效
                st80s.append(float(np.mean(vals)))
        st80s = np.array(st80s)
        raw_st80[g] = st80s
        results[g] = {
            "n": len(st80s), "n_bad": int(n_bad),
            "mean_uv": float(np.mean(st80s) * 1000),
            "median_uv": float(np.median(st80s) * 1000),
            "std_uv": float(np.std(st80s) * 1000),
            "q25_uv": float(np.percentile(st80s, 25) * 1000),
            "q75_uv": float(np.percentile(st80s, 75) * 1000),
            "lead": lead,
        }
        print(f"  [{g}] n={len(st80s)} ST80 中位 {np.median(st80s)*1000:.0f}µV "
              f"(IQR {np.percentile(st80s,25)*1000:.0f}~{np.percentile(st80s,75)*1000:.0f}) "
              f"({time.time()-t0:.0f}s)")

    # 判别: IMI/ASMI/STTC vs NORM (AUC + Cohen's d, 复用 raw_st80)
    disc = {}
    for g in ("IMI", "ASMI", "STTC"):
        pos_raw = raw_st80[g]
        neg_raw = raw_st80["NORM"]
        y = np.concatenate([np.ones(len(pos_raw)), np.zeros(len(neg_raw))])
        x = np.concatenate([pos_raw, neg_raw])
        if len(set(y)) == 2 and len(np.unique(x)) > 1:
            auc = roc_auc_score(y, x)
        else:
            auc = float('nan')
        d_ = (np.mean(pos_raw) - np.mean(neg_raw)) / np.sqrt(
            (np.std(pos_raw)**2 + np.std(neg_raw)**2) / 2)
        disc[g] = {"auc": float(auc), "cohen_d": float(d_),
                   "n_pos": len(pos_raw), "n_neg": len(neg_raw),
                   "pos_mean_uv": float(np.mean(pos_raw) * 1000),
                   "neg_mean_uv": float(np.mean(neg_raw) * 1000)}
        print(f"  [{g} vs NORM] AUC={auc:.3f} Cohen's d={d_:+.2f} "
              f"(pos {np.mean(pos_raw)*1000:+.0f}µV vs neg {np.mean(neg_raw)*1000:+.0f}µV)")
    return {"groups": results, "discrimination": disc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ludb-only", action="store_true")
    ap.add_argument("--ptbxl-only", action="store_true")
    ap.add_argument("--n-max", type=int, default=0)
    args = ap.parse_args()

    results = {"config": vars(args),
               "method": {"j_offset_ms": J_OFFSET_MS, "st_window_ms": ST_WIN,
                          "baseline_ms": BASE_WIN}}
    if not args.ptbxl_only:
        results["ludb_jpoint"] = ludb_jpoint_validation()
    if not args.ludb_only:
        results["ptbxl"] = ptbxl_st80_analysis(args.n_max)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 已保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
