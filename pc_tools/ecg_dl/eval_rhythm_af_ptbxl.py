#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_rhythm_af_ptbxl.py — 下一步待办#5: AF 短窗验证 (PTB-XL 10s 节律标签)
==========================================================================
任务: 消费设备"一键测房颤"可行性 — 现有 CV+Shannon 熵规则能否在 10s 窗上
     判别 AFIB (PTB-XL 记录级节律标签), 与 30s 窗 AFDB 结果 (AUC 0.935) 对比。

设计 (与 eval_rhythm_af.py 保持规则一致, 仅窗长/最小RR数变化):
  特征: CV (SDNN/mean) + Shannon 熵 (RR 直方图, 16 bins, 0.3-1.5s)
  组合分数: 0.5*(cv/0.2) + 0.5*(ent/4.5)   [与 AFDB 30s 口径相同]
  R 峰检测: 简化 Pan-Tompkins (带通 5-15Hz + 微分平方 + 150ms 滑动积分 + 自适应阈值)
  标签: 正类 = AFIB (validated_by_human); 负类 = 规则窦性 NEG-SR (主口径) / 非AF NEG-ALL (敏感性)

数据: PTB-XL_ECG/ (records500: 12 导联交错 int16, 500Hz, 10s)
输出: models/rhythm_af_ptbxl_eval.json
用法 (WSL): python3 eval_rhythm_af_ptbxl.py [--n-max N] [--lead 1|2|...|12]
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

ROOT = Path(__file__).resolve().parents[2]   # 项目根 (含 PTB-XL_ECG/)
REPO = Path(__file__).resolve().parent       # pc_tools/ecg_dl
PTBXL_DIR = ROOT / "PTB-XL_ECG"
PTBXL_CSV = PTBXL_DIR / "ptbxl_database.csv"
MODELS = REPO / "models"
OUT_JSON = MODELS / "rhythm_af_ptbxl_eval.json"

FS = 500
REC_LEN = 5000          # 10s @500Hz
N_LEADS = 12

# ---- 规则参数 (与 eval_rhythm_af.py 一致) ----
AF_WIN_S = 10.0
AF_MIN_RR = 6           # 10s 窗最少 RR (30s 窗用 20; 10s 窗按 40bpm 下限 ~6)
AF_CV = 0.10
AF_ENTROPY = 1.9

IRREG_RHYTHM = {'AFIB', 'AFLT', 'SVARR', 'BIGU', 'TRIGU', 'PVC', 'PAC',
                'PRC(S)', 'PACE', 'VCLVH', 'LPR', 'WPW'}
REG_RHYTHM = {'SR', 'NORM', 'SBRAD', 'STACH'}
COND_CODES = {'1AVB', '2AVB', '3AVB', 'CRBBB', 'CLBBB', 'IVCD', 'IRBBB'}


def parse_scp(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


def shannon_entropy(rr, bins=16, lo=0.3, hi=1.5):
    h, _ = np.histogram(rr, bins=bins, range=(lo, hi))
    p = h / max(1, h.sum())
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


class RPeakDetector:
    """简化 Pan-Tompkins (PC 侧零相位版; 固件侧实现见 src/heartrate/)."""

    def __init__(self, fs=FS):
        self.fs = fs
        self.b, self.a = scipy_signal.butter(2, [5.0, 15.0],
                                             btype='band', fs=fs)

    def detect(self, x):
        y = scipy_signal.filtfilt(self.b, self.a, x)
        d = np.diff(y, prepend=y[0])
        sq = d * d
        win = int(0.15 * self.fs)
        m = np.convolve(sq, np.ones(win) / win, mode='same')
        thr = 0.30 * np.max(m)          # 自适应阈值 (峰值 30%)
        peaks, _ = scipy_signal.find_peaks(m, height=thr,
                                           distance=int(0.2 * self.fs))
        return peaks


def load_lead(path, lead_idx=1):
    """读 records500 .dat (12 导联交错 int16, 500Hz) 单导联, 返回 mV 数组."""
    raw = np.fromfile(path, dtype='<i2')
    sig = raw.reshape(-1, N_LEADS)[:, lead_idx].astype(np.float64) / 1000.0
    return sig[:REC_LEN]


def score_window(rr):
    """单窗评分: (label, score, n_rr). 与 eval_rhythm_af.py 一致."""
    if len(rr) < AF_MIN_RR:
        return 2, 0.0, len(rr)
    cv = float(np.std(rr) / max(1e-9, np.mean(rr)))
    ent = shannon_entropy(rr)
    score = 0.5 * (cv / 0.2) + 0.5 * (ent / 4.5)
    if cv > AF_CV and ent > AF_ENTROPY:
        return 1, score, len(rr)
    return 0, score, len(rr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=0, help="每条目最大样本数 (调试用, 0=全部)")
    ap.add_argument("--lead", type=int, default=1, help="导联索引 0-11 (0=I, 1=II, ...)")
    ap.add_argument("--ent-bins", type=int, default=16, help="熵直方图 bins")
    ap.add_argument("--neg", choices=["sr", "all"], default="sr",
                    help="负类口径: sr=规则窦性 (主), all=非AF/AFL 全部 (敏感性)")
    ap.add_argument("--tag", default="", help="输出 JSON 后缀 (敏感性变体区分)")
    ap.add_argument("--replay-only", action="store_true")
    args = ap.parse_args()

    out_json = OUT_JSON if not args.tag else OUT_JSON.with_name(
        f"{OUT_JSON.stem}_{args.tag}.json")

    global AF_ENTROPY
    if args.ent_bins != 16:
        pass  # bins 作为参数传入 shannon_entropy

    if not PTBXL_CSV.exists():
        print(f"PTB-XL 数据库不存在: {PTBXL_CSV}")
        sys.exit(1)

    print("=" * 60)
    print(f"PTB-XL 10s 窗 AF 判别验证 (Lead {args.lead + 1}, {args.ent_bins} bins)")
    print("=" * 60)

    rows = list(csv.DictReader(open(PTBXL_CSV)))
    val = [r for r in rows if r['validated_by_human'] == 'True']

    pos = [r for r in val if 'AFIB' in parse_scp(r['scp_codes'])]
    neg_sr = [r for r in val
              if any(c in parse_scp(r['scp_codes']) for c in REG_RHYTHM)
              and not any(c in parse_scp(r['scp_codes'])
                          for c in IRREG_RHYTHM | COND_CODES)]
    neg_all = [r for r in val
               if not any(c in parse_scp(r['scp_codes']) for c in ('AFIB', 'AFLT'))]
    neg = neg_all if args.neg == "all" else neg_sr
    print(f"AFIB: {len(pos)} | 负类({args.neg}): {len(neg)} "
          f"(NEG-SR {len(neg_sr)} / NEG-ALL {len(neg_all)})")

    det = RPeakDetector()
    t0 = time.time()

    def feature(rec):
        fname = PTBXL_DIR / (rec['filename_hr'] + '.dat')
        x = load_lead(fname, args.lead)
        peaks = det.detect(x)
        rr = np.diff(peaks) / FS
        lab, score, n = score_window(rr)
        cv = float(np.std(rr) / max(1e-9, np.mean(rr))) if len(rr) >= AF_MIN_RR else 0.0
        ent = shannon_entropy(rr, bins=args.ent_bins) if len(rr) >= AF_MIN_RR else 0.0
        return {'label': lab, 'score': score, 'n_rr': n,
                'cv': cv, 'ent': ent, 'n_peaks': len(peaks)}

    results = {"replay": None, "ptbxl": None, "config": vars(args),
               "rule_params": {"AF_CV": AF_CV, "AF_ENTROPY": AF_ENTROPY,
                               "AF_MIN_RR": AF_MIN_RR, "win_s": AF_WIN_S}}

    # ---- 冒烟 (合成) ----
    if args.replay_only:
        print("\n[合成回放] (验证管线)")
        rng = np.random.default_rng(0)
        rr_n = np.full(12, 0.85) + rng.normal(0, 0.02, 12)
        rr_a = rng.uniform(0.3, 0.9, 16)
        for name, rr in [("窦性", rr_n), ("AF 随机", rr_a)]:
            lab, sc, n = score_window(rr)
            print(f"  {name}: label={lab} score={sc:.3f} n_rr={n}")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 已保存: {out_json}")
        return

    # ---- 全量特征提取 ----
    n_pos = len(pos) if args.n_max == 0 else min(args.n_max, len(pos))
    n_neg = len(neg) if args.n_max == 0 else min(args.n_max, len(neg))
    feats_pos, feats_neg = [], []
    for i, rec in enumerate(pos[:n_pos]):
        feats_pos.append(feature(rec))
        if (i + 1) % 200 == 0:
            print(f"  正类 {i + 1}/{n_pos} ({time.time() - t0:.0f}s)")
    for i, rec in enumerate(neg[:n_neg]):
        feats_neg.append(feature(rec))
        if (i + 1) % 200 == 0:
            print(f"  负类 {i + 1}/{n_neg} ({time.time() - t0:.0f}s)")

    # ---- 判别力 ----
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    X = np.array([(f['cv'], f['ent'], f['score']) for f in feats_pos + feats_neg])
    ok = np.array([(f['label'] != 2) for f in feats_pos + feats_neg])  # 排除无法判定
    valid = ok & (X[:, 0] > 0)  # cv>0 才有判别意义
    print(f"\n有效窗: {valid.sum()}/{len(y)} (无法判定: {(~ok).sum()})")
    if valid.sum() < 20:
        print("有效窗过少, 中止")
        sys.exit(1)

    auc_cv = roc_auc_score(y[valid], X[valid, 0])
    auc_ent = roc_auc_score(y[valid], X[valid, 1])
    auc_c = roc_auc_score(y[valid], X[valid, 2])

    # 阈值扫描 (Se+Sp 最优, 同 eval_rhythm_af.py)
    best = {"cv": 0.1, "ent": 1.9, "se": 0.0, "sp": 0.0}
    for thr_cv in np.arange(0.06, 0.30, 0.02):
        for thr_ent in np.arange(1.2, 2.8, 0.1):
            pr = ((X[valid, 0] > thr_cv) & (X[valid, 1] > thr_ent)).astype(int)
            yy = y[valid]
            tp = ((pr == 1) & (yy == 1)).sum()
            fp = ((pr == 1) & (yy == 0)).sum()
            fn = ((pr == 0) & (yy == 1)).sum()
            tn = ((pr == 0) & (yy == 0)).sum()
            se = tp / max(1, tp + fn)
            sp = tn / max(1, tn + fp)
            if (se + sp) > (best["se"] + best["sp"]):
                best = {"cv": float(thr_cv), "ent": float(thr_ent),
                        "se": float(se), "sp": float(sp)}

    # 固定阈值 (AF_CV/AF_ENTROPY) 的 Se/Sp
    pred = ((X[valid, 0] > AF_CV) & (X[valid, 1] > AF_ENTROPY)).astype(int)
    yy = y[valid]
    tp = ((pred == 1) & (yy == 1)).sum()
    fp = ((pred == 1) & (yy == 0)).sum()
    fn = ((pred == 0) & (yy == 1)).sum()
    tn = ((pred == 0) & (yy == 0)).sum()
    se_fix = tp / max(1, tp + fn)
    sp_fix = tn / max(1, tn + fp)

    # 峰数诊断 (R 峰检测质量代理)
    n_peaks = np.array([f['n_peaks'] for f in feats_pos + feats_neg])
    n_rr = np.array([f['n_rr'] for f in feats_pos + feats_neg])
    print(f"\n=== 结果 (Lead {args.lead + 1}, {args.ent_bins} bins) ===")
    print(f"  窗数: {len(y)} (AF {n_pos} / 非AF {n_neg}) | 有效: {valid.sum()}")
    print(f"  R 峰/记录: 中位 {np.median(n_peaks):.0f} (IQR {np.percentile(n_peaks,25):.0f}-{np.percentile(n_peaks,75):.0f})")
    print(f"  AUC: CV={auc_cv:.4f} 熵={auc_ent:.4f} 组合={auc_c:.4f}")
    print(f"  固定阈值 (CV>{AF_CV}, 熵>{AF_ENTROPY}): Se={se_fix:.4f} Sp={sp_fix:.4f}")
    print(f"  最优阈值 (CV>{best['cv']}, 熵>{best['ent']}): Se={best['se']:.4f} Sp={best['sp']:.4f}")
    print(f"  [AFDB 30s 窗参照: 组合 AUC 0.935, 最优 Se 0.814/Sp 0.954]")

    results["ptbxl"] = {
        "n_af": int(n_pos), "n_neg": int(n_neg), "n_valid": int(valid.sum()),
        "auc_cv": float(auc_cv), "auc_entropy": float(auc_ent),
        "auc_combo": float(auc_c),
        "fixed_thr_se": se_fix, "fixed_thr_sp": sp_fix,
        "best_thr_cv": best["cv"], "best_thr_ent": best["ent"],
        "best_se": best["se"], "best_sp": best["sp"],
        "median_r_peaks": float(np.median(n_peaks)),
        "q25_r_peaks": float(np.percentile(n_peaks, 25)),
        "q75_r_peaks": float(np.percentile(n_peaks, 75)),
        "n_undetermined": int((~ok).sum()),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 已保存: {out_json}")


if __name__ == "__main__":
    main()
