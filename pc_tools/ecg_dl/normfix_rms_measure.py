#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""normfix_rms_measure.py — §105 规格常数测量 (仅训练侧数据)
================================================================================
目的: 为 normfix_spec_v1 固定两个常数 (包络初值 s_init 与除零底 smin),
在修正后因果链 (corrected_deployment_chain) 的 250Hz 输出流上测量逐记录
RMS 分布。铁律合规: 只取 SplitGuard 判定的训练患者记录 (MIT/INCART) 与
真实 AFE 记录; 测试记录零接触; 只读不写任何既有数组。

产物: models/deploy_match/normfix_rms_measure.json
"""
import json
import struct
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TARGET_FS
from data.preprocess import load_mit_bih_record, resample_ecg
from data.preprocess_incart import load_incart_record
from data.split_guard import get_guard
from eval_deploy_match import (
    _comb_filter, _hp_lp_filter, causal_hp_05_fs250,
    corrected_deployment_chain, align_stream_lengths,
)

BASE = Path(__file__).resolve().parent
OUT = BASE / "models" / "deploy_match" / "normfix_rms_measure.json"
DATA_REAL = BASE / "data" / "real"
REPO_ROOT = BASE.parents[1]
# 真实 AFE 源与既有拍文件一一对应:
#   real_normal_beats_exp7c.npy     <- data/real/ecg_real_052.ecgr (preprocess_real_exp7c.py)
#   real_normal_beats_rec_latest.npy <- rec_latest.ecgr @ 仓库根 (retest_ai_rec_latest.py)
# (data/real/ecg_real_140.ecgr 是 21 字节坏桩, 非来源, 不读)
REAL_SOURCES = [
    ("ecg_real_052.ecgr", DATA_REAL / "ecg_real_052.ecgr"),
    ("rec_latest.ecgr", REPO_ROOT / "rec_latest.ecgr"),
]

ROLL_S = 10.0      # 滚动 RMS 窗 (秒)
EDGE_S = 12.5      # 首/尾段 RMS (秒, ≈3τ, τ=4s)


def stream_stats(s250):
    """单条 250Hz 流的 RMS 统计 (全部派生自给定数组, 无随机)。"""
    fs = TARGET_FS
    n = len(s250)
    x2 = s250 * s250
    rms_full = float(np.sqrt(x2.mean()))
    ne = int(EDGE_S * fs)
    rms_head = float(np.sqrt(x2[:ne].mean())) if n >= ne else rms_full
    rms_tail = float(np.sqrt(x2[-ne:].mean())) if n >= ne else rms_full
    rw = int(ROLL_S * fs)
    step = rw // 4
    rms_roll = []
    for st in range(0, n - rw + 1, step):
        rms_roll.append(float(np.sqrt(x2[st:st + rw].mean())))
    return {
        "n_samples": int(n),
        "dur_s": n / fs,
        "rms_full": rms_full,
        "rms_head_12.5s": rms_head,
        "rms_tail_12.5s": rms_tail,
        "rms_roll10s_min": float(min(rms_roll)) if rms_roll else rms_full,
        "rms_roll10s_p10": float(np.percentile(rms_roll, 10)) if rms_roll else rms_full,
        "rms_roll10s_median": float(np.median(rms_roll)) if rms_roll else rms_full,
    }


def mit_chain_stream(rid):
    """与 build_deploy_npz.build_mit 同一链: CHAIN(signal[:,0], fs) + align。"""
    signal, ann_idx, ann_sym, fs = load_mit_bih_record(str(rid))
    deploy_250 = corrected_deployment_chain(signal[:, 0].astype(np.float64), fs)
    base_250 = resample_ecg(signal[:, :1], fs, TARGET_FS).flatten()
    return align_stream_lengths(base_250, deploy_250)


def incart_chain_stream(rid):
    sig, ann_idx, ann_sym, fs = load_incart_record(f"I{rid:02d}")
    deploy_250 = corrected_deployment_chain(sig.astype(np.float64), fs)
    base_250 = resample_ecg(sig, fs, TARGET_FS)
    return align_stream_lengths(base_250, deploy_250)


def real_chain_stream(ecgr_path):
    """与 preprocess_real_exp7c.py 逐行同链 (225.68Hz 有理重采样起)。"""
    raw = ecgr_path.read_bytes()
    n = struct.unpack_from("<I", raw, 18)[0]
    dur = struct.unpack_from("<I", raw, 14)[0]
    x = np.frombuffer(raw, dtype="<i2", count=n, offset=32).astype(np.float64) / 8000.0
    fs_eff = n / dur
    ratio = Fraction(int(round(500.0 / fs_eff * 10000)), 10000).limit_denominator(100000)
    s500 = resample_poly(x, ratio.numerator, ratio.denominator)
    dc = s500 - np.mean(s500)
    combed = _comb_filter(dc)
    filt = _hp_lp_filter(combed)
    dec = filt[0::2]
    return causal_hp_05_fs250(dec), fs_eff


def main():
    t0 = time.time()
    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "normfix_spec_v1 常数测量: 修正后因果链 250Hz 流的逐记录 RMS",
        "data_scope": "MIT/INCART 仅 SplitGuard train 患者记录; 真实 AFE 两记录; "
                      "测试记录零接触; 不写任何既有数组",
        "records": {},
    }

    g_mit = get_guard("mit_bih")
    mit_train = [int(r) for r in g_mit.train_record_ids()]
    g_inc = get_guard("incart")
    inc_train = [int(r) for r in g_inc.train_record_ids()]
    report["train_record_ids"] = {"mit_bih": mit_train, "incart": inc_train}

    for rid in mit_train:
        report["records"][f"mit_{rid}"] = stream_stats(mit_chain_stream(rid))
        print(f"[RMS] mit_{rid}: rms_full={report['records'][f'mit_{rid}']['rms_full']:.4f}",
              flush=True)
    for rid in inc_train[:3]:
        report["records"][f"incart_{rid}"] = stream_stats(incart_chain_stream(rid))
        print(f"[RMS] incart_{rid}: rms_full={report['records'][f'incart_{rid}']['rms_full']:.4f}",
              flush=True)
    for name, path in REAL_SOURCES:
        s, fs_eff = real_chain_stream(path)
        report["records"][f"real_{name}"] = stream_stats(s)
        report["records"][f"real_{name}"]["fs_eff_hz"] = float(fs_eff)
        print(f"[RMS] real_{name}: rms_full={report['records'][f'real_{name}']['rms_full']:.6f}",
              flush=True)

    rms_all = [v["rms_full"] for v in report["records"].values()]
    rms_all = np.array(sorted(rms_all))
    report["aggregate"] = {
        "n_records": int(len(rms_all)),
        "rms_full_min": float(rms_all.min()),
        "rms_full_p10": float(np.percentile(rms_all, 10)),
        "rms_full_median": float(np.median(rms_all)),
        "rms_full_p90": float(np.percentile(rms_all, 90)),
        "rms_full_max": float(rms_all.max()),
        "rms_roll10s_min_overall": float(min(
            v["rms_roll10s_min"] for v in report["records"].values())),
        "rms_head_over_median_max": float(max(
            v["rms_head_12.5s"] / v["rms_full"] for v in report["records"].values())),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[RMS] saved {OUT.name} ({time.time() - t0:.0f}s)", flush=True)
    print("[RMS] aggregate:", json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
