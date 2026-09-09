#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p3c_map_check.py — P3c 前置: 数组行 ↔ 重建链映射验证
=====================================================
验证 *_processed_deploy_causal_*.npy 的行可由以下公式无歧义映射回原始记录:
  MIT:   每记录块 = raw(n) + 5 增强副本 (augment_data 顺序), 行块内顺序
         [raw_0..raw_{n-1}, noisy_0.., scale1_0.., scale2_0.., drift1_0.., drift2_0..];
         raw 行与重建链 (corrected_deployment_chain + extract_beats_deploy)
         逐拍一致, 标签顺序一致。
  INCART: 每记录块 = raw only, 行与重建链逐拍一致, 标签一致。
对账通过 → P3c 可把任一 (记录, 块内行号) 映射回 AAMI 标注序列中的 R 峰位置,
从而在原始信号上切真实 10s 上下文窗。
运行: WSL, export ECG_PROCESSED_DIR=/home/devcontainers/ecg_data
"""
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from config import TARGET_FS, AAMI_CLASSES
from data.preprocess import (
    load_mit_bih_record, extract_beats as mit_extract_beats, resample_ecg,
)
from data.preprocess_incart import (
    load_incart_record, extract_beats as incart_extract_beats,
)
from eval_deploy_match import (
    corrected_deployment_chain, align_stream_lengths, extract_beats_deploy,
)
from data.split_guard import load_arrays


def _aami_r_idx(ann_idx, ann_sym, fs):
    ann_idx = np.asarray(ann_idx)
    m = np.array([s in AAMI_CLASSES for s in ann_sym])
    return (ann_idx[m] * (TARGET_FS / fs)).astype(int)


def rebuild_mit(rid):
    signal, ann_idx, ann_sym, fs = load_mit_bih_record(str(rid))
    labels_b = mit_extract_beats(
        signal, ann_idx, ann_sym, orig_fs=fs, target_fs=TARGET_FS,
        dual_lead=False)[1]
    r250 = _aami_r_idx(ann_idx, ann_sym, fs)
    dep = corrected_deployment_chain(signal[:, 0].astype(np.float64), fs)
    base = resample_ecg(signal[:, :1], fs, TARGET_FS).flatten()
    dep = align_stream_lengths(base, dep)
    beats_d = extract_beats_deploy(dep, r250, "mit")
    n = min(len(beats_d), len(labels_b))
    return beats_d[:n], labels_b[:n]


def rebuild_incart(rid):
    rec_name = f"I{rid:02d}"
    sig, ann_idx, ann_sym, fs = load_incart_record(rec_name)
    labels_b = incart_extract_beats(sig, ann_idx, ann_sym, fs, TARGET_FS)[1]
    r250 = _aami_r_idx(ann_idx, ann_sym, fs)
    dep = corrected_deployment_chain(sig.astype(np.float64), fs)
    base = resample_ecg(sig, fs, TARGET_FS)
    dep = align_stream_lengths(base, dep)
    beats_d = extract_beats_deploy(dep, r250, "incart")
    n = min(len(beats_d), len(labels_b))
    return beats_d[:n], labels_b[:n]


def check(tag, rids, rebuild):
    b, l, r = load_arrays(tag)
    b, l, r = np.asarray(b), np.asarray(l), np.asarray(r)
    ok = True
    for rid in rids:
        rows = np.where(r == rid)[0]
        start = rows[0]
        assert np.array_equal(rows, np.arange(start, start + len(rows))), \
            f"{tag} {rid}: 行不连续"
        blk = b[start:start + len(rows)]
        lbl = l[start:start + len(rows)]
        beats_d, labels_b = rebuild(rid)
        n_raw = len(beats_d)
        if tag == "mit_bih":
            if len(rows) % 6 != 0:
                print(f"  [FAIL] MIT {rid}: 块长 {len(rows)} 非 6 倍数"); ok = False; continue
            n_blk_raw = len(rows) // 6
            if n_blk_raw != n_raw:
                print(f"  [FAIL] MIT {rid}: raw 数 {n_blk_raw} != 重建 {n_raw}"); ok = False; continue
            d = float(np.abs(blk[:n_raw] - beats_d).max())
            lab_ok = bool((lbl[:n_raw] == labels_b).all())
            aug_lab_ok = all(
                bool((lbl[v * n_raw:(v + 1) * n_raw] == labels_b).all())
                for v in range(6))
            aug_diff = float(np.abs(blk[n_raw:2 * n_raw] - blk[:n_raw]).mean())
            verdict = "OK" if (d < 1e-6 and lab_ok and aug_lab_ok and aug_diff > 1e-4) else "FAIL"
            if verdict == "FAIL":
                ok = False
            print(f"  [{verdict}] MIT {rid}: raw={n_raw} max|Δ|={d:.2e} "
                  f"labels={lab_ok} aug_labels={aug_lab_ok} "
                  f"aug_mean_abs_diff={aug_diff:.3e}")
        else:
            if len(rows) != n_raw:
                print(f"  [FAIL] INCART {rid}: 块长 {len(rows)} != 重建 {n_raw}"); ok = False; continue
            d = float(np.abs(blk - beats_d).max())
            lab_ok = bool((lbl == labels_b).all())
            verdict = "OK" if (d < 1e-6 and lab_ok) else "FAIL"
            if verdict == "FAIL":
                ok = False
            print(f"  [{verdict}] INCART I{rid:02d}: n={n_raw} max|Δ|={d:.2e} labels={lab_ok}")
    return ok


def main():
    print("[MAPCHECK] MIT (raw+6x 结构 + 重建一致性)")
    ok1 = check("mit_bih", [201, 231], rebuild_mit)
    print("[MAPCHECK] INCART (raw only + 重建一致性)")
    ok2 = check("incart", [1, 2], rebuild_incart)
    print("[MAPCHECK] RESULT:", "PASS" if (ok1 and ok2) else "FAIL")


if __name__ == "__main__":
    main()
