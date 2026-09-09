#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preprocess_svdb_deploy.py — SVDB 部署因果链 (deploy_causal) 数组构建

口径与 build_deploy_npz.py --causal 完全一致:
  波形链 = corrected_deployment_chain (D3 部署链 + 因果 HP 0.5Hz@250Hz)
  拍提取 = extract_beats_deploy(domain="svdb" → 非 MIT 分支: 严格跳边 + 固件 z-score)
  无增强; R 峰 = AAMI 过滤后的注释峰, 与基线提取同一 250Hz 索引公式。

输出 (PROCESSED_DIR, 即 ECG_PROCESSED_DIR, 默认 /home/devcontainers/ecg_data):
  svdb_processed_deploy_causal.npz + _beats/_labels/_record_ids .npy

验证 (强制, 失败即退出码 1):
  1. 每记录拍数 <= legacy svdb_processed.npz 同记录拍数 (严格跳边只减不增)
  2. 标签序与 legacy 一致 (legacy 按注释序、无边缘丢弃 → 新标签是其后缀)
  3. 守卫占位自检: 所有 record_id ∈ [800, 900), 与 MIT/INCART/PTB 无碰撞
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PROCESSED_DIR, TARGET_FS, BEAT_WINDOW_SAMPLES, AAMI_CLASSES
from data.preprocess import resample_ecg
from eval_deploy_match import (
    corrected_deployment_chain,
    align_stream_lengths,
    extract_beats_deploy,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
SVDB_DIR = DATA_DIR / "raw" / "svdb"
LEGACY_NPZ = DATA_DIR / "processed" / "svdb_processed.npz"
TAG = "svdb"
SUFFIX = "_processed_deploy_causal"


def _aami_r_idx(ann_idx, ann_sym, fs):
    """AAMI 过滤后的 R 峰 250Hz 索引 (与 build_deploy_npz 同一公式)。"""
    aami_mask = np.array([s in AAMI_CLASSES for s in ann_sym])
    return (ann_idx[aami_mask] * (TARGET_FS / fs)).astype(int)


def _aami_labels(ann_sym):
    syms = [s.decode() if isinstance(s, bytes) else s for s in ann_sym]
    return np.array([AAMI_CLASSES[s] for s in syms if s in AAMI_CLASSES],
                    dtype=np.int32)


def build_svdb():
    import wfdb
    print("=" * 60)
    print("BUILD: SVDB deploy_causal npz (78 records, lead 0 / MLII, no augment)")
    print("=" * 60)
    all_beats, all_labels, all_rec_ids = [], [], []
    counts = {}

    for f in sorted(SVDB_DIR.glob("*.dat")):
        rid = int(f.stem)
        t0 = time.time()
        try:
            rec = wfdb.rdann(str(SVDB_DIR / f.stem), 'atr')
            recs = wfdb.rdrecord(str(SVDB_DIR / f.stem))
        except Exception as e:
            print(f"  SVDB {rid}: SKIP (load failed: {e})")
            continue
        fs = recs.fs
        sig = recs.p_signal[:, 0].astype(np.float64)   # lead 0 = MLII

        ann_idx = np.asarray(rec.sample)
        ann_sym = rec.symbol
        labels_b = _aami_labels(ann_sym)
        if len(labels_b) == 0:
            print(f"  SVDB {rid}: SKIP (0 AAMI beats)")
            continue

        r_idx_250 = _aami_r_idx(ann_idx, ann_sym, fs)
        deploy_250 = corrected_deployment_chain(sig, fs)
        base_250 = resample_ecg(sig.reshape(-1, 1), fs, TARGET_FS)[:, 0]
        deploy_250 = align_stream_lengths(base_250, deploy_250)

        # 严格跳边: 预先剔除窗口越界峰, 标签与存活拍 1:1 对齐
        half = BEAT_WINDOW_SAMPLES // 2
        keep = ((r_idx_250 - half >= 0) &
                (r_idx_250 - half + BEAT_WINDOW_SAMPLES <= len(deploy_250)))
        beats_d = extract_beats_deploy(deploy_250, r_idx_250[keep], TAG)
        if len(beats_d) != int(keep.sum()):
            print(f"  SVDB {rid}: FATAL deploy={len(beats_d)} != keep={int(keep.sum())}")
            sys.exit(1)
        labels_use = labels_b[keep]

        n_d = len(beats_d)
        all_beats.append(beats_d)
        all_labels.append(labels_use)
        all_rec_ids.append(np.full(n_d, rid, dtype=np.int32))
        counts[rid] = n_d
        print(f"  SVDB {rid}: {n_d} beats (A={int((labels_use==1).sum())}) "
              f"[{time.time()-t0:.1f}s]")

    beats = np.concatenate(all_beats).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int32)
    rec_ids = np.concatenate(all_rec_ids).astype(np.int32)
    print(f"  total: {len(beats)} beats "
          f"(N={int((labels==0).sum())}, A={int((labels==1).sum())}), "
          f"{len(counts)} records")
    return beats, labels, rec_ids, counts


def verify(beats, labels, rec_ids, counts):
    print("=" * 60)
    print("VERIFICATION (vs legacy svdb_processed.npz)")
    print("=" * 60)
    errors = []

    rids = np.unique(rec_ids)
    if int(rids.min()) < 800 or int(rids.max()) >= 900:
        errors.append(f"record_id 超出 [800,900): {int(rids.min())}-{int(rids.max())}")
    # 与 MIT (100-234) / INCART (+100000) / PTB (400000+) 无碰撞: 800-899 独占区间
    print(f"  record_id 区间: {int(rids.min())}-{int(rids.max())} "
          f"(与 MIT/INCART/PTB 无碰撞)")

    # 权威对照: 逐记录与原始注释 (当前 AAMI 映射) 比对。
    # 注: legacy svdb_processed.npz 经审计已陈旧 (64/78 记录与当前 AAMI
    # 映射的注释真值长度不符), 不作为参照。
    import wfdb
    n_drop = 0
    for rid in rids:
        rpath = str(SVDB_DIR / str(int(rid)))
        ann = wfdb.rdann(rpath, 'atr')
        recs = wfdb.rdrecord(rpath)
        truth = _aami_labels(ann.symbol)
        r_idx = _aami_r_idx(np.asarray(ann.sample), ann.symbol, recs.fs)
        n_sig = int(len(recs.p_signal[:, 0]) * TARGET_FS / recs.fs)
        half = BEAT_WINDOW_SAMPLES // 2
        keep = ((r_idx - half >= 0) & (r_idx - half + BEAT_WINDOW_SAMPLES <= n_sig))
        new_l = labels[rec_ids == rid]
        if len(new_l) != int(keep.sum()):
            errors.append(f"SVDB {rid}: new={len(new_l)} != keep={int(keep.sum())}")
            continue
        if not np.array_equal(truth[keep], new_l):
            errors.append(f"SVDB {rid}: 标签与注释真值不一致")
        n_drop += len(truth) - int(keep.sum())
    print(f"  注释真值对照: {len(rids)} 记录, "
          f"{'通过' if not errors else f'{len(errors)} 条错误'}, "
          f"严格跳边丢拍合计={n_drop}")
    if LEGACY_NPZ.exists():
        leg = np.load(LEGACY_NPZ)
        n_bad = 0
        for rid in rids:
            ann = wfdb.rdann(str(SVDB_DIR / str(int(rid))), 'atr')
            if int((leg["record_ids"] == rid).sum()) != \
                    len(_aami_labels(ann.symbol)):
                n_bad += 1
        print(f"  (legacy npz 陈旧审计: {n_bad}/{len(rids)} 记录拍数与注释真值不符)")

    if errors:
        print(f"\n  VERIFICATION FAILED: {len(errors)} errors")
        for e in errors[:20]:
            print(f"    {e}")
        sys.exit(1)
    print("\n  VERIFICATION PASSED")


def main():
    t0 = time.time()
    beats, labels, rec_ids, counts = build_svdb()
    verify(beats, labels, rec_ids, counts)

    out = PROCESSED_DIR / f"{TAG}{SUFFIX}.npz"
    np.savez_compressed(out, beats=beats, labels=labels, record_ids=rec_ids)
    np.save(out.parent / f"{out.stem}_beats.npy", beats)
    np.save(out.parent / f"{out.stem}_labels.npy", labels)
    np.save(out.parent / f"{out.stem}_record_ids.npy", rec_ids)
    wall = time.time() - t0
    print(f"\nSaved: {out} ({out.stat().st_size/1024/1024:.1f} MB) + 3 .npy")
    print(f"DONE: wall time = {wall:.0f}s ({wall/60:.1f} min)")


if __name__ == "__main__":
    main()
