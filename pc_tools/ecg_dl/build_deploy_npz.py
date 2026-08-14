#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_deploy_npz.py — 部署链 (D3) 训练数据重建 (阶段 1.5, TUNING_HISTORY 十三章)
=================================================================================
用 eval_deploy_match.py 中已验证的固件部署链 (resample_poly→500Hz → 均值去DC
→ 双级10抽头梳状 → 因果 HP/LP 双二阶 (240点预热) → 2:1 抽取 → 固件 z-score)
重建全部训练数据, 供部署链重训 (exp4/5/6 配置不变)。

输出 (不覆盖任何原文件):
  data/processed/mit_bih_processed_deploy.npz   (raw + 6×增强, 与原管线一致)
  data/processed/incart_processed_deploy.npz    (raw only)
  data/processed/ptb_processed_deploy.npz       (raw only)
  models/deploy_match/ptb_all_peaks.npy         (XQRS 峰值缓存)
  models/deploy_match/build_npz_manifest.json

验证 (强制): 与原 npz 的逐记录拍数断言 (MIT: 原=6×新raw; INCART/PTB: 1:1) + 总数断言。
"""
import sys
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PROCESSED_DIR, TARGET_FS, BEAT_WINDOW_SAMPLES, AAMI_CLASSES, MIT_BIH_RECORDS,
)
from data.preprocess import (
    load_mit_bih_record,
    extract_beats as mit_extract_beats,
    resample_ecg,
    augment_data,
)
from data.preprocess_incart import (
    load_incart_record,
    extract_beats as incart_extract_beats,
)
from data.preprocess_ptb import (
    PTB_DIR,
    load_records as ptb_load_records,
    load_controls as ptb_load_controls,
)
# 复用 harness 已验证链函数 (import 即完成 INCART_DIR WSL 补丁, main() 有守护)
from eval_deploy_match import (
    deployment_chain,
    corrected_deployment_chain,
    align_stream_lengths,
    extract_beats_deploy,
    baseline_chain_ptb,
)

CACHE_DIR = Path(__file__).resolve().parent / "models" / "deploy_match"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# P0-2: 链与输出后缀 (默认 D3 部署链; --causal 切修正后因果链 exp7 用)
CHAIN_FUNC = deployment_chain
NPZ_SUFFIX = "_deploy"
CHAIN_LABEL = "部署链 (D3)"


def set_causal_mode():
    global CHAIN_FUNC, NPZ_SUFFIX, CHAIN_LABEL
    CHAIN_FUNC = corrected_deployment_chain
    NPZ_SUFFIX = "_deploy_causal"
    CHAIN_LABEL = "修正后因果链 (D3 + 因果 HP 0.5Hz@250Hz)"


def _save_npz_and_npy(out, beats, labels, rec_ids):
    """保存压缩 npz + mmap 友好独立 .npy (dataset._load_arrays 优先读 .npy, 降 RSS).

    numpy≥2.0 的 npz 不支持 mmap (TUNING_HISTORY 十三章), 训练期若走 npz 全量加载
    RSS ~900MB; 独立 .npy 可 mmap 只按页加载。
    """
    np.savez_compressed(out, beats=beats, labels=labels, record_ids=rec_ids)
    np.save(out.parent / f"{out.stem}_beats.npy", beats)
    np.save(out.parent / f"{out.stem}_labels.npy", labels)
    np.save(out.parent / f"{out.stem}_record_ids.npy", rec_ids)


def _aami_r_idx(ann_idx, ann_sym, fs):
    """AAMI 过滤后的 R 峰 250Hz 索引 (与 baseline 提取同一公式)."""
    aami_mask = np.array([s in AAMI_CLASSES for s in ann_sym])
    return (ann_idx[aami_mask] * (TARGET_FS / fs)).astype(int)


def _pair_or_warn(rec_tag, beats_d, labels_b):
    """双链拍数配对: 不等则告警并截断到 min (harness 已证明测试记录全等)."""
    n_d, n_b = len(beats_d), len(labels_b)
    if n_d != n_b:
        print(f"WARNING: {rec_tag} deploy={n_d} vs baseline={n_b}, truncate to min")
        n_use = min(n_d, n_b)
        return beats_d[:n_use], labels_b[:n_use], n_use
    return beats_d, labels_b, n_b


# ============================================================
# MIT-BIH (48 记录, raw + 6× 增强)
# ============================================================
def build_mit():
    print("=" * 60)
    print("BUILD: MIT-BIH deploy npz (48 records, 6x augment)")
    print("=" * 60)
    all_beats, all_labels, all_rec_ids = [], [], []
    raw_counts, total_counts = {}, {}

    for rid in MIT_BIH_RECORDS:
        rec_name = str(rid)
        t0 = time.time()
        try:
            signal, ann_idx, ann_sym, fs = load_mit_bih_record(rec_name)
        except Exception as e:
            print(f"  MIT {rec_name}: SKIP (load failed: {e})")
            continue

        # baseline 标签与计数 (波形弃用)
        _, labels_b = mit_extract_beats(
            signal, ann_idx, ann_sym,
            orig_fs=fs, target_fs=TARGET_FS, dual_lead=False)

        r_idx_250 = _aami_r_idx(ann_idx, ann_sym, fs)
        deploy_250 = CHAIN_FUNC(signal[:, 0].astype(np.float64), fs)
        base_250 = resample_ecg(signal[:, :1], fs, TARGET_FS).flatten()
        deploy_250 = align_stream_lengths(base_250, deploy_250)

        beats_d = extract_beats_deploy(deploy_250, r_idx_250, "mit")
        beats_d, labels_b, n_use = _pair_or_warn(f"MIT {rec_name}", beats_d, labels_b)

        beats_aug, labels_aug = augment_data(beats_d, labels_b)
        assert len(beats_aug) == n_use * 6, f"augment: {len(beats_aug)} != {n_use}*6"

        all_beats.append(beats_aug)
        all_labels.append(labels_aug)
        all_rec_ids.append(np.full(len(beats_aug), rid, dtype=np.int32))
        raw_counts[rid] = n_use
        total_counts[rid] = len(beats_aug)
        print(f"  MIT {rec_name}: raw={n_use} aug={len(beats_aug)} [{time.time()-t0:.1f}s]")

    beats = np.concatenate(all_beats).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int32)
    rec_ids = np.concatenate(all_rec_ids).astype(np.int32)
    out = PROCESSED_DIR / f"mit_bih_processed{NPZ_SUFFIX}.npz"
    _save_npz_and_npy(out, beats, labels, rec_ids)
    print(f"  => {out.name}: {len(beats)} beats "
          f"(N={int((labels==0).sum())}, A={int((labels==1).sum())}), "
          f"{out.stat().st_size/1024/1024:.1f} MB")
    return beats, labels, rec_ids, raw_counts


# ============================================================
# INCART (75 记录, 无增强)
# ============================================================
def build_incart():
    print("=" * 60)
    print("BUILD: INCART deploy npz (I01-I75, no augment)")
    print("=" * 60)
    all_beats, all_labels, all_rec_ids = [], [], []
    counts = {}

    for rid in range(1, 76):
        rec_name = f"I{rid:02d}"
        t0 = time.time()
        try:
            sig, ann_idx, ann_sym, fs = load_incart_record(rec_name)
        except Exception as e:
            print(f"  INCART {rec_name}: SKIP (load failed: {e})")
            continue

        _, labels_b, _ = incart_extract_beats(sig, ann_idx, ann_sym, fs, TARGET_FS)
        if len(labels_b) == 0:
            print(f"  INCART {rec_name}: SKIP (0 baseline beats)")
            continue

        r_idx_250 = _aami_r_idx(ann_idx, ann_sym, fs)
        deploy_250 = CHAIN_FUNC(sig.astype(np.float64), fs)
        base_250 = resample_ecg(sig, fs, TARGET_FS)
        deploy_250 = align_stream_lengths(base_250, deploy_250)

        beats_d = extract_beats_deploy(deploy_250, r_idx_250, "incart")
        beats_d, labels_b, n_use = _pair_or_warn(f"INCART {rec_name}", beats_d, labels_b)

        all_beats.append(beats_d)
        all_labels.append(labels_b)
        all_rec_ids.append(np.full(n_use, rid, dtype=np.int32))
        counts[rid] = n_use
        print(f"  INCART {rec_name}: {n_use} beats [{time.time()-t0:.1f}s]")

    beats = np.concatenate(all_beats).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int32)
    rec_ids = np.concatenate(all_rec_ids).astype(np.int32)
    out = PROCESSED_DIR / f"incart_processed{NPZ_SUFFIX}.npz"
    _save_npz_and_npy(out, beats, labels, rec_ids)
    print(f"  => {out.name}: {len(beats)} beats "
          f"(N={int((labels==0).sum())}, A={int((labels==1).sum())}), "
          f"{out.stat().st_size/1024/1024:.1f} MB")
    return beats, labels, rec_ids, counts


# ============================================================
# PTB (549 记录, 无增强; XQRS 峰值经 baseline 链检测并复用)
# ============================================================
def build_ptb():
    print("=" * 60)
    print("BUILD: PTB deploy npz (549 records, no augment)")
    print("=" * 60)
    import wfdb as _wfdb

    records_list = ptb_load_records()
    controls = ptb_load_controls()
    all_beats, all_labels, all_rec_ids = [], [], []
    counts, failed = {}, []
    peaks_map = {}

    for i, rec_name in enumerate(records_list):
        rid = 400000 + i
        t0 = time.time()
        try:
            rec = _wfdb.rdrecord(str(PTB_DIR / rec_name))
        except Exception as e:
            print(f"  PTB {rec_name}: SKIP (load failed: {e})")
            failed.append(rid)
            continue

        fs = rec.fs
        lead = rec.p_signal[:, 1].astype(np.float64)
        label = 0 if rec_name in controls else 1

        # baseline 链: resample→filtfilt→XQRS→严格窗口+std守卫 → kept_peaks
        beats_b, kept_peaks, _ = baseline_chain_ptb(lead, fs)
        if len(kept_peaks) == 0:
            print(f"  PTB {rec_name}: SKIP (0 valid beats)")
            failed.append(rid)
            continue
        peaks_map[rid] = np.asarray(kept_peaks, dtype=np.int32)

        deploy_250 = CHAIN_FUNC(lead, fs)
        base_250 = resample_ecg(lead, fs, TARGET_FS)
        deploy_250 = align_stream_lengths(base_250, deploy_250)

        beats_d = extract_beats_deploy(deploy_250, np.asarray(kept_peaks), "ptb")
        n_d, n_b = len(beats_d), len(beats_b)
        if n_d != n_b:
            print(f"  PTB {rec_name}: WARNING deploy={n_d} vs baseline={n_b}, truncate")
        n_use = min(n_d, n_b)
        if n_use == 0:
            print(f"  PTB {rec_name}: SKIP (0 deploy beats)")
            failed.append(rid)
            continue

        all_beats.append(beats_d[:n_use])
        all_labels.append(np.full(n_use, label, dtype=np.int32))
        all_rec_ids.append(np.full(n_use, rid, dtype=np.int32))
        counts[rid] = n_use
        print(f"  PTB [{i+1}/{len(records_list)}] {rec_name}: {n_use} beats "
              f"(label={label}) [{time.time()-t0:.1f}s]")

    np.save(CACHE_DIR / "ptb_all_peaks.npy", peaks_map, allow_pickle=True)

    beats = np.concatenate(all_beats).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int32)
    rec_ids = np.concatenate(all_rec_ids).astype(np.int32)
    out = PROCESSED_DIR / f"ptb_processed{NPZ_SUFFIX}.npz"
    _save_npz_and_npy(out, beats, labels, rec_ids)
    print(f"  => {out.name}: {len(beats)} beats "
          f"(N={int((labels==0).sum())}, A={int((labels==1).sum())}), "
          f"{len(counts)} records, {len(failed)} failed, "
          f"{out.stat().st_size/1024/1024:.1f} MB")
    return beats, labels, rec_ids, counts, failed


# ============================================================
# 验证: 与原 npz 逐记录 + 总数断言
# ============================================================
def verify(mit, incart, ptb):
    print("=" * 60)
    if NPZ_SUFFIX == "_deploy_causal":
        print("VERIFICATION (vs *_deploy.npz, 1:1 beat counts)")
    else:
        print("VERIFICATION (vs original npz)")
    print("=" * 60)
    errors = []

    _, _, _, mit_raw = mit
    _, _, _, inc_counts = incart
    _, _, _, ptb_counts, ptb_failed = ptb

    if NPZ_SUFFIX == "_deploy_causal":
        # 因果 HP 0.5Hz 不改变 R 峰位置/拍数/边缘跳过决策 → 与 D3 部署链逐记录 1:1
        ref_mit = np.load(PROCESSED_DIR / "mit_bih_processed_deploy.npz")
        ref_inc = np.load(PROCESSED_DIR / "incart_processed_deploy.npz")
        ref_ptb = np.load(PROCESSED_DIR / "ptb_processed_deploy.npz")

        for rid in MIT_BIH_RECORDS:
            n_ref = int((ref_mit["record_ids"] == rid).sum())
            n_new = mit_raw.get(rid, 0) * 6
            if n_ref != n_new:
                errors.append(f"MIT {rid}: deploy={n_ref} != causal={n_new}")
        if len(ref_mit["beats"]) != len(mit[0]):
            errors.append(f"MIT total: deploy={len(ref_mit['beats'])} != causal={len(mit[0])}")
        print(f"  MIT: deploy={len(ref_mit['beats'])} causal={len(mit[0])}")

        for rid in range(1, 76):
            n_ref = int((ref_inc["record_ids"] == rid).sum())
            if n_ref != inc_counts.get(rid, 0):
                errors.append(f"INCART I{rid:02d}: deploy={n_ref} != causal={inc_counts.get(rid, 0)}")
        if len(ref_inc["beats"]) != len(incart[0]):
            errors.append(f"INCART total: deploy={len(ref_inc['beats'])} != causal={len(incart[0])}")
        print(f"  INCART: deploy={len(ref_inc['beats'])} causal={len(incart[0])}")

        ref_recs = set(np.unique(ref_ptb["record_ids"]).tolist())
        n_diff = 0
        for rid in sorted(ref_recs & set(ptb_counts)):
            if int((ref_ptb["record_ids"] == rid).sum()) != ptb_counts[rid]:
                n_diff += 1
                errors.append(f"PTB rid={rid}: deploy count mismatch")
        if len(ref_ptb["beats"]) != len(ptb[0]):
            errors.append(f"PTB total: deploy={len(ref_ptb['beats'])} != causal={len(ptb[0])}")
        print(f"  PTB: deploy={len(ref_ptb['beats'])} causal={len(ptb[0])}, diffs={n_diff}")

        if errors:
            print(f"\n  VERIFICATION FAILED: {len(errors)} errors")
            for e in errors[:20]:
                print(f"    {e}")
            sys.exit(1)
        print("\n  VERIFICATION PASSED: 因果链拍数与 D3 部署链逐记录 1:1")
        return

    # ---- 原 D3 部署链验证 (vs original npz) ----
    # MIT: 原 npz 每记录拍数 == 6 × 新 raw
    orig = np.load(PROCESSED_DIR / "mit_bih_processed.npz")
    for rid in MIT_BIH_RECORDS:
        n_orig = int((orig["record_ids"] == rid).sum())
        n_expect = mit_raw.get(rid, 0) * 6
        if n_orig != n_expect:
            errors.append(f"MIT {rid}: orig={n_orig} != 6*raw={n_expect}")
    print(f"  MIT per-record: {len(MIT_BIH_RECORDS)} records checked, "
          f"{len([e for e in errors if e.startswith('MIT')])} mismatches")
    if len(orig["beats"]) != len(mit[0]):
        errors.append(f"MIT total: orig={len(orig['beats'])} != new={len(mit[0])}")
    print(f"  MIT total: orig={len(orig['beats'])} new={len(mit[0])}")

    # INCART: 1:1
    _, _, _, inc_counts = incart
    orig_inc = np.load(PROCESSED_DIR / "incart_processed.npz")
    for rid in range(1, 76):
        n_orig = int((orig_inc["record_ids"] == rid).sum())
        n_new = inc_counts.get(rid, 0)
        if n_orig != n_new:
            errors.append(f"INCART I{rid:02d}: orig={n_orig} != new={n_new}")
    print(f"  INCART per-record: 75 records checked, "
          f"{len([e for e in errors if e.startswith('INCART')])} mismatches")
    if len(orig_inc["beats"]) != len(incart[0]):
        errors.append(f"INCART total: orig={len(orig_inc['beats'])} != new={len(incart[0])}")
    print(f"  INCART total: orig={len(orig_inc['beats'])} new={len(incart[0])}")

    # PTB: 共有记录 1:1
    _, _, _, ptb_counts, ptb_failed = ptb
    orig_ptb = np.load(PROCESSED_DIR / "ptb_processed.npz")
    orig_recs = set(np.unique(orig_ptb["record_ids"]).tolist())
    n_diff = 0
    for rid in sorted(orig_recs & set(ptb_counts)):
        n_orig = int((orig_ptb["record_ids"] == rid).sum())
        if n_orig != ptb_counts[rid]:
            n_diff += 1
            errors.append(f"PTB rid={rid}: orig={n_orig} != new={ptb_counts[rid]}")
    print(f"  PTB common records: {len(orig_recs & set(ptb_counts))}, diffs: {n_diff}")
    print(f"  PTB failed: orig={549 - len(orig_recs)} new={len(ptb_failed)}")
    if len(orig_ptb["beats"]) != len(ptb[0]):
        errors.append(f"PTB total: orig={len(orig_ptb['beats'])} != new={len(ptb[0])}")
    print(f"  PTB total: orig={len(orig_ptb['beats'])} new={len(ptb[0])}")

    if errors:
        print(f"\n  VERIFICATION FAILED: {len(errors)} errors")
        for e in errors[:20]:
            print(f"    {e}")
        sys.exit(1)
    print("\n  VERIFICATION PASSED: all per-record and total assertions match")


# ============================================================
# Manifest
# ============================================================
def write_manifest(mit, incart, ptb, wall):
    mb, ml, _, mit_raw = mit
    ib, il, _, inc_counts = incart
    pb, pl, _, ptb_counts, ptb_failed = ptb
    manifest = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": f"{CHAIN_LABEL} 训练数据 (P0-2 exp7 用 _deploy_causal / 历史 _deploy)",
        "chain": CHAIN_LABEL,
        "npz_files": {
            "mit_bih": f"data/processed/mit_bih_processed{NPZ_SUFFIX}.npz",
            "incart": f"data/processed/incart_processed{NPZ_SUFFIX}.npz",
            "ptb": f"data/processed/ptb_processed{NPZ_SUFFIX}.npz",
        },
        "totals": {
            "mit_bih": {"records": len(mit_raw),
                        "raw_beats": int(sum(mit_raw.values())),
                        "augmented_beats": int(len(mb)),
                        "normal": int((ml == 0).sum()), "abnormal": int((ml == 1).sum())},
            "incart": {"records": len(inc_counts), "beats": int(len(ib)),
                       "normal": int((il == 0).sum()), "abnormal": int((il == 1).sum())},
            "ptb": {"records": len(ptb_counts), "beats": int(len(pb)),
                    "normal": int((pl == 0).sum()), "abnormal": int((pl == 1).sum()),
                    "failed_records": len(ptb_failed)},
        },
        "assumptions": [
            "波形链 = 固件部署链 (resample_poly→500→均值DC→双级10抽头梳状→因果HP/LP→2:1抽取→固件z-score)",
            "R 峰位置/标签/患者划分/增强倍数与原管线完全一致; PTB 峰值在 baseline 链检测并复用",
            "增强按当前 config (noise_std=0.015) 重新随机, 与原 npz 增强副本不逐拍相同 (设计上即随机)",
            "INCART/PTB 无增强; MIT 6× = 原始 + 1 噪声 + 2 缩放 + 2 漂移",
        ],
        "failed_ptb_records": sorted(int(r) for r in ptb_failed),
        "wall_time_seconds": round(wall, 1),
    }
    out = CACHE_DIR / "build_npz_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Manifest: {out}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="部署链训练数据重建")
    parser.add_argument("--causal", action="store_true",
                        help="P0-2: 用修正后因果链 (D3 + 因果 HP 0.5Hz@250Hz), "
                             "输出 *_deploy_causal.npz (exp7)")
    args = parser.parse_args()
    if args.causal:
        set_causal_mode()
        print(f"[build_deploy_npz] 链切换: {CHAIN_LABEL}")
    t0 = time.time()
    mit = build_mit()
    incart = build_incart()
    ptb = build_ptb()
    verify(mit, incart, ptb)
    wall = time.time() - t0
    write_manifest(mit, incart, ptb, wall)
    print(f"\nDONE: wall time = {wall:.0f}s ({wall/60:.1f} min)")


if __name__ == "__main__":
    main()
