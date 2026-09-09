#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_normfix_arrays.py — §105 Phase A-1: MIT+INCART normfix 数组重建
=======================================================================
逐记录复刻 build_deploy_npz.py --causal 的 MIT/INCART 流程, 唯一替换:
窗口内固件 z-score → normfix_spec_v1 流级归一化 (normfix.normfix_stream +
window_beats_normfix)。其余 (链、对齐、R 峰、AAMI 过滤、窗口/边缘规则、
标签来源、配对截断、增强结构) 逐行同构。

强制验证 (任何一条失败 → 立即中止, 不写任何输出):
  1. 链忠实度: 每条记录用旧公式 (固件窗口 z-score) 在新建链流上重算拍,
     与既有 *_deploy_causal 数组的对应原始拍块逐位相等 (float bit-exact)。
  2. normfix 窗口拍数 == z-score 窗口拍数 (索引决定, 必须逐记录成立)。
  3. 全量: record_ids 序列逐位相等、labels 序列逐位相等、总拍数相等。

输出 (全部新文件, _normfix 后缀, 不覆盖任何既有数组):
  $ECG_PROCESSED_DIR/mit_bih_processed_deploy_causal_normfix{,_beats,_labels,_record_ids}.npy/.npz
  $ECG_PROCESSED_DIR/incart_processed_deploy_causal_normfix{,_beats,_labels,_record_ids}.npy/.npz
  models/deploy_match/build_normfix_arrays_manifest.json
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
# import 即完成 INCART_DIR WSL 补丁
from eval_deploy_match import (
    corrected_deployment_chain,
    align_stream_lengths,
    extract_beats_deploy,
)
from normfix import load_spec, normalize_stream, window_beats_normfix

CACHE_DIR = Path(__file__).resolve().parent / "models" / "deploy_match"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SPEC = None
STEMS = {
    "mit_bih": "mit_bih_processed_deploy_causal",
    "incart": "incart_processed_deploy_causal",
}
OUT_SUFFIX = "_normfix"


def _aami_r_idx(ann_idx, ann_sym, fs):
    aami_mask = np.array([s in AAMI_CLASSES for s in ann_sym])
    return (ann_idx[aami_mask] * (TARGET_FS / fs)).astype(int)


def _pair_truncate(rec_tag, n_d, n_b):
    # 与旧构建 _pair_or_warn 相同路径: 截断到 min。不作为致命错误 —
    # 旧数组若经历过同样截断, 最终逐位比对仍会成立。
    if n_d != n_b:
        print(f"WARNING: {rec_tag} deploy={n_d} vs baseline={n_b}, truncate to min")
    return min(n_d, n_b)


def load_ref(tag):
    """既有 _deploy_causal 数组 (参考基准, 只读)。"""
    stem = STEMS[tag]
    beats = np.load(PROCESSED_DIR / f"{stem}_beats.npy", mmap_mode="r")
    labels = np.load(PROCESSED_DIR / f"{stem}_labels.npy")
    rec_ids = np.load(PROCESSED_DIR / f"{stem}_record_ids.npy")
    return beats, labels, rec_ids


def check_no_overwrite(tag):
    stem = STEMS[tag] + OUT_SUFFIX
    targets = [PROCESSED_DIR / f"{stem}.npz",
               PROCESSED_DIR / f"{stem}_beats.npy",
               PROCESSED_DIR / f"{stem}_labels.npy",
               PROCESSED_DIR / f"{stem}_record_ids.npy"]
    for p in targets:
        if p.exists():
            print(f"ABORT: 输出已存在 (禁止覆盖): {p}")
            sys.exit(1)
    return targets


def build_domain(tag, spec, ref_rids, errors):
    """tag: 'mit_bih' | 'incart'。返回 (beats, labels, rec_ids, per_record)。"""
    is_mit = tag == "mit_bih"
    all_beats, all_labels, all_rec_ids = [], [], []
    per_record = {}

    rids = MIT_BIH_RECORDS if is_mit else list(range(1, 76))
    for rid in rids:
        rec_name = str(rid) if is_mit else f"I{rid:02d}"
        t0 = time.time()
        try:
            if is_mit:
                signal, ann_idx, ann_sym, fs = load_mit_bih_record(rec_name)
            else:
                signal, ann_idx, ann_sym, fs = load_incart_record(rec_name)
        except Exception as e:
            print(f"  {tag} {rec_name}: SKIP (load failed: {e})", flush=True)
            if int((ref_rids == rid).sum()) != 0:
                errors.append(f"{tag} {rec_name}: 加载失败但旧数组含该记录")
            continue

        if is_mit:
            _, labels_b = mit_extract_beats(
                signal, ann_idx, ann_sym,
                orig_fs=fs, target_fs=TARGET_FS, dual_lead=False)
            sig_chain = signal[:, 0].astype(np.float64)
            base_250 = resample_ecg(signal[:, :1], fs, TARGET_FS).flatten()
        else:
            _, labels_b, _ = incart_extract_beats(signal, ann_idx, ann_sym, fs, TARGET_FS)
            if len(labels_b) == 0:
                print(f"  {tag} {rec_name}: SKIP (0 baseline beats)", flush=True)
                if int((ref_rids == rid).sum()) != 0:
                    errors.append(f"{tag} {rec_name}: 0 基线拍但旧数组含该记录")
                continue
            sig_chain = signal.astype(np.float64)
            base_250 = resample_ecg(signal, fs, TARGET_FS)

        r_idx_250 = _aami_r_idx(ann_idx, ann_sym, fs)
        deploy_250 = corrected_deployment_chain(sig_chain, fs)
        deploy_250 = align_stream_lengths(base_250, deploy_250)

        # ---- 1) 链忠实度: 旧公式重算 → 与旧数组原始拍块逐位相等 ----
        beats_z = extract_beats_deploy(deploy_250, r_idx_250,
                                       "mit" if is_mit else "incart")
        n_use = _pair_truncate(f"{tag} {rec_name}", len(beats_z), len(labels_b))
        beats_z = beats_z[:n_use]
        labels_b = labels_b[:n_use]

        # ---- 2) normfix 归一化 + 窗口提取 (唯一变量) ----
        y250 = normalize_stream(deploy_250, spec)
        beats_n = window_beats_normfix(y250, r_idx_250,
                                       "mit" if is_mit else "incart",
                                       BEAT_WINDOW_SAMPLES)
        if len(beats_n) != len(beats_z):
            errors.append(f"{tag} {rec_name}: normfix 拍数 {len(beats_n)} != z-score 拍数 {len(beats_z)}")
            print(f"ABORT-COUNT: {tag} {rec_name}", flush=True)
            sys.exit(1)
        beats_n = beats_n[:n_use]

        # ---- 3) MIT 增强 (与旧构建同结构, 增强随机数按设计不固定) ----
        if is_mit:
            beats_out, labels_out = augment_data(beats_n, labels_b)
            assert len(beats_out) == n_use * 6, f"augment: {len(beats_out)} != {n_use}*6"
        else:
            beats_out, labels_out = beats_n, labels_b

        all_beats.append(beats_out)
        all_labels.append(labels_out.astype(np.int32))
        all_rec_ids.append(np.full(len(beats_out), rid, dtype=np.int32))
        per_record[int(rid)] = {"raw_beats": int(n_use),
                                "out_beats": int(len(beats_out)),
                                "seconds": round(time.time() - t0, 1)}
        print(f"  {tag} {rec_name}: raw={n_use} out={len(beats_out)} "
              f"[{time.time()-t0:.1f}s]", flush=True)
        # 返回链流供调用方做逐位对比 (避免重复计算): 存 (rid, beats_z, n_use)
        per_record[int(rid)]["z_beats"] = beats_z

    beats = np.concatenate(all_beats).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int32)
    rec_ids = np.concatenate(all_rec_ids).astype(np.int32)
    return beats, labels, rec_ids, per_record


def verify_bitexact(tag, per_record, ref_beats, ref_rids, errors):
    """逐记录: 旧公式重算拍 == 旧数组原始拍块 (MIT 取每记录块的前 n_use 行)。"""
    n_checked = 0
    for rid, info in sorted(per_record.items()):
        mask = ref_rids == rid
        block = np.asarray(ref_beats[mask])
        z = info["z_beats"]
        if is_mit_domain(tag):
            ref_raw = block[: info["raw_beats"]]
        else:
            ref_raw = block
        if ref_raw.shape != z.shape or not np.array_equal(ref_raw, z):
            errors.append(f"{tag} rid={rid}: 链重算拍与旧数组不一致 (链漂移)")
        else:
            n_checked += 1
    print(f"  {tag}: 链忠实度逐位比对 {n_checked}/{len(per_record)} 记录通过", flush=True)


def is_mit_domain(tag):
    return tag == "mit_bih"


def save_outputs(tag, beats, labels, rec_ids):
    stem = STEMS[tag] + OUT_SUFFIX
    npz_path = PROCESSED_DIR / f"{stem}.npz"
    np.savez_compressed(npz_path, beats=beats, labels=labels, record_ids=rec_ids)
    np.save(PROCESSED_DIR / f"{stem}_beats.npy", beats)
    np.save(PROCESSED_DIR / f"{stem}_labels.npy", labels)
    np.save(PROCESSED_DIR / f"{stem}_record_ids.npy", rec_ids)
    print(f"  => {stem}: {len(beats)} beats "
          f"(N={int((labels == 0).sum())}, A={int((labels == 1).sum())}), "
          f"{npz_path.stat().st_size/1024/1024:.1f} MB npz", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="normfix 数组重建 (A/B 规格)")
    ap.add_argument("--spec", default=None,
                    help="规格 JSON 路径; 默认 A (normfix_spec_v1.json); "
                         "B 用 models/deploy_match/normfix_spec_v1_b.json")
    args = ap.parse_args()

    global SPEC, OUT_SUFFIX
    SPEC = load_spec(args.spec) if args.spec else load_spec()
    if SPEC["option"] == "B":
        OUT_SUFFIX = "_normfix_b"
    else:
        OUT_SUFFIX = "_normfix"

    t0 = time.time()
    print(f"[build_normfix_arrays] spec={SPEC['spec_id']} v{SPEC['version']} "
          f"option={SPEC['option']}", flush=True)
    print(f"[build_normfix_arrays] PROCESSED_DIR={PROCESSED_DIR}", flush=True)

    errors = []
    results = {}
    new_arrays = {}
    for tag in ("mit_bih", "incart"):
        check_no_overwrite(tag)

    for tag in ("mit_bih", "incart"):
        print("=" * 60, flush=True)
        print(f"BUILD: {tag} normfix arrays", flush=True)
        print("=" * 60, flush=True)
        ref_beats, ref_labels, ref_rids = load_ref(tag)
        beats, labels, rec_ids, per_record = build_domain(tag, SPEC, ref_rids, errors)
        verify_bitexact(tag, per_record, ref_beats, ref_rids, errors)

        # ---- 3) 全量断言: record_ids / labels 序列逐位, 总数 ----
        if len(beats) != len(ref_beats):
            errors.append(f"{tag}: 总拍数 new={len(beats)} != old={len(ref_beats)}")
        if len(rec_ids) != len(ref_rids) or not np.array_equal(rec_ids, ref_rids):
            errors.append(f"{tag}: record_ids 序列与旧数组不一致")
        if len(labels) != len(ref_labels) or not np.array_equal(labels, ref_labels):
            errors.append(f"{tag}: labels 序列与旧数组不一致")
        print(f"  {tag}: total new={len(beats)} old={len(ref_beats)}", flush=True)

        for info in per_record.values():
            info.pop("z_beats", None)
        new_arrays[tag] = (beats, labels, rec_ids)
        results[tag] = {"records": len(per_record),
                        "total_beats": int(len(beats)),
                        "normal": int((labels == 0).sum()),
                        "abnormal": int((labels == 1).sum()),
                        "per_record": per_record}

    if errors:
        print(f"\nVERIFICATION FAILED: {len(errors)} errors — 不写任何输出", flush=True)
        for e in errors[:40]:
            print(f"  {e}", flush=True)
        sys.exit(1)

    for tag in ("mit_bih", "incart"):
        save_outputs(tag, *new_arrays[tag])

    manifest = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "§105 Phase A-1: MIT+INCART normfix 探针数组 (仅窗口归一化被替换)",
        "spec": {"id": SPEC["spec_id"], "version": SPEC["version"],
                 "option": SPEC["option"], "constants": SPEC["constants"]},
        "chain": "修正后因果链 (D3 + 因果 HP 0.5Hz@250Hz), 与 build_deploy_npz --causal 同",
        "reference_arrays": {tag: STEMS[tag] for tag in STEMS},
        "verification": [
            "链忠实度: 旧公式 (固件窗口 z-score) 在新建链流上重算拍, 与旧 _deploy_causal 数组逐位相等",
            "normfix 拍数 == z-score 拍数 (逐记录)",
            "record_ids / labels 序列与旧数组逐位相等; 总拍数相等",
        ],
        "outputs": {tag: STEMS[tag] + OUT_SUFFIX for tag in STEMS},
        "totals": {tag: {k: v for k, v in results[tag].items() if k != "per_record"}
                   for tag in results},
        "wall_time_seconds": round(time.time() - t0, 1),
    }
    out = CACHE_DIR / f"build_normfix_arrays{OUT_SUFFIX}_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Manifest: {out}", flush=True)
    print(f"\nDONE: wall time = {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
