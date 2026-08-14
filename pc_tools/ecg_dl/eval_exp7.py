#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_exp7.py — P0-2 exp7 修正后因果链口径评估 + 阈值重校准
================================================================================
评估 exp7 (best_resnet_large_exp7.h5) 在"修正后因果链"测试拍上的 MIT/PTB AUC,
并与 exp6-SGD (best_resnet_large_exp6_sgd.h5) 在同口径下对照, 隔离"重训效应"。
阈值扫描 θ∈{0.35, 0.5, 0.6, 0.65, 0.8} 产出 R/P/F1 阈值表。

产出: models/deploy_match/retrain_exp7_eval.json
      (沿用 retrain_exp6_sgd_eval.json 的 results/verdicts 结构, 扩展多阈值)

锚点 (不可篡改, 来自 docs/FINAL_RESULTS.md):
  exp6-SGD 部署链(D3) : MIT 0.9122 / PTB 0.7697
  exp6 患者级清洁 D0  : MIT 0.8942 / PTB 0.8232 (filtfilt 训练链上限)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TARGET_FS, BEAT_WINDOW_SAMPLES, AAMI_CLASSES
from eval_deploy_match import (
    corrected_deployment_chain,
    extract_beats_deploy,
    align_stream_lengths,
    compute_mit_domain_test_records,
    compute_ptb_domain_test_records,
    baseline_chain_mit,
    baseline_chain_incart,
    baseline_chain_ptb,
)
from data.preprocess import load_mit_bih_record, resample_ecg
from data.preprocess_incart import load_incart_record
from data.preprocess_ptb import (
    PTB_DIR, load_records as ptb_load_records, load_controls as ptb_load_controls,
)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
CACHE_DIR = MODEL_DIR / "deploy_match"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.35, 0.5, 0.6, 0.65, 0.8]
OUT_JSON = CACHE_DIR / "retrain_exp7_eval.json"

MIT_CACHE = CACHE_DIR / "mit_deploy_causal_match.npz"
PTB_CACHE = CACHE_DIR / "ptb_deploy_causal_match.npz"

EVAL_MODELS = [
    ("exp7", "best_resnet_large_exp7.h5"),
    ("exp6-SGD", "best_resnet_large_exp6_sgd.h5"),
]

# 锚点 (docs/FINAL_RESULTS.md 表2/表4)
ANCHORS = {
    "mit": {"exp6_sgd_d3": 0.9122, "d0_target": 0.8942},
    "ptb": {"exp6_sgd_d3": 0.7697, "d0_target": 0.8232},
}


def add_channel_dim(x):
    return x.astype(np.float32)[..., np.newaxis]


def compute_metrics(y_true, prob):
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
    result = {"auc": float(roc_auc_score(y_true, prob))}
    for thr in THRESHOLDS:
        pred = (prob >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0)
        result[f"thr_{thr}"] = {
            "recall": float(rec), "precision": float(prec), "f1": float(f1),
        }
    return result


# ============================================================
# 构建修正后因果链测试拍 (MIT+INCART / PTB 域)
# ============================================================

def build_mit_causal_beats():
    mit_test, incart_test, stats = compute_mit_domain_test_records()
    print(f"[MIT 域] {len(mit_test)} MIT + {len(incart_test)} INCART 测试记录 "
          f"({stats['n_test']} 患者)")
    all_beats, all_labels, all_rec_ids = [], [], []

    for rid in mit_test:
        rec_name = str(rid)
        try:
            signal, ann_idx, ann_sym, fs = load_mit_bih_record(rec_name)
        except Exception as e:
            print(f"  MIT {rec_name}: SKIP ({e})")
            continue
        aami_mask = np.array([s in AAMI_CLASSES for s in ann_sym])
        r_idx_250 = (ann_idx[aami_mask] * TARGET_FS / fs).astype(int)
        stream = corrected_deployment_chain(signal[:, 0].astype(np.float64), fs)
        base = resample_ecg(signal[:, :1], fs, TARGET_FS).flatten()
        stream = align_stream_lengths(base, stream)
        beats = extract_beats_deploy(stream, r_idx_250, "mit")
        _, labels_b = baseline_chain_mit(signal, ann_idx, ann_sym, fs)
        n_use = min(len(beats), len(labels_b))
        all_beats.append(beats[:n_use])
        all_labels.append(labels_b[:n_use])
        all_rec_ids.append(np.full(n_use, rid, dtype=np.int32))

    for rid in incart_test:
        rec_name = f"I{rid:02d}"
        try:
            sig, ann_idx, ann_sym, fs = load_incart_record(rec_name)
        except Exception as e:
            print(f"  INCART {rec_name}: SKIP ({e})")
            continue
        aami_mask = np.array([s in AAMI_CLASSES for s in ann_sym])
        r_idx_250 = (ann_idx[aami_mask] * TARGET_FS / fs).astype(int)
        stream = corrected_deployment_chain(sig.astype(np.float64), fs)
        base = resample_ecg(sig, fs, TARGET_FS)
        stream = align_stream_lengths(base, stream)
        beats = extract_beats_deploy(stream, r_idx_250, "incart")
        _, labels_b = baseline_chain_incart(sig, ann_idx, ann_sym, fs)
        n_use = min(len(beats), len(labels_b))
        all_beats.append(beats[:n_use])
        all_labels.append(labels_b[:n_use])
        all_rec_ids.append(np.full(n_use, rid + 100000, dtype=np.int32))

    beats = np.concatenate(all_beats).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int32)
    rec_ids = np.concatenate(all_rec_ids).astype(np.int32)
    np.savez_compressed(MIT_CACHE, beats=beats, labels=labels, record_ids=rec_ids)
    print(f"  => {MIT_CACHE.name}: {len(beats)} 拍 "
          f"(N={int((labels==0).sum())}, A={int((labels==1).sum())})")
    return beats, labels, rec_ids


def build_ptb_causal_beats():
    import wfdb as _wfdb
    ptb_test, stats = compute_ptb_domain_test_records()
    records_list = ptb_load_records()
    controls = ptb_load_controls()
    print(f"[PTB 域] {len(ptb_test)} 测试记录 ({stats['n_test']} 患者)")

    peak_file = CACHE_DIR / "ptb_deploy_match_peaks.npy"
    cached_peaks = np.load(peak_file, allow_pickle=True) if peak_file.exists() else None
    if cached_peaks is not None:
        print(f"  复用 XQRS 峰值缓存 ({len(cached_peaks)} 记录)")

    all_beats, all_labels, all_rec_ids = [], [], []
    failed = []

    for i, rid in enumerate(ptb_test):
        rec_idx = rid - 400000
        if rec_idx < 0 or rec_idx >= len(records_list):
            failed.append(rid)
            continue
        rec_name = records_list[rec_idx]
        try:
            rec = _wfdb.rdrecord(str(PTB_DIR / rec_name))
        except Exception as e:
            failed.append(rid)
            continue
        fs = rec.fs
        lead = rec.p_signal[:, 1].astype(np.float64)
        label = 0 if rec_name in controls else 1

        # R 峰索引: 复用缓存或 baseline 链重检测
        if cached_peaks is not None and i < len(cached_peaks):
            r_idx_250 = cached_peaks[i]
        else:
            sig250 = resample_ecg(lead, fs, TARGET_FS)
            from data.preprocess_ptb import detect_r_peaks, apply_filters
            sig_f = apply_filters(sig250, TARGET_FS)
            r_idx_250 = detect_r_peaks(sig_f)

        stream = corrected_deployment_chain(lead, fs)
        base = resample_ecg(lead, fs, TARGET_FS)
        stream = align_stream_lengths(base, stream)
        beats = extract_beats_deploy(stream, r_idx_250, "ptb")
        if len(beats) == 0:
            failed.append(rid)
            continue
        all_beats.append(beats)
        all_labels.append(np.full(len(beats), label, dtype=np.int32))
        all_rec_ids.append(np.full(len(beats), rid, dtype=np.int32))

    beats = np.concatenate(all_beats).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int32)
    rec_ids = np.concatenate(all_rec_ids).astype(np.int32)
    np.savez_compressed(PTB_CACHE, beats=beats, labels=labels, record_ids=rec_ids)
    print(f"  => {PTB_CACHE.name}: {len(beats)} 拍 "
          f"(N={int((labels==0).sum())}, A={int((labels==1).sum())}), "
          f"{len(failed)} failed")
    return beats, labels, rec_ids


def main():
    import tensorflow as tf

    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="强制重建测试拍缓存")
    args = parser.parse_args()

    print("=" * 70)
    print("eval_exp7.py — exp7 修正后因果链口径评估 + 阈值重校准")
    print("=" * 70)

    # ---- 构建/加载修正后因果链测试拍 ----
    t0 = time.time()
    if args.rebuild or not MIT_CACHE.exists():
        mit_beats, mit_labels, mit_rids = build_mit_causal_beats()
    else:
        d = np.load(MIT_CACHE)
        mit_beats, mit_labels, mit_rids = d["beats"], d["labels"], d["record_ids"]
        print(f"[MIT 域] 加载缓存: {len(mit_beats)} 拍 "
              f"(N={int((mit_labels==0).sum())}, A={int((mit_labels==1).sum())})")

    if args.rebuild or not PTB_CACHE.exists():
        ptb_beats, ptb_labels, ptb_rids = build_ptb_causal_beats()
    else:
        d = np.load(PTB_CACHE)
        ptb_beats, ptb_labels, ptb_rids = d["beats"], d["labels"], d["record_ids"]
        print(f"[PTB 域] 加载缓存: {len(ptb_beats)} 拍 "
              f"(N={int((ptb_labels==0).sum())}, A={int((ptb_labels==1).sum())})")
    print(f"[构建/加载] {time.time()-t0:.1f}s")

    # ---- 评估 ----
    results = {}
    for model_name, model_file in EVAL_MODELS:
        path = MODEL_DIR / model_file
        if not path.exists():
            print(f"\n[模型] {model_name} ({model_file}) MISSING — 跳过")
            continue
        print(f"\n[模型] 加载 {model_file} ...")
        model = tf.keras.models.load_model(str(path), compile=False)
        results[model_name] = {}
        for dom, (beats, labels) in [("mit", (mit_beats, mit_labels)),
                                      ("ptb", (ptb_beats, ptb_labels))]:
            x = add_channel_dim(beats)
            prob = model.predict(x, batch_size=512, verbose=0)[:, 1]
            m = compute_metrics(labels, prob)
            results[model_name][dom] = {
                "auc": m["auc"],
                "n_beats": int(len(labels)),
                "n_abnormal": int((labels == 1).sum()),
            }
            for thr in THRESHOLDS:
                results[model_name][dom][f"thr_{thr}"] = m[f"thr_{thr}"]
            print(f"  {model_name:8s} {dom:3s}: AUC={m['auc']:.4f} "
                  f"(n={len(labels)}, abn={int((labels==1).sum())})")
        del model
        tf.keras.backend.clear_session()

    # ---- 汇总 verdict (对照 exp6-SGD D3 锚点 + D0 上限) ----
    verdicts = {}
    if "exp7" in results and "exp6-SGD" in results:
        for dom in ["mit", "ptb"]:
            e7 = results["exp7"][dom]["auc"]
            e6 = results["exp6-SGD"][dom]["auc"]
            d3 = ANCHORS[dom]["exp6_sgd_d3"]
            d0 = ANCHORS[dom]["d0_target"]
            lines = [
                f"{dom.upper()} exp7 因果链 AUC={e7:.4f} "
                f"vs exp6-SGD 因果链={e6:.4f} (Δ={e7-e6:+.4f})",
                f"{dom.upper()} exp7 因果链 AUC={e7:.4f} "
                f"vs exp6-SGD D3={d3:.4f} (Δ={e7-d3:+.4f})",
                f"{dom.upper()} exp7 因果链 AUC={e7:.4f} "
                f"vs D0 上限={d0:.4f} (残余失配={e7-d0:+.4f})",
            ]
            verdicts[dom] = lines
            print("\n  " + "\n  ".join(lines))

    # ---- 阈值表 ----
    threshold_table = {}
    for model_name in results:
        threshold_table[model_name] = {}
        for dom in ["mit", "ptb"]:
            threshold_table[model_name][dom] = {}
            for thr in THRESHOLDS:
                k = f"thr_{thr}"
                threshold_table[model_name][dom][k] = results[model_name][dom][k]
    print("\n" + "=" * 70)
    print("阈值表 (θ: R / P / F1)")
    print("=" * 70)
    for model_name in results:
        for dom in ["mit", "ptb"]:
            line = f"  {model_name:8s} {dom:3s}: "
            for thr in THRESHOLDS:
                k = f"thr_{thr}"
                r = results[model_name][dom][k]
                line += f"θ={thr}:R{r['recall']:.2f}/P{r['precision']:.2f}/F{r['f1']:.2f}  "
            print(line)

    # ---- 输出 JSON ----
    output = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "purpose": "P0-2 exp7 修正后因果链 (D3 + 因果 HP 0.5Hz@250Hz) 口径评估 + 阈值重校准",
            "chain": "corrected_deployment_chain = D3 部署链 + 因果 HP 0.5Hz (butter(2,0.5,fs=250))",
            "models": [m for m, _ in EVAL_MODELS],
            "thresholds": THRESHOLDS,
            "anchors": ANCHORS,
            "note": "exp7 与 exp6-SGD 在同一修正后因果链测试拍上评估, 隔离重训效应; "
                    "D3 锚点为 exp6-SGD 在旧 D3 链 (无因果 HP) 上的历史值。",
        },
        "results": results,
        "threshold_table": threshold_table,
        "verdicts": verdicts,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[输出] {OUT_JSON}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
