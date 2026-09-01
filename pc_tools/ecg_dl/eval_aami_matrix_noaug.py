#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_aami_matrix_noaug.py — AAMI 矩阵 noaug 口径（部署链，患者级）
================================================================
与 eval_aami_matrix.py 相同模型集合，但测试集只保留 MIT 每条记录的
第一个增强块，还原未增强测试拍（n=51,883，与 FINAL_RESULTS noaug
口径一致），用于主表级 AAMI 对比。

输出：
  models/aami_matrix_deploy_patient_noaug.json
  models/aami_matrix_deploy_patient_noaug.csv
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODELS_DIR
from data.dataset import set_npz_suffix, load_mit_incart_merged
from data.patient_split import (
    build_mit_patient_map,
    build_incart_patient_map,
    patient_level_split,
)
from eval_aami_breakdown import (
    recover_mit_symbols_per_record,
    recover_incart_symbols_per_record,
    align_symbols_to_npz,
)
from eval_aami_matrix import (
    DEFAULT_MODELS,
    THRESHOLDS,
    add_channel_dim,
    evaluate_model,
    write_csv,
)

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def reduce_mit_augmentation(beats, labels, rids):
    """MIT deploy 数据每条记录 6× 增强；只保留第一块原始拍。"""
    keep = []
    n_records = 0
    for rid in np.unique(rids):
        idx = np.flatnonzero(rids == rid)
        if rid < 100000:
            if len(idx) % 6 != 0:
                raise ValueError(
                    f"MIT record {rid}: augmented count not divisible by 6: {len(idx)}"
                )
            keep.extend(idx[: len(idx) // 6].tolist())
        else:
            keep.extend(idx.tolist())
        n_records += 1
    keep = np.asarray(keep, dtype=np.int64)
    print(f"reduce augmentation: {len(beats)} → {len(keep)} beats, "
          f"{n_records} records", flush=True)
    return beats[keep], labels[keep], rids[keep], keep


def main():
    set_npz_suffix("_deploy")
    t0 = time.time()
    print("=" * 78, flush=True)
    print("加载 MIT+INCART deploy 数据并还原 noaug 序列 ...", flush=True)
    data = load_mit_incart_merged()
    full_rids = data["record_ids"]
    beats, labels, rids, keep_idx = reduce_mit_augmentation(
        data["beats"], data["labels"], full_rids)

    print("恢复 AAMI 符号 ...", flush=True)
    per_rec_syms = recover_mit_symbols_per_record()
    incart_dir = Path(__file__).resolve().parent / "data" / "raw" / "incart"
    per_rec_syms.update(recover_incart_symbols_per_record(incart_dir))
    sym_full, n_incart_unknown = align_symbols_to_npz(
        per_rec_syms, full_rids, 6)
    if sym_full is None:
        raise RuntimeError("AAMI symbol alignment failed")
    symbols = sym_full[keep_idx]

    print("患者级划分 ...", flush=True)
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat
                 for rid, pat in build_incart_patient_map().items()})
    tr_m, va_m, te_m, pstats = patient_level_split(rids, pmap)
    train_recs = set(np.unique(rids[tr_m]).tolist())
    test_recs = set(np.unique(rids[te_m]).tolist())
    if train_recs & test_recs:
        raise RuntimeError("train/test record leakage detected")

    x_te = add_channel_dim(beats[te_m].astype(np.float32))
    y_te = np.asarray(labels[te_m]).astype(np.int32)
    sym_te = symbols[te_m]
    print(f"noaug test set: n={len(y_te)}, abn={int(y_te.sum())}, "
          f"records={len(test_recs)}", flush=True)

    results = []
    for spec in DEFAULT_MODELS:
        results.append(evaluate_model(spec, x_te, y_te, sym_te))
        tf.keras.backend.clear_session()

    out_json = MODELS_DIR / "aami_matrix_deploy_patient_noaug.json"
    out_csv = MODELS_DIR / "aami_matrix_deploy_patient_noaug.csv"
    output = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "AAMI superclass matrix under deployment chain, noaug test beats",
            "eval_chain": "deploy",
            "split": "patient-level 60/20/20, seed=42, MIT+INCART",
            "noaug_rule": "MIT 6x augmented deploy data reduced to first raw block per record; INCART kept unchanged",
            "thresholds": THRESHOLDS,
            "n_test": int(len(y_te)),
            "n_abn": int((y_te == 1).sum()),
            "n_incart_unknown": int(n_incart_unknown),
            "train_records": len(train_recs),
            "test_records": len(test_recs),
            "train_test_record_intersection": 0,
            "patient_stats": pstats,
            "metric_notes": [
                "Per-class table reports recall/AUC only; classwise precision is omitted because it can be an identity when no negative samples exist (AGENTS §30).",
                "Global precision/FAR are reported separately.",
                "INCART beats without recoverable .atr symbols are excluded from per-class rows but included in global rows.",
            ],
        },
        "models": results,
    }
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    write_csv(results, out_csv)
    print(f"\nSaved: {out_json}", flush=True)
    print(f"Saved: {out_csv}", flush=True)
    print(f"elapsed={time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
