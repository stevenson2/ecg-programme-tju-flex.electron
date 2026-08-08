#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_s_class.py — T3-6/M6: S 类 (SVEB) 构成分析 (拍级 vs 患者级召回差异归因)
======================================================================
任务: 必做清单 T3-6 ④ / solutions.md M6
目标: 验证"测试构成效应" — 拍级 S 召回 vs 患者级平均 S 召回之差是否由
      少数记录/患者含大量 S 拍且召回低所致
方法:
  1. 未增强测试拍 (T1-2 口径, n_aug_mit=1) + 患者级 test mask (seed 42)
  2. AAMI 符号恢复 + 逐拍对齐 (复用 eval_aami_breakdown)
  3. exp6-SGD 概率 → S 类拍级召回 (θ=0.5)
  4. 按记录/患者聚合: S 拍数 + 召回 → 低召回高样本识别 → 构成效应量化
     (拍级加权召回 vs 患者级平均召回)
输出: models/s_class_audit.json
用法 (WSL): export ECG_PROCESSED_DIR=$HOME/ecg_data; python3 audit_s_class.py
"""
import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, AAMI_CLASSES
from data.dataset import load_incart_data, add_channel_dim
from data.patient_split import (build_mit_patient_map, build_incart_patient_map,
                                patient_level_split)
from eval_aami_breakdown import (recover_mit_symbols_per_record,
                                 recover_incart_symbols_per_record,
                                 align_symbols_to_npz)
import data.preprocess_incart as _inc
from pathlib import Path as _P

MODELS = Path(__file__).resolve().parent / "models"
OUT_JSON = MODELS / "s_class_audit.json"
S_SYMS = {s for s, name in AAMI_CLASSES.items() if name == "S"}
INCART_DIR = _P("/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/"
                "ecg-programme-tju-flex.electron-master/"
                "st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/files")


def main():
    print("=" * 70)
    print("T3-6/M6 S 类构成分析 (未增强测试口径)")
    print("=" * 70)

    # ---- 数据: 未增强 MIT + INCART ----
    d_mit = np.load(PROCESSED_DIR / "mit_bih_processed_noaug.npz")
    inc = load_incart_data()
    beats = np.concatenate([d_mit["beats"], inc["beats"]], axis=0)
    labels = np.concatenate([d_mit["labels"], inc["labels"]], axis=0)
    rids = np.concatenate([d_mit["record_ids"], inc["record_ids"] + 100000], axis=0)
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat
                 for rid, pat in build_incart_patient_map().items()})
    tr, va, te, stats = patient_level_split(rids, pmap)
    x_test, y_test, r_test = beats[te], labels[te], rids[te]
    print(f"患者级 test: {len(x_test)} 拍")

    # ---- S 符号逐拍对齐 (未增强, n_aug=1) ----
    per_rec = recover_mit_symbols_per_record()
    per_rec.update(recover_incart_symbols_per_record(INCART_DIR))
    sym_full, n_unk = align_symbols_to_npz(per_rec, rids, n_aug_mit=1)
    if sym_full is None:
        raise SystemExit("符号对齐失败")
    sym_test = sym_full[te]
    n_s = int((sym_test == "S").sum())
    print(f"测试拍中 S 类: {n_s} 拍 ({n_s/len(sym_test)*100:.1f}%), "
          f"未知符号(INCART): {n_unk}")

    # ---- 模型概率 ----
    model = tf.keras.models.load_model(
        str(MODELS / "best_resnet_large_exp6_sgd.h5"), compile=False)
    prob = model.predict(add_channel_dim(x_test), batch_size=512, verbose=0)[:, 1]

    # ---- 拍级 S 召回 (θ=0.5) ----
    is_s = sym_test == "S"
    pred = prob >= 0.5
    s_rec_beat = float((pred[is_s] & y_test[is_s] == 1).mean()) if is_s.sum() else None
    # 注意: S 拍在二分类任务中标签为 1 (异常); 召回 = S 拍中被判异常比例
    s_rec_beat = float(pred[is_s].mean()) if is_s.sum() else None
    print(f"拍级 S 召回 (θ=0.5): {s_rec_beat:.4f}")

    # ---- 按记录聚合 ----
    rec_stat = defaultdict(lambda: {"n_s": 0, "tp_s": 0})
    for i in np.where(is_s)[0]:
        rec_stat[int(r_test[i])]["n_s"] += 1
        rec_stat[int(r_test[i])]["tp_s"] += int(pred[i])
    rec_rows = [(rid, st["n_s"], st["tp_s"] / st["n_s"]) for rid, st in rec_stat.items()]
    rec_rows.sort(key=lambda r: -r[1])
    # 低召回高样本记录 (样本 > 中位数 且 召回 < 总体)
    n_s_all = sum(r[1] for r in rec_rows)
    tp_all = sum(r[2] * r[1] for r in rec_rows)
    overall = tp_all / n_s_all
    bad_recs = [(rid, n, round(rec, 4)) for rid, n, rec in rec_rows
                if n > max(1, n_s_all / max(1, len(rec_rows))) and rec < overall]

    # ---- 按患者聚合 ----
    pat_of_rec = {rid: pmap.get(rid, f"unknown_{rid}") for rid in set(int(r) for r in r_test)}
    pat_stat = defaultdict(lambda: {"n_s": 0, "tp_s": 0})
    for rid, n, rec in rec_rows:
        p = pat_of_rec.get(rid, "?")
        pat_stat[p]["n_s"] += n
        pat_stat[p]["tp_s"] += int(round(rec * n))
    # 患者级平均召回 (每个患者等权)
    pat_vals = [st["tp_s"] / st["n_s"] for st in pat_stat.values() if st["n_s"] > 0]
    s_rec_patient = float(np.mean(pat_vals))
    print(f"患者级平均 S 召回: {s_rec_patient:.4f} (患者数 {len(pat_vals)})")
    print(f"构成效应: 拍级 {s_rec_beat:.4f} vs 患者级平均 {s_rec_patient:.4f} "
          f"(差 {s_rec_beat - s_rec_patient:+.4f})")
    print(f"低召回高样本记录: {bad_recs[:8]}")

    output = {
        "meta": {
            "date": "2026-08-06", "task": "T3-6/M6 S 类构成分析",
            "model": "exp6-SGD (best_resnet_large_exp6_sgd.h5)",
            "data": "未增强测试拍 (T1-2 口径) + 患者级划分 seed42",
            "method": "AAMI 符号逐拍对齐 (n_aug=1); S 拍 = 符号 S; 召回 = 判异常比例 (θ=0.5); "
                      "构成效应 = 拍级加权召回 − 患者级平均召回",
        },
        "result": {
            "n_s_beats": int(n_s), "n_records_with_s": len(rec_rows),
            "n_patients_with_s": len(pat_vals),
            "s_recall_beat_level": s_rec_beat,
            "s_recall_patient_mean": round(s_rec_patient, 4),
            "composition_effect": round(s_rec_beat - s_rec_patient, 4),
            "low_recall_high_sample_records": bad_recs[:10],
            "note": "构成效应 >0 表示低召回记录含大量 S 拍, 拉低拍级加权召回 "
                    "(患者级平均回避样本量加权)",
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 已保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
