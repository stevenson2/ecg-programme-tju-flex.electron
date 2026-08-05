#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_patient_split_noaug.py — T1-2: 未增强测试拍患者级重评
================================================================
任务: 必做清单 T1-2 / solutions.md M5
方法:
  - MIT 域: 未增强测试拍 (mit_bih_processed_noaug.npz, 109,827 拍) + INCART 原始拍
    (INCART 本就无增强) → 同一患者级划分 (seed=42, 60/20/20)
  - PTB 域: 无增强, 用现有 ptb_processed.npz 复算 (一致性验证)
  - 模型: exp4/exp5/exp6 (患者级清洁) + P2A (部署)
输出: 追加 patient_split_eval.json 新条目 (名称后缀 "未增强测试")
用法 (WSL): export ECG_PROCESSED_DIR=$HOME/ecg_data; python3 eval_patient_split_noaug.py
"""
import sys
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR
from data.dataset import load_incart_data, load_ptb_data, add_channel_dim
from data.patient_split import (build_mit_patient_map, build_incart_patient_map,
                                build_ptb_patient_map, patient_level_split)

MODELS = Path(__file__).resolve().parent / "models"
OUT_JSON = MODELS / "patient_split_eval.json"
THRESHOLDS = [0.35, 0.5, 0.65, 0.8]

# ==================== 数据 ====================
# 未增强 MIT (T1-2 重建) + INCART 原始 (无增强)
d_mit = np.load(PROCESSED_DIR / "mit_bih_processed_noaug.npz")
x_mit_all, y_mit_all, rid_mit_all = d_mit["beats"], d_mit["labels"], d_mit["record_ids"]
print(f"[T1-2] 未增强 MIT: {len(x_mit_all)} 拍")
inc = load_incart_data()
rid_inc = inc["record_ids"] + 100000
beats = np.concatenate([x_mit_all, inc["beats"]], axis=0)
labels = np.concatenate([y_mit_all, inc["labels"]], axis=0)
rids = np.concatenate([rid_mit_all, rid_inc], axis=0)

pmap = {}
pmap.update(build_mit_patient_map())
pmap.update({rid + 100000: "inc_" + pat
             for rid, pat in build_incart_patient_map().items()})
tr, va, te, stats = patient_level_split(rids, pmap)
x_mit, y_mit = beats[te], labels[te]
print(f"[MIT域/未增强] test: {len(x_mit)} 拍 "
      f"(N={(y_mit == 0).sum()}, A={(y_mit == 1).sum()}), test 患者 {stats['n_test']}")

# PTB (无增强, 一致性复算)
ptb = load_ptb_data()
pmap_ptb = build_ptb_patient_map()
tr2, va2, te2, stats2 = patient_level_split(ptb["record_ids"], pmap_ptb)
x_ptb, y_ptb = ptb["beats"][te2], ptb["labels"][te2]
print(f"[PTB域] test: {len(x_ptb)} 拍 (N={(y_ptb == 0).sum()}, A={(y_ptb == 1).sum()})")

# ==================== 模型 ====================
cands = [
    ("exp4(患者级清洁/未增强测试)", MODELS / "best_resnet_large_exp4_patient_clean.h5"),
    ("exp5(患者级清洁/未增强测试)", MODELS / "best_resnet_large_exp5_patient_clean.h5"),
    ("exp6(患者级清洁/未增强测试)", MODELS / "best_resnet_large_exp6_patient_clean.h5"),
    ("P2A(部署/未增强测试)", MODELS / "archived" / "final_resnet_l_p2a_backup.h5"),
]

results = []
for name, path in cands:
    if not path.exists():
        print(f"{name}: 缺失 ({path.name})")
        continue
    m = tf.keras.models.load_model(str(path), compile=False)
    out = {"name": name, "saw_ptb": ("exp" in name), "input_len": 250, "file": path.name,
           "test_semantics": "no_augment"}
    for dom, x, y in [("mit", x_mit, y_mit), ("ptb", x_ptb, y_ptb)]:
        xi = add_channel_dim(x)
        prob_raw = m.predict(xi, verbose=0, batch_size=512)
        if isinstance(prob_raw, (list, tuple)):
            prob_raw = prob_raw[0]
        prob = prob_raw[:, 1]
        auc = roc_auc_score(y, prob)
        thr_out = {}
        for thr in THRESHOLDS:
            p, r, f1, _ = precision_recall_fscore_support(
                y, (prob >= thr).astype(int), average="binary", zero_division=0)
            thr_out[f"{thr:.2f}"] = {"rec": float(r), "prec": float(p), "f1": float(f1)}
        t05 = thr_out["0.50"]
        out[dom] = {"auc": float(auc),
                    "rec": t05["rec"], "prec": t05["prec"], "f1": t05["f1"],
                    "thr": thr_out,
                    "n_test": int(len(y)),
                    "p_abn|normal": float(prob[y == 0].mean()),
                    "p_abn|abn": float(prob[y == 1].mean())}
        print(f"  [{dom}] {name}: AUC={auc:.4f} R@0.5={t05['rec']:.3f} "
              f"P@0.5={t05['prec']:.3f} F1={t05['f1']:.3f} R@0.35={thr_out['0.35']['rec']:.3f}")
    results.append(out)
    del m
    tf.keras.backend.clear_session()

# ==================== 合并到 patient_split_eval.json ====================
with open(OUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
names = {r["name"] for r in data["results"]}
added = 0
for r in results:
    if r["name"] in names:
        print(f"[跳过] 已存在: {r['name']}")
        continue
    data["results"].append(r)
    added += 1
data["meta"]["notes"] = data["meta"].get("notes", []) + [
    "T1-2 (2026-08-05): 未增强测试拍重评 (M5) — MIT 域测试拍 = 原始拍 (mit_bih_processed_noaug.npz, "
    "6×增强变体剔除, 109,827 拍); 训练数据不变 (仍为 6× 增强); 患者级划分 seed=42 与主表一致 "
    "(划分仅依赖记录集合, 未增强后记录集合不变); PTB 域无增强, 复算值应与原条目一致 (一致性验证)",
]
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"\n[T1-2] ✅ 已追加 {added} 个未增强测试条目到 {OUT_JSON}")
