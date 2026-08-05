#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史模型患者级重评估 (4.4-4)
=============================
用历史训练好的权重 + 患者级划分的测试集重新评估所有模型,
对比拍级/记录级 vs 患者级测试下的性能排名变化。

- MIT 域: MIT+INCART 患者级 test 拍 (模型训练时未见这些患者? 注意:
  历史模型训练集是记录级划分, INCART 同患者多记录可能跨 train/test,
  所以这是"保守下限"评估——患者级 test 中部分拍可能被训练见过)
- PTB 域: PTB 患者级 test 拍 (只在 exp4/exp5/ensemble/ssl 等含 PTB
  训练的模型上有泄漏风险; P2A 等未见 PTB 的是干净评估)
"""
import sys
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import load_mit_incart_merged, load_3beat_merged, add_channel_dim
from data.patient_split import (build_mit_patient_map, build_incart_patient_map,
                                build_ptb_patient_map, patient_level_split)

MODELS = Path(__file__).resolve().parent / "models"
OUT_JSON = MODELS / "patient_split_eval.json"

# ==================== 患者级划分 ====================
# 250 点单拍数据 (所有部署模型)
mit_inc = load_mit_incart_merged()
pmap = {}
pmap.update(build_mit_patient_map())
pmap.update({rid + 100000: "inc_" + pat
             for rid, pat in build_incart_patient_map().items()})
tr, va, te, stats = patient_level_split(mit_inc["record_ids"], pmap)
x_mit, y_mit = mit_inc["beats"][te], mit_inc["labels"][te]
print(f"[MIT域] 患者级 test (250点): {len(x_mit)} 拍 "
      f"(N={(y_mit==0).sum()}, A={(y_mit==1).sum()})")

# 750 点 3-beat 数据 (仅 CNN-M)
mit_3 = load_3beat_merged()
tr3, va3, te3, stats3 = patient_level_split(mit_3["record_ids"], pmap)
x_mit3, y_mit3 = mit_3["beats"][te3], mit_3["labels"][te3]
print(f"[MIT域] 患者级 test (750点): {len(x_mit3)} 拍 "
      f"(N={(y_mit3==0).sum()}, A={(y_mit3==1).sum()})")

# PTB 患者级 test
ptb_npz = Path(__file__).resolve().parent / "data" / "processed" / "ptb_processed.npz"
d = np.load(ptb_npz)
x_ptb, y_ptb, rids_ptb = d["beats"], d["labels"], d["record_ids"]
pmap_ptb = build_ptb_patient_map()
tr2, va2, te2, stats2 = patient_level_split(rids_ptb, pmap_ptb)
x_ptb, y_ptb = x_ptb[te2], y_ptb[te2]
print(f"[PTB域] 患者级 test: {len(x_ptb)} 拍 "
      f"(N={(y_ptb==0).sum()}, A={(y_ptb==1).sum()})")

# PTB 3-beat (CNN-M 专用)
from data.preprocess_3beat import stitch_3beat
x_ptb3, y_ptb3, rids_ptb3 = stitch_3beat(
    np.load(ptb_npz)["beats"], np.load(ptb_npz)["labels"], np.load(ptb_npz)["record_ids"])
# 3-beat 序列的 record_id 与原始一致, 直接复用 te2 mask 需对齐长度
# stitch 输出顺序 = 原始顺序, 长度 = len - 边界, 用 rids_ptb3 重新划分
tr3p, va3p, te3p, _ = patient_level_split(rids_ptb3, pmap_ptb)
x_ptb3, y_ptb3 = x_ptb3[te3p], y_ptb3[te3p]
print(f"[PTB域] 患者级 test (750点): {len(x_ptb3)} 拍")

# ==================== 模型清单 ====================
# 实测输入尺寸: 全部 250 点, 仅 CNN-M=750, PTBXL预训练=1000(跳过)
# 4.4-4 核查修正 (2026-08-01, sha256/h5py 权重对比证实):
#   - final_resnet_l_exp4_ptb.h5 是 exp3-focal 的逐位副本 (exp4 训练崩溃后
#     复制了残留 final_resnet_l.h5); 真 exp4 = best_resnet_large_exp4_ptb.h5
#   - archived/final_model.h5 是 158K 参数的 ResNet-M 架构变体 (非 ResNet-L!),
#     与 final_resnet_m.h5 权重不同 (独立训练), 改标注为 ResNet-M(存档v2)
#   - 通用名 best_resnet_large.h5 仅用于单次训练输出, 归档后必须改为实名
cands = [
    # (名称, 路径, 是否训练见过PTB, 输入点数)
    ("CNN-v2",           MODELS / "final_cnn_v2.h5", False, 250),
    ("CNN-M(3beat)",     MODELS / "final_cnn_m.h5", False, 750),
    ("ResNet-S(v3)",     MODELS / "archived" / "best_small.h5", False, 250),
    ("ResNet-M",         MODELS / "final_resnet_m.h5", False, 250),
    ("ResNet-M(存档v2)", MODELS / "archived" / "final_model.h5", False, 250),
    ("ResNet-L(focal)",  MODELS / "final_resnet_l_exp3_focal_a075.h5", False, 250),
    ("P2A(部署)",        MODELS / "archived" / "final_resnet_l_p2a_backup.h5", False, 250),
    ("exp4(ptb全量)",    MODELS / "best_resnet_large_exp4_ptb.h5", True, 250),
    ("exp4(患者级清洁)", MODELS / "best_resnet_large_exp4_patient_clean.h5", True, 250),
    ("exp5(ptb限量)",    MODELS / "best_resnet_large_exp5_ptb_capped.h5", True, 250),
    ("exp5(患者级清洁)", MODELS / "best_resnet_large_exp5_patient_clean.h5", True, 250),
    ("exp6(域平衡)",     MODELS / "best_resnet_large_exp6_domain_balanced.h5", True, 250),
    ("exp6(患者级清洁)", MODELS / "best_resnet_large_exp6_patient_clean.h5", True, 250),
    ("SSL微调",          MODELS / "final_ssl_finetuned.h5", False, 250),
    ("Ensemble(seed42)", MODELS / "ensemble_seed42.h5", False, 250),
    ("多任务",           MODELS / "final_resnet_multitask.h5", False, 250),
]

# 多阈值工作点 (蹊跷点5: 单阈值 0.5 对 FocalLoss 模型不公平;
# 历史 eval_deploy_decision.py 用 [0.35, 0.5, 0.65, 0.80])
THRESHOLDS = [0.35, 0.5, 0.65, 0.8]

results = []
for name, path, saw_ptb, in_len in cands:
    if not path.exists():
        print(f"{name}: 缺失 ({path.name})")
        continue
    m = tf.keras.models.load_model(str(path), compile=False)
    out = {"name": name, "saw_ptb": saw_ptb, "input_len": in_len,
           "file": path.name}
    # 按输入尺寸选择数据
    if in_len == 750:
        x_mit_d, y_mit_d = x_mit3, y_mit3
        x_ptb_d, y_ptb_d = x_ptb3, y_ptb3
    else:
        x_mit_d, y_mit_d = x_mit, y_mit
        x_ptb_d, y_ptb_d = x_ptb, y_ptb
    for dom, x, y in [("mit", x_mit_d, y_mit_d), ("ptb", x_ptb_d, y_ptb_d)]:
        xi = add_channel_dim(x)
        prob_raw = m.predict(xi, verbose=0, batch_size=512)
        # 多任务模型输出 [cls, bpm, sqi] 三头, 取分类头
        if isinstance(prob_raw, (list, tuple)):
            prob_raw = prob_raw[0]
        prob = prob_raw[:, 1]
        auc = roc_auc_score(y, prob)
        thr_out = {}
        for thr in THRESHOLDS:
            p, r, f1, _ = precision_recall_fscore_support(
                y, (prob >= thr).astype(int), average="binary",
                zero_division=0)
            thr_out[f"{thr:.2f}"] = {"rec": float(r), "prec": float(p),
                                     "f1": float(f1)}
        t05 = thr_out["0.50"]
        out[dom] = {"auc": float(auc),
                    "rec": t05["rec"], "prec": t05["prec"], "f1": t05["f1"],
                    "thr": thr_out,
                    "n_test": int(len(y)),
                    "p_abn|normal": float(prob[y == 0].mean()),
                    "p_abn|abn": float(prob[y == 1].mean())}
        r35 = thr_out["0.35"]["rec"]
        print(f"  [{dom}] {name}: AUC={auc:.4f} R@0.5={t05['rec']:.3f} "
              f"P@0.5={t05['prec']:.3f} F1={t05['f1']:.3f} R@0.35={r35:.3f}")
    results.append(out)
    del m
    tf.keras.backend.clear_session()

meta = {
    "generated": "2026-08-02 4.4-4 重跑 (患者级清洁重训 + 多阈值)",
    "seed": 42, "split": "患者级 60/20/20 (permutation 语义)",
    "thresholds": THRESHOLDS,
    "mit_test_250": int(len(y_mit)), "mit_test_750": int(len(y_mit3)),
    "ptb_test_250": int(len(y_ptb)), "ptb_test_750": int(len(y_ptb3)),
    "notes": [
        "蹊跷点1: exp4 改用 best_resnet_large_exp4_ptb.h5 (final 为 exp3 副本)",
        "蹊跷点1: ResNet-L(v2) 实为 ResNet-M 架构, 改标注 ResNet-M(存档v2)",
        "蹊跷点7: 750点 PTB 测试患者集合与250点不同 (stitch丢记录致患者漂移), 不可跨输入尺寸比拍数",
        "蹊跷点3: 本划分(permutation) 与历史 eval 脚本(choice) 测试患者交集仅 12/58",
        "MIT域测试拍含 6 倍增强拍 (旧37条时代模型训练数据无增强, 数据域不同, 为保守下限)",
    ],
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump({"meta": meta, "results": results}, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存: {OUT_JSON}")
