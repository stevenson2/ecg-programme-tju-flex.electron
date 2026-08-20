#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_kd_a070_t1.py — KD a070_t1 (心梗筛查专家) INT8 导出
=============================================================================
M0: 双专家规划前置验证 —— 把 KD a070_t1 导出 INT8，
    供 eval_ptbxl_record_level.py 按板上部署链做记录级验证。

输入: models/final_kd_a070_t1.h5 (KD 定稿, PTB 最优专家)
输出: models/ecg_model_kd_a070_t1_int8.tflite
校准集: 部署链口径 (set_npz_suffix("_deploy")): MIT+INCART train 700 拍 + PTB 300 拍
        (与 export_exp6_sgd.py / export_dual_tflite.py 同策略)
说明: 不生成 C 头文件，本脚本仅用于 PC 端部署链评测，不替换固件模型。
用法 (WSL): python3 export_kd_a070_t1.py
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR
from data.dataset import (
    load_mit_incart_merged, load_ptb_data, train_val_test_split, set_npz_suffix,
)

H5_PATH = MODELS_DIR / "final_kd_a070_t1.h5"
OUT_TFLITE = MODELS_DIR / "ecg_model_kd_a070_t1_int8.tflite"

print(f"[M0] 模型: {H5_PATH.name}")
print(f"[M0] 输出: {OUT_TFLITE}")

# ---------- 1. 校准集 (部署链口径, 双域混合) ----------
set_npz_suffix("_deploy")
data = load_mit_incart_merged()
splits = train_val_test_split(data["beats"], data["labels"], record_ids=data["record_ids"])
x_train = splits["train"][0]
rng = np.random.default_rng(0)
idx = rng.choice(len(x_train), 700, replace=False)
calib = [x_train[i:i + 1][..., np.newaxis].astype(np.float32) for i in idx]
print(f"[M0] 校准集 MIT+INCART (deploy): {len(calib)} 拍")

ptb = load_ptb_data()
idx_ptb = rng.choice(len(ptb["beats"]), 300, replace=False)
calib += [ptb["beats"][i:i + 1][..., np.newaxis].astype(np.float32) for i in idx_ptb]
print(f"[M0] 校准集 PTB (deploy): {len(calib) - 700} 拍, 总计 {len(calib)} 拍")


def rep_ds():
    for s in calib:
        yield [s]


# ---------- 2. INT8 全整数量化 ----------
if not H5_PATH.exists():
    raise SystemExit(f"[M0] 权重缺失: {H5_PATH}")

model = tf.keras.models.load_model(str(H5_PATH), compile=False)
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_ds
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
print("[M0] INT8 转换中...")
tflite_bytes = conv.convert()
OUT_TFLITE.write_bytes(tflite_bytes)
print(f"[M0] ✅ TFLite 已保存: {OUT_TFLITE} ({len(tflite_bytes)/1024:.1f} KB)")
