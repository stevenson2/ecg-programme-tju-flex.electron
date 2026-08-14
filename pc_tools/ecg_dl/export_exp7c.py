#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_exp7c.py — exp7c INT8 导出 → 固件 C 头文件 (与 export_exp7b.py 同流程)
校准集: 修正后因果链口径 MIT+INCART 700 拍 + PTB 300 拍 (与训练同链)
+ 真实正常拍 200 (域适应: 校准集覆盖部署分布)。
输出: models/ecg_model_exp7c_int8.tflite + include/ai_inference/ecg_model_data.h
"""
import sys
import re
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR
from data.dataset import (
    load_mit_incart_merged, load_ptb_data, train_val_test_split, set_npz_suffix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
H5_PATH = MODELS_DIR / "best_resnet_large_exp7c.h5"
OUT_TFLITE = MODELS_DIR / "ecg_model_exp7c_int8.tflite"
OUT_HEADER = REPO_ROOT / "include" / "ai_inference" / "ecg_model_data.h"

print(f"[T3 exp7c] 模型: {H5_PATH.name}")

# ---------- 1. 校准集 (修正后因果链, 双域 + 真实拍) ----------
set_npz_suffix("_deploy_causal")
data = load_mit_incart_merged()
splits = train_val_test_split(data["beats"], data["labels"], record_ids=data["record_ids"])
x_train = splits["train"][0]
rng = np.random.default_rng(0)
idx = rng.choice(len(x_train), 700, replace=False)
calib = [x_train[i:i + 1][..., np.newaxis].astype(np.float32) for i in idx]
ptb = load_ptb_data()
idx_ptb = rng.choice(len(ptb["beats"]), 300, replace=False)
calib += [ptb["beats"][i:i + 1][..., np.newaxis].astype(np.float32) for i in idx_ptb]
real = np.load(Path(__file__).resolve().parent / "data" / "real" / "real_normal_beats_exp7c.npy")
calib += [real[i:i + 1][..., np.newaxis].astype(np.float32)
          for i in rng.choice(len(real), min(200, len(real)), replace=False)]
print(f"[T3 exp7c] 校准集: MIT+INCART 700 + PTB 300 + 真实 200 = {len(calib)} 拍")


def rep_ds():
    for s in calib:
        yield [s]


# ---------- 2. INT8 全整数量化 ----------
model = tf.keras.models.load_model(str(H5_PATH), compile=False)
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_ds
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
print("[T3 exp7c] INT8 转换中...")
tflite_bytes = conv.convert()
OUT_TFLITE.write_bytes(tflite_bytes)
print(f"[T3 exp7c] TFLite: {OUT_TFLITE.name} ({len(tflite_bytes)/1024:.1f} KB)")

# ---------- 3. C 头文件 ----------
from export import tflite_to_c_array

tflite_to_c_array(
    str(OUT_TFLITE), str(OUT_HEADER),
    variable_name="ecg_model_data", guard_name="ECG_MODEL_DATA_H",
)

hdr = OUT_HEADER.read_text(encoding="utf-8")
m = re.search(r"const int ecg_model_data_len = (\d+);", hdr)
assert m and int(m.group(1)) == len(tflite_bytes), "头文件长度与 TFLite 不一致!"
print(f"[T3 exp7c] 校验通过: ecg_model_data_len = {m.group(1)} == tflite {len(tflite_bytes)} bytes")
print("[T3 exp7c] 完成. 下一步: pio run 编译检查")
