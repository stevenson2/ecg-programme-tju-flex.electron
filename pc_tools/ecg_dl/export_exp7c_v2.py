#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_exp7c_v2.py — 校准集加量变体: MIT+INCART 2000 + PTB 3000 + 真实 200
(PTB 域 3000 拍全覆盖其信号幅度分布, 收窄 INT8 量化误差)"""
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

set_npz_suffix("_deploy_causal")
data = load_mit_incart_merged()
splits = train_val_test_split(data["beats"], data["labels"], record_ids=data["record_ids"])
x_train = splits["train"][0]
rng = np.random.default_rng(0)
idx = rng.choice(len(x_train), 2000, replace=False)
calib = [x_train[i:i + 1][..., np.newaxis].astype(np.float32) for i in idx]
ptb = load_ptb_data()
idx_ptb = rng.choice(len(ptb["beats"]), 3000, replace=False)
calib += [ptb["beats"][i:i + 1][..., np.newaxis].astype(np.float32) for i in idx_ptb]
real = np.load(Path(__file__).resolve().parent / "data" / "real" / "real_normal_beats_exp7c.npy")
calib += [real[i:i + 1][..., np.newaxis].astype(np.float32)
          for i in rng.choice(len(real), min(200, len(real)), replace=False)]
print(f"[T3 exp7c v2] 校准集: MIT+INCART 2000 + PTB 3000 + 真实 200 = {len(calib)} 拍")


def rep_ds():
    for s in calib:
        yield [s]


model = tf.keras.models.load_model(str(H5_PATH), compile=False)
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_ds
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tflite_bytes = conv.convert()
OUT_TFLITE.write_bytes(tflite_bytes)
print(f"[T3 exp7c v2] TFLite: {len(tflite_bytes)/1024:.1f} KB")

from export import tflite_to_c_array
tflite_to_c_array(str(OUT_TFLITE), str(OUT_HEADER),
                  variable_name="ecg_model_data", guard_name="ECG_MODEL_DATA_H")
hdr = OUT_HEADER.read_text(encoding="utf-8")
m = re.search(r"const int ecg_model_data_len = (\d+);", hdr)
assert m and int(m.group(1)) == len(tflite_bytes)
print(f"[T3 exp7c v2] 头文件校验通过: {m.group(1)} bytes")
