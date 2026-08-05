#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_exp6_sgd.py — T0-1: exp6-SGD 部署链定稿模型 INT8 导出 → 固件 C 头文件
=============================================================================
任务: 必做清单 T0-1 (2026-08-05)
输入: models/best_resnet_large_exp6_sgd.h5  (deploy_match/retrain_exp6_sgd_eval.json 的模型)
输出: models/ecg_model_exp6_sgd_int8.tflite  (INT8, 全整数量化)
      <repo>/include/ai_inference/ecg_model_data.h  (固件头文件, 替换旧 CNN-v2)
校准集: 部署链口径 (set_npz_suffix("_deploy")): MIT+INCART train 700 拍 + PTB 300 拍
  (与 export_dual_tflite.py 一致的双域校准策略, 校准分布与模型训练口径匹配)
用法 (WSL, 与训练环境一致):
  export ECG_PROCESSED_DIR=$HOME/ecg_data
  python3 export_exp6_sgd.py
验证: 产物后运行 eval_deploy_match.py --int8 复核量化损失 (见 TUNING_HISTORY 新章)
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR, INFERENCE_CONFIG, CLASS_NAMES
from data.dataset import (
    load_mit_incart_merged, load_ptb_data, train_val_test_split, set_npz_suffix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]  # <repo>/pc_tools/ecg_dl/../..
H5_PATH = MODELS_DIR / "best_resnet_large_exp6_sgd.h5"
OUT_TFLITE = MODELS_DIR / "ecg_model_exp6_sgd_int8.tflite"
OUT_HEADER = REPO_ROOT / "include" / "ai_inference" / "ecg_model_data.h"

print(f"[T0-1] 模型: {H5_PATH.name}")
print(f"[T0-1] 输出 TFLite: {OUT_TFLITE}")
print(f"[T0-1] 输出头文件: {OUT_HEADER}")

# ---------- 1. 校准集 (部署链口径, 双域) ----------
set_npz_suffix("_deploy")
data = load_mit_incart_merged()
splits = train_val_test_split(data["beats"], data["labels"], record_ids=data["record_ids"])
x_train = splits["train"][0]
rng = np.random.default_rng(0)
idx = rng.choice(len(x_train), 700, replace=False)
calib = [x_train[i:i + 1][..., np.newaxis].astype(np.float32) for i in idx]
print(f"[T0-1] 校准集 MIT+INCART (deploy): {len(calib)} 拍")

ptb = load_ptb_data()
idx_ptb = rng.choice(len(ptb["beats"]), 300, replace=False)
calib += [ptb["beats"][i:i + 1][..., np.newaxis].astype(np.float32) for i in idx_ptb]
print(f"[T0-1] 校准集 PTB (deploy): {len(calib) - 700} 拍, 总计 {len(calib)} 拍")


def rep_ds():
    for s in calib:
        yield [s]


# ---------- 2. INT8 全整数量化 ----------
if not H5_PATH.exists():
    raise SystemExit(f"[T0-1] 权重缺失: {H5_PATH}")
model = tf.keras.models.load_model(str(H5_PATH), compile=False)
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_ds
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
print("[T0-1] INT8 转换中...")
tflite_bytes = conv.convert()
OUT_TFLITE.write_bytes(tflite_bytes)
print(f"[T0-1] ✅ TFLite 已保存: {OUT_TFLITE} ({len(tflite_bytes)/1024:.1f} KB)")

# ---------- 3. C 头文件 (变量名/guard 与 src/ai_inference/ai_inference.cpp 引用一致) ----------
from export import tflite_to_c_array

tflite_to_c_array(
    str(OUT_TFLITE), str(OUT_HEADER),
    variable_name="ecg_model_data", guard_name="ECG_MODEL_DATA_H",
)

# ---------- 4. 校验: 头文件数组长度 == tflite 字节数 ----------
import re

hdr = OUT_HEADER.read_text(encoding="utf-8")
m = re.search(r"const int ecg_model_data_len = (\d+);", hdr)
assert m and int(m.group(1)) == len(tflite_bytes), "头文件长度与 TFLite 不一致!"
print(f"[T0-1] ✅ 校验通过: ecg_model_data_len = {m.group(1)} == tflite {len(tflite_bytes)} bytes")
print(f"[T0-1] ✅ 完成. 下一步: pio run 编译检查")
