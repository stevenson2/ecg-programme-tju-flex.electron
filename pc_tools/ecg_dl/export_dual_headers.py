#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双专家 TFLite -> C 头文件 (ESP32 双模型部署)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR
from export import tflite_to_c_array

jobs = [
    ("ecg_model_p2a_int8.tflite", "ecg_model_p2a_data.h",
     "ecg_model_p2a_data", "ECG_MODEL_P2A_DATA_H"),
    ("ecg_model_exp5_int8.tflite", "ecg_model_exp5_data.h",
     "ecg_model_exp5_data", "ECG_MODEL_EXP5_DATA_H"),
]
for tflite, header, var, guard in jobs:
    src = MODELS_DIR / tflite
    if not src.exists():
        print(f"跳过 (不存在): {src}")
        continue
    tflite_to_c_array(str(src), str(MODELS_DIR / header),
                      variable_name=var, guard_name=guard)
print("\n[DONE] 双专家 C 头文件已生成")
