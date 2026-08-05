#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_exp6_sgd_int8.py — T0-1 验证: INT8 导出模型的部署链评估 vs retrain_exp6_sgd_eval.json (FP32 D3)
语义: 与固件一致 — INT8 输出反量化 → softmax → P(abnormal)
用法 (WSL): python3 verify_exp6_sgd_int8.py
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

BASE = Path("/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl")
TFLITE = BASE / "models" / "ecg_model_exp6_sgd_int8.tflite"
DM = BASE / "models" / "deploy_match"

# 参考数字 (retrain_exp6_sgd_eval.json, FP32 D3)
REF = {"mit": 0.9121911776741493, "ptb": 0.7696956852459305}

it = tf.lite.Interpreter(model_path=str(TFLITE))
it.allocate_tensors()
in_d = it.get_input_details()[0]
out_d = it.get_output_details()[0]
print("input:", in_d["shape"], in_d["dtype"], "| output:", out_d["shape"], out_d["dtype"])
in_scale = float(in_d["quantization_parameters"]["scales"].flatten()[0])
in_zp = int(in_d["quantization_parameters"]["zero_points"].flatten()[0])
out_scale = float(out_d["quantization_parameters"]["scales"].flatten()[0])
out_zp = int(out_d["quantization_parameters"]["zero_points"].flatten()[0])


def predict(x):
    xs = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
    out = np.zeros((len(x), 2), dtype=np.float32)
    for i in range(len(x)):
        it.set_tensor(in_d["index"], xs[i:i + 1])
        it.invoke()
        o = it.get_tensor(out_d["index"])
        o = (o.astype(np.float32) - out_zp) * out_scale
        e = np.exp(o - o.max(axis=-1, keepdims=True))
        out[i] = (e / e.sum(axis=-1, keepdims=True))[:, 1]
    return out[:, 1]


for tag, ref_auc in REF.items():
    d = np.load(f"{DM}/{tag}_deploy_match.npz")
    x, y = d["beats_deploy"], d["labels"]
    x = x[..., None].astype(np.float32)
    p = predict(x)
    auc = roc_auc_score(y, p)
    thr35 = (p >= 0.35).astype(int)
    tp = ((thr35 == 1) & (y == 1)).sum(); fp = ((thr35 == 1) & (y == 0)).sum()
    fn = ((thr35 == 0) & (y == 1)).sum()
    r35 = tp / max(1, tp + fn); pr35 = tp / max(1, tp + fp)
    print(f"[{tag}] INT8 D3 AUC = {auc:.4f} | FP32 ref = {ref_auc:.4f} | Δ = {auc - ref_auc:+.4f} | R@0.35 = {r35:.4f} P@0.35 = {pr35:.4f} (n={len(x)}, abn={y.sum()})")
