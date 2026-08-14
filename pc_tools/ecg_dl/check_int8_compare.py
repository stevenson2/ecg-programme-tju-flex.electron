#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对照: exp7b INT8 vs exp7c INT8 在 MIT/PTB/真实拍上的量化损耗"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"

def run_tflite(path, beats):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    scale, zp = out["quantization"]
    probs = []
    for i in range(len(beats)):
        xf = beats[i:i+1].astype(np.float32)[..., np.newaxis]
        xq = np.clip(np.round(xf / in_scale) + in_zp, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], xq)
        interp.invoke()
        q = interp.get_tensor(out["index"])
        f = (q.astype(np.float32) - zp) * scale
        e = np.exp(f - f.max(axis=1, keepdims=True))
        probs.append(e[0, 1] / e[0].sum())
        if (i + 1) % 5000 == 0:
            print(f"  ... {i+1}/{len(beats)}")
    return np.array(probs)

def auc(y, p):
    y = np.asarray(y).ravel(); p = np.asarray(p).ravel()
    order = np.argsort(p); y = y[order]
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = np.arange(1, len(y) + 1)[y == 1].sum()
    return float((ranks - n1 * (n1 + 1) / 2.0) / (n1 * n0))

tfls = {"exp7b": BASE / "models" / "ecg_model_exp7b_int8.tflite",
        "exp7c": BASE / "models" / "ecg_model_exp7c_int8.tflite"}
for name, tfl in tfls.items():
    for tag, f in [("mit", CACHE / "mit_deploy_causal_match.npz"),
                   ("ptb", CACHE / "ptb_deploy_causal_match.npz")]:
        d = np.load(f)
        p = run_tflite(tfl, d["beats"])
        print(f"{name} int8 {tag}: AUC={auc(d['labels'], p):.4f}")
real = np.load(BASE / "data" / "real" / "real_normal_beats_exp7c.npy")
for name, tfl in tfls.items():
    p = run_tflite(tfl, real)
    print(f"{name} int8 real: mean={p.mean():.4f} frac>0.5={(p>0.5).mean():.4f}")
