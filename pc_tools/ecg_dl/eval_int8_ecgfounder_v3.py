#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_int8_ecgfounder_v3.py — ECGFounder v3 QAT INT8 评估
================================================================================
评估：
  - 原 exp7c float
  - exp7c_ecgfounder_v3 float
  - exp7c_ecgfounder_v3 QAT INT8（BUILTIN_REF）
在 MIT/PTB 部署链缓存和真实 AFE 正常拍上的 AUC / 置信度。
"""
import sys, json, time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.lite.python.interpreter import OpResolverType

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
DATA_REAL = BASE / "data" / "real"

FLOAT_BASE = MODELS / "best_resnet_large_exp7c.h5"
FLOAT_V3 = MODELS / "best_resnet_large_exp7c_ecgfounder_v3.h5"
TFLITE_V3 = MODELS / "ecg_model_exp7c_ecgfounder_v3_qat_int8.tflite"
OUT = MODELS / "deploy_match" / "int8_ecgfounder_v3_eval.json"


def run_tflite_probs(path, beats, batch=None):
    interp = tf.lite.Interpreter(
        model_path=str(path),
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
        num_threads=1,
    )
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale = float(in_d["quantization_parameters"]["scales"].flatten()[0])
    in_zp = int(in_d["quantization_parameters"]["zero_points"].flatten()[0])
    out_scale = float(out_d["quantization_parameters"]["scales"].flatten()[0])
    out_zp = int(out_d["quantization_parameters"]["zero_points"].flatten()[0])
    probs = np.zeros(len(beats), dtype=np.float64)
    for i, b in enumerate(beats):
        xq = np.clip(np.round(b.astype(np.float32)[None, :, None] / in_scale) + in_zp,
                     -128, 127).astype(np.int8)
        interp.set_tensor(in_d["index"], xq)
        interp.invoke()
        q = interp.get_tensor(out_d["index"])[0]
        p = (q.astype(np.float32) - out_zp) * out_scale
        probs[i] = float(p[1])
        if (i + 1) % 5000 == 0:
            print(f"  TFLite {i+1}/{len(beats)}", flush=True)
    return probs


def main():
    real = np.concatenate([
        np.load(DATA_REAL / "real_normal_beats_exp7c.npy"),
        np.load(DATA_REAL / "real_normal_beats_rec_latest.npy"),
    ]).astype(np.float32)
    real = real[:311]  # 与微调实验一致的 311 拍
    print(f"[EVAL] real={len(real)}", flush=True)

    results = {}
    # float baselines
    for name, path in [("exp7c", FLOAT_BASE), ("v3_float", FLOAT_V3)]:
        model = tf.keras.models.load_model(str(path), compile=False)
        results[name] = {}
        for dom in ("mit", "ptb"):
            d = np.load(CACHE / f"{dom}_deploy_causal_match.npz")
            p = model.predict(d["beats"].astype(np.float32)[..., None],
                              batch_size=512, verbose=0)[:, 1]
            results[name][dom] = {
                "auc": float(roc_auc_score(d["labels"], p)),
                "n": int(len(p)),
            }
            print(f"[{name}] {dom} AUC={results[name][dom]['auc']:.4f}", flush=True)
        p = model.predict(real[..., None], batch_size=64, verbose=0)[:, 1]
        results[name]["real_normal"] = {
            "mean": float(p.mean()), "frac_gt_0.5": float((p > 0.5).mean()),
            "n": int(len(p)),
        }
        print(f"[{name}] real mean={p.mean():.4f} frac>0.5={(p>0.5).mean():.4f}", flush=True)
        del model
        tf.keras.backend.clear_session()

    # QAT INT8
    results["v3_qat_int8"] = {}
    for dom in ("mit", "ptb"):
        d = np.load(CACHE / f"{dom}_deploy_causal_match.npz")
        p = run_tflite_probs(TFLITE_V3, d["beats"])
        results["v3_qat_int8"][dom] = {
            "auc": float(roc_auc_score(d["labels"], p)),
            "n": int(len(p)),
        }
        print(f"[v3_qat_int8] {dom} AUC={results['v3_qat_int8'][dom]['auc']:.4f}", flush=True)
    p = run_tflite_probs(TFLITE_V3, real)
    results["v3_qat_int8"]["real_normal"] = {
        "mean": float(p.mean()), "frac_gt_0.5": float((p > 0.5).mean()),
        "n": int(len(p)),
    }
    print(f"[v3_qat_int8] real mean={p.mean():.4f} frac>0.5={(p>0.5).mean():.4f}", flush=True)

    out = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "exp7c": str(FLOAT_BASE),
            "v3_float": str(FLOAT_V3),
            "v3_qat_int8": str(TFLITE_V3),
        },
        "pc_resolver": "BUILTIN_REF",
        "results": results,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[EVAL] saved {OUT}")


if __name__ == "__main__":
    main()
