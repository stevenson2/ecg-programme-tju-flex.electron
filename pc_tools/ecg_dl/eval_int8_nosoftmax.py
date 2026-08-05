#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_int8_nosoftmax.py — T3-6/M3: INT8 去 softmax 对照 (量化误差 vs softmax 压缩分离)
======================================================================
任务: 必做清单 T3-6 ③ / solutions.md M3
背景: 模型输出层自带 softmax → TFLite INT8 输出 = 概率量化; 固件/评估脚本
      (eval_deploy_match.py L2059) 再 softmax = 二次压缩 (double-softmax),
      P(abnormal) 动态范围被压缩至 ≈[0.27,0.73] (T0-1 发现)
对照:
  FP32  : Keras 输出 = softmax 概率 (参考)
  INT8-double : 反量化 → 再 softmax (固件当前语义)
  INT8-single : 反量化 → 直接取 p[:,1] (去二次 softmax)
分离: 量化误差 = single − FP32; softmax 压缩损失 = double − single
输出: models/int8_nosoftmax_eval.json
用法 (WSL): export ECG_PROCESSED_DIR=$HOME/ecg_data; python3 eval_int8_nosoftmax.py
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_deploy_match import CACHE_DIR, _add_channel_dim

MODELS = Path(__file__).resolve().parent / "models"
OUT_JSON = MODELS / "int8_nosoftmax_eval.json"
H5 = MODELS / "best_resnet_large_exp6_sgd.h5"
TFLITE = MODELS / "ecg_model_exp6_sgd_int8.tflite"


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    t0 = time.time()
    print("=" * 70)
    print("T3-6/M3 INT8 去 softmax 对照")
    print("=" * 70)

    model = tf.keras.models.load_model(str(H5), compile=False)
    it = tf.lite.Interpreter(model_path=str(TFLITE))
    it.allocate_tensors()
    in_d = it.get_input_details()[0]
    out_d = it.get_output_details()[0]
    in_scale = float(in_d["quantization_parameters"]["scales"].flatten()[0])
    in_zp = int(in_d["quantization_parameters"]["zero_points"].flatten()[0])
    out_scale = float(out_d["quantization_parameters"]["scales"].flatten()[0])
    out_zp = int(out_d["quantization_parameters"]["zero_points"].flatten()[0])
    print(f"in_scale={in_scale:.5f} in_zp={in_zp} | out_scale={out_scale:.6f} out_zp={out_zp}")

    results = {}
    for dom in ("mit", "ptb"):
        d = np.load(CACHE_DIR / f"{dom}_deploy_match.npz")
        beats, labels = d["beats_deploy"], d["labels"]
        # FP32
        p32 = model.predict(_add_channel_dim(beats), batch_size=512, verbose=0)[:, 1]
        # INT8 (批处理)
        xs = np.clip(np.round(beats[..., None].astype(np.float32) / in_scale + in_zp),
                     -128, 127).astype(np.int8)
        y_int8 = np.zeros((len(beats), 2), dtype=np.float32)
        it.resize_tensor_input(0, [len(beats), 250, 1], strict=False)
        it.allocate_tensors()
        it.set_tensor(in_d["index"], xs)
        it.invoke()
        y_fp = (it.get_tensor(out_d["index"]).astype(np.float32) - out_zp) * out_scale
        p_single = y_fp[:, 1]              # 去二次 softmax (输出已是概率)
        p_double = softmax(y_fp)[:, 1]     # 固件语义 (二次 softmax)

        auc32 = float(roc_auc_score(labels, p32))
        auc_s = float(roc_auc_score(labels, p_single))
        auc_d = float(roc_auc_score(labels, p_double))
        results[dom] = {
            "auc_fp32": auc32, "auc_int8_single": auc_s, "auc_int8_double": auc_d,
            "quant_error": auc_s - auc32,       # 纯量化误差
            "softmax_loss": auc_d - auc_s,      # 二次 softmax 压缩损失
            "p_range_single": [float(p_single.min()), float(p_single.max())],
            "p_range_double": [float(p_double.min()), float(p_double.max())],
            "n_beats": int(len(beats)),
        }
        print(f"[{dom}] FP32={auc32:.4f} | INT8-single={auc_s:.4f} "
              f"(量化 {auc_s-auc32:+.4f}) | INT8-double={auc_d:.4f} "
              f"(softmax {auc_d-auc_s:+.4f})")
        print(f"      概率范围: single [{p_single.min():.3f},{p_single.max():.3f}] "
              f"double [{p_double.min():.3f},{p_double.max():.3f}]")

    output = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "T3-6/M3 INT8 去 softmax 对照",
            "model": "exp6-SGD (h5 + ecg_model_exp6_sgd_int8.tflite)",
            "method": "FP32 (Keras softmax) vs INT8-single (反量化直接取概率) vs "
                      "INT8-double (固件语义二次 softmax); 分离量化误差与 softmax 压缩",
        },
        "results": results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 已保存: {OUT_JSON} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
