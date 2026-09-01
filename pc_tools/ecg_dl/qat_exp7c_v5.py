#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qat_exp7c_v5.py — exp7c_ecgfounder_v3 INT8 QAT + full PTB-XL hard-normal
================================================================================
目的：
  用 tensorflow-model-optimization 对 exp7c FP32 模型做量化感知训练，
  再导出 INT8 TFLite，尽量减少后训练量化损失。

流程：
  1. 加载 best_resnet_large_exp7c_ecgfounder_v3.h5
  2. tfmot.quantization.keras.quantize_model
  3. 用因果部署链混合数据 + 真实正常拍微调少量 epoch
  4. 导出 INT8 TFLite: models/ecg_model_exp7c_ecgfounder_v5_qat_int8.tflite
"""
import sys, os, time
from pathlib import Path

# TF 2.21 + tfmot 0.8.1 需要 Legacy Keras 兼容
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BEAT_WINDOW_SAMPLES
from models.resnet_lite_1d import build_ecg_resnet_lite_large

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA_REAL = BASE / "data" / "real"
ECG_DATA = Path(os.environ.get("ECG_PROCESSED_DIR", "/home/devcontainers/ecg_data"))
OUT_TFLITE = MODELS / "ecg_model_exp7c_ecgfounder_v5_qat_int8.tflite"
OUT_JSON = MODELS / "qat_exp7c_v5.json"
SEED = 42
rng = np.random.default_rng(SEED)
tf.random.set_seed(SEED)

from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import (
    default_8bit_quantize_registry,
    default_8bit_quantize_scheme,
)
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_registry import (
    Default8BitQuantizeRegistry,
    Default8BitQuantizeConfig,
    Default8BitConvQuantizeConfig,
)
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_configs


class CustomECGQuantizeRegistry(Default8BitQuantizeRegistry):
    """Default 8-bit registry extended with Conv1D / DepthwiseConv1D support."""
    def __init__(self, disable_per_axis=False):
        super().__init__(disable_per_axis=disable_per_axis)
        # 对 Conv1D 使用非 per-axis 的通用 Default8BitQuantizeConfig，避免 tfmot 的
        # ConvWeightsQuantizer 在 1D 权重量化时报 per_channel reduce_dims 未定义。
        self._layer_quantize_map[layers.Conv1D] = Default8BitQuantizeConfig(
            ['kernel'], ['activation'], False)
        self._layer_quantize_map[layers.DepthwiseConv1D] = Default8BitQuantizeConfig(
            ['depthwise_kernel'], ['activation'], False)
        self._layer_quantize_map[layers.BatchNormalization] = default_8bit_quantize_configs.NoOpQuantizeConfig()
        self._layer_quantize_map[layers.Multiply] = default_8bit_quantize_configs.NoOpQuantizeConfig()
        self._layer_quantize_map[layers.Reshape] = default_8bit_quantize_configs.NoOpQuantizeConfig()

class CustomECGQuantizeScheme(default_8bit_quantize_scheme.Default8BitQuantizeScheme):
    def get_quantize_registry(self):
        return CustomECGQuantizeRegistry(disable_per_axis=self._disable_per_axis)

def load_domain(tag, n_abn, n_norm):
    b = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy", mmap_mode="r")
    l = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_labels.npy", mmap_mode="r")
    ia = np.where(l == 1)[0]; inn = np.where(l == 0)[0]
    sa = rng.choice(ia, min(n_abn, len(ia)), replace=False)
    sn = rng.choice(inn, min(n_norm, len(inn)), replace=False)
    # 患者级泄漏守卫 (2026-09 审计: 本脚本历史抽样混入测试患者, 见
    # models/deploy_match/provenance_leakage_audit.json); 再次运行将直接失败。
    from pathlib import Path as _P
    import sys as _sys
    _sys.path.insert(0, str(_P(__file__).resolve().parent))
    from data.split_guard import get_guard
    import os as _os
    _ecg = _os.environ.get("ECG_PROCESSED_DIR", "/home/devcontainers/ecg_data")
    r = np.load(str(_ecg) + "/" + tag + "_processed_deploy_causal_record_ids.npy")
    get_guard(tag).assert_train_only(np.concatenate([r[sa], r[sn]]),
                                     context="load_domain")
    return (np.asarray(b[sa], dtype=np.float32), np.ones(len(sa), dtype=np.int32),
            np.asarray(b[sn], dtype=np.float32), np.zeros(len(sn), dtype=np.int32))

def main():
    t0 = time.time()
    real = np.load(DATA_REAL / "real_normal_beats_exp7c.npy").astype(np.float32)
    extra = np.load(DATA_REAL / "real_normal_beats_rec_latest.npy").astype(np.float32)
    real = np.concatenate([real, extra])

    mit_a, mit_al, mit_n, mit_nl = load_domain("mit_bih", 1200, 400)
    inc_a, inc_al, inc_n, inc_nl = load_domain("incart", 300, 100)
    ptb_a, ptb_al, ptb_n, ptb_nl = load_domain("ptb", 500, 150)
    x_mix = np.concatenate([mit_a, inc_a, ptb_a, mit_n, inc_n, ptb_n])[..., np.newaxis]
    y_mix = np.concatenate([mit_al, inc_al, ptb_al, mit_nl, inc_nl, ptb_nl])
    x_real = real[..., np.newaxis]
    y_real = np.zeros(len(real), dtype=np.int32)
    pubnorm = np.load(MODELS / "ecgfounder" / "real_like_normal_beats.npy").astype(np.float32)
    hardnorm = np.load(MODELS / "ecgfounder" / "full_normal_hard_v3b.npy").astype(np.float32)
    x_pubnorm = pubnorm[..., np.newaxis]
    y_pubnorm = np.zeros(len(pubnorm), dtype=np.int32)
    x_hardnorm = hardnorm[..., np.newaxis]
    y_hardnorm = np.zeros(len(hardnorm), dtype=np.int32)
    x_train = np.concatenate([x_mix, x_real, x_pubnorm, x_hardnorm])
    y_train = np.concatenate([y_mix, y_real, y_pubnorm, y_hardnorm])
    # 权重：真实 AFE 正常 2.5，公共正常 1.0，v3b 视角 hard normal 5.0
    w_train = np.concatenate([
        np.ones(len(x_mix), dtype=np.float32),
        np.full(len(x_real), 2.5, dtype=np.float32),
        np.full(len(pubnorm), 1.0, dtype=np.float32),
        np.full(len(hardnorm), 3.0, dtype=np.float32),
    ])
    perm = rng.permutation(len(x_train)); x_train = x_train[perm]; y_train = y_train[perm]
    x_val = x_train[-400:]; y_val = y_train[-400:]
    print(f"[QAT] train={len(x_train)} val={len(x_val)}", flush=True)

    model = build_ecg_resnet_lite_large(input_shape=(BEAT_WINDOW_SAMPLES, 1))
    model.load_weights(str(MODELS / "best_resnet_large_exp7c_ecgfounder_v3.h5"))
    print("[QAT] applying tfmot quantization...", flush=True)
    annotated_model = tfmot.quantization.keras.quantize_annotate_model(model)
    qat_model = tfmot.quantization.keras.quantize_apply(annotated_model, scheme=CustomECGQuantizeScheme())
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    # QAT only needs a few epochs for weight range adaptation
    hist = qat_model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        batch_size=32, epochs=10, verbose=2,
        sample_weight=w_train,
    )
    # convert to TFLite INT8 with representative dataset from training data
    def rep_ds():
        idx = rng.choice(len(x_train), 200, replace=False)
        for i in idx:
            yield [x_train[i:i+1]]
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_ds
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    OUT_TFLITE.write_bytes(tflite_model)
    print(f"[QAT] wrote {OUT_TFLITE} ({len(tflite_model)} bytes)", flush=True)

    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "exp7c_ecgfounder_v3 QAT -> INT8",
        "data": {"train": int(len(x_train)), "real": int(len(real)), "pub_normal": int(len(pubnorm)), "hard_normal": int(len(hardnorm)), "epochs": int(len(hist.epoch))},
        "tflite_size": len(tflite_model),
        "tflite_path": str(OUT_TFLITE.relative_to(BASE)),
    }
    OUT_JSON.write_text(__import__("json").dumps(result, indent=2, ensure_ascii=False))
    print(f"[QAT] done in {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
