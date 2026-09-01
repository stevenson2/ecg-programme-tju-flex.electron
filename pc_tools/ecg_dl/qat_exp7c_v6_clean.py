#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qat_exp7c_v6_clean.py — 无泄漏 QAT：基于 exp7b + 仅 train 患者数据
================================================================================
修复 v3b/v4/v5 患者级泄漏：
  - 基础模型改为 patient-clean 的 best_resnet_large_exp7b.h5
  - MIT/INCART/PTB 金标准只取 patient_level_split 的 train 患者
  - PTB-XL 公共正常拍作为外部辅助（不与现有 MIT/PTB 测试患者身份重叠）
  - 真实 AFE 正常拍照旧
  - 验证/参数选择后续在 val 患者，测试冻结

输出：
  models/ecg_model_exp7c_clean_v6_qat_int8.tflite
  models/qat_exp7c_clean_v6.json
"""
import sys, os, json, time
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BEAT_WINDOW_SAMPLES
from models.resnet_lite_1d import build_ecg_resnet_lite_large
from data.patient_split import (
    build_mit_patient_map, build_incart_patient_map, build_ptb_patient_map,
    patient_level_split,
)


BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA_REAL = BASE / "data" / "real"
ECG_DATA = Path(os.environ.get("ECG_PROCESSED_DIR", "/home/devcontainers/ecg_data"))
OUT_TFLITE = MODELS / "ecg_model_exp7c_clean_v6_qat_int8.tflite"
OUT_JSON = MODELS / "qat_exp7c_clean_v6.json"
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
)
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_configs


class CustomECGQuantizeRegistry(Default8BitQuantizeRegistry):
    def __init__(self, disable_per_axis=False):
        super().__init__(disable_per_axis=disable_per_axis)
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


def load_arrays(tag):
    return (np.load(ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy", mmap_mode="r"),
            np.load(ECG_DATA / f"{tag}_processed_deploy_causal_labels.npy", mmap_mode="r"),
            np.load(ECG_DATA / f"{tag}_processed_deploy_causal_record_ids.npy", mmap_mode="r"))


def get_train_masks():
    # MIT+INCART
    mit_b, mit_l, mit_r = load_arrays("mit_bih")
    inc_b, inc_l, inc_r = load_arrays("incart")
    merged_rids = np.concatenate([mit_r, inc_r + 100000])
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat for rid, pat in build_incart_patient_map().items()})
    tr_m, va_m, te_m, stats = patient_level_split(merged_rids, pmap, seed=SEED)
    mit_tr = tr_m[:len(mit_r)]
    inc_tr = tr_m[len(mit_r):]
    # PTB
    ptb_b, ptb_l, ptb_r = load_arrays("ptb")
    tr_p, va_p, te_p, pstats = patient_level_split(ptb_r, build_ptb_patient_map(), seed=SEED)
    return {
        "mit_bih": mit_tr,
        "incart": inc_tr,
        "ptb": tr_p,
    }, stats, pstats


def load_domain_clean(tag, n_abn, n_norm, train_mask):
    b, l, r = load_arrays(tag)
    mask = train_mask[:len(l)] & (l == 1)
    ia = np.where(mask)[0]
    mask = train_mask[:len(l)] & (l == 0)
    inn = np.where(mask)[0]
    sa = rng.choice(ia, min(n_abn, len(ia)), replace=False)
    sn = rng.choice(inn, min(n_norm, len(inn)), replace=False)
    return (np.asarray(b[sa], dtype=np.float32), np.ones(len(sa), dtype=np.int32),
            np.asarray(b[sn], dtype=np.float32), np.zeros(len(sn), dtype=np.int32))


def main():
    t0 = time.time()
    masks, mit_stats, ptb_stats = get_train_masks()
    print(f"[CLEAN QAT] MIT/INCART patient stats: train={mit_stats['n_train']} "
          f"val={mit_stats['n_val']} test={mit_stats['n_test']}", flush=True)
    print(f"[CLEAN QAT] PTB patient stats: train={ptb_stats['n_train']} "
          f"val={ptb_stats['n_val']} test={ptb_stats['n_test']}", flush=True)

    # 金标准（只用 train 患者）
    mit_a, mit_al, mit_n, mit_nl = load_domain_clean("mit_bih", 1200, 400, masks["mit_bih"])
    inc_a, inc_al, inc_n, inc_nl = load_domain_clean("incart", 300, 100, masks["incart"])
    ptb_a, ptb_al, ptb_n, ptb_nl = load_domain_clean("ptb", 500, 150, masks["ptb"])
    x_mix = np.concatenate([mit_a, inc_a, ptb_a, mit_n, inc_n, ptb_n])[..., np.newaxis]
    y_mix = np.concatenate([mit_al, inc_al, ptb_al, mit_nl, inc_nl, ptb_nl])
    print(f"[CLEAN QAT] gold mix={len(x_mix)} abn={int((y_mix==1).sum())}", flush=True)

    # 真实 AFE 正常 + 公共正常
    real = np.concatenate([
        np.load(DATA_REAL / "real_normal_beats_exp7c.npy"),
        np.load(DATA_REAL / "real_normal_beats_rec_latest.npy"),
    ]).astype(np.float32)
    pubnorm = np.load(MODELS / "ecgfounder" / "real_like_normal_beats.npy").astype(np.float32)

    x_real = real[..., np.newaxis]
    y_real = np.zeros(len(real), dtype=np.int32)
    x_pub = pubnorm[..., np.newaxis]
    y_pub = np.zeros(len(pubnorm), dtype=np.int32)

    x_train = np.concatenate([x_mix, x_real, x_pub])
    y_train = np.concatenate([y_mix, y_real, y_pub])
    w_train = np.concatenate([
        np.ones(len(x_mix), dtype=np.float32),
        np.full(len(x_real), 2.5, dtype=np.float32),
        np.full(len(x_pub), 1.0, dtype=np.float32),
    ])
    perm = rng.permutation(len(x_train))
    x_train, y_train, w_train = x_train[perm], y_train[perm], w_train[perm]
    # 验证集从金标准混合中取小量（仅用于 QAT 观察，不用于阈值选择）
    x_val = x_train[-400:]
    y_val = y_train[-400:]
    print(f"[CLEAN QAT] train={len(x_train)} val={len(x_val)}", flush=True)

    model = build_ecg_resnet_lite_large(input_shape=(BEAT_WINDOW_SAMPLES, 1))
    model.load_weights(str(MODELS / "best_resnet_large_exp7b.h5"))
    print("[CLEAN QAT] base=exp7b, applying tfmot...", flush=True)
    annotated_model = tfmot.quantization.keras.quantize_annotate_model(model)
    qat_model = tfmot.quantization.keras.quantize_apply(annotated_model, scheme=CustomECGQuantizeScheme())
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    hist = qat_model.fit(x_train, y_train, validation_data=(x_val, y_val),
                         batch_size=32, epochs=10, verbose=2,
                         sample_weight=w_train)

    def rep_ds():
        idx = rng.choice(len(x_train), 200, replace=False)
        for i in idx:
            yield [x_train[i:i + 1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_ds
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    OUT_TFLITE.write_bytes(tflite_model)
    print(f"[CLEAN QAT] wrote {OUT_TFLITE} ({len(tflite_model)} bytes)", flush=True)

    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Clean patient-level QAT v6: base exp7b, train-only MIT/INCART/PTB",
        "mit_incart_patients": {
            "train": mit_stats["n_train"], "val": mit_stats["n_val"], "test": mit_stats["n_test"],
        },
        "ptb_patients": {
            "train": ptb_stats["n_train"], "val": ptb_stats["n_val"], "test": ptb_stats["n_test"],
        },
        "data": {"train": int(len(x_train)), "real": int(len(real)),
                 "pub_normal": int(len(pubnorm)), "epochs": int(len(hist.epoch))},
        "tflite_size": len(tflite_model),
        "tflite_path": str(OUT_TFLITE.relative_to(BASE)),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[CLEAN QAT] done {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
