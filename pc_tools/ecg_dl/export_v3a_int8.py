#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_v3a_int8.py -- export v3-A clean baseline to INT8 TFLite.

Uses the clean v3-A train split (distill_pilot_teacher_targets.npz x_train_perm)
as the representative dataset.  No test contact.
"""
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
H5 = MODELS / "best_resnet_large_clean_baseline_v3.h5"
DATA = CACHE / "distill_pilot_teacher_targets.npz"
OUT = CACHE / "ecg_model_v3a_int8.tflite"


def representative_dataset():
    d = np.load(DATA)
    x = np.asarray(d["x_train_perm"], dtype=np.float32)
    # Shuffle deterministically and take a representative 1000 samples.
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x), min(1000, len(x)), replace=False)
    for i in idx:
        yield [x[i:i + 1, ..., np.newaxis].astype(np.float32)]


def main():
    t0 = time.time()
    model = tf.keras.models.load_model(str(H5), compile=False)
    print(f"[V3X] loaded {H5.name}, params={model.count_params():,}", flush=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    OUT.write_bytes(tflite_model)
    print(f"[V3X] saved {OUT.name}: {len(tflite_model)} bytes "
          f"({len(tflite_model)/1024:.1f} KB), {time.time()-t0:.1f}s", flush=True)

    it = tf.lite.Interpreter(model_path=str(OUT))
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    print(f"[V3X] input: {inp['shape']} {inp['dtype']} "
          f"scale={inp['quantization_parameters']['scales'].flatten()} "
          f"zp={inp['quantization_parameters']['zero_points'].flatten()}", flush=True)
    print(f"[V3X] output: {out['shape']} {out['dtype']}", flush=True)


if __name__ == "__main__":
    main()
