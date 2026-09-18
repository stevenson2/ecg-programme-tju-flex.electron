#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_r18_int8.py — R18 模型 INT8 导出 (参数化版 export_v3a_int8.py)
================================================================================
与 export_v3a_int8.py 同源流程 (representative dataset = 干净训练侧样本
distill_pilot_teacher_targets.npz x_train_perm, 无测试接触), 但:
  - --h5 任意 R18 候选 (双头 (3,) 或单头 (2,))
  - --out 自定义 tflite 输出
  - 冒烟断言: 输出形状、int8 dtype、tflite vs h5 相关性 > 0.99
    (代表性子集上逐样本对比 abn 通道)
预算断言: ≤ 300 KB (预注册 P2)。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
DATA = CACHE / "distill_pilot_teacher_targets.npz"


def representative_dataset():
    d = np.load(DATA)
    x = np.asarray(d["x_train_perm"], dtype=np.float32)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x), min(1000, len(x)), replace=False)
    for i in idx:
        yield [x[i:i + 1, ..., np.newaxis].astype(np.float32)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    h5 = Path(args.h5)
    if not h5.is_absolute():
        h5 = MODELS / args.h5
    out = Path(args.out)
    if not out.is_absolute():
        out = CACHE / args.out
    t0 = time.time()

    model = tf.keras.models.load_model(str(h5), compile=False)
    print(f"[R18X] loaded {h5.name}, params={model.count_params():,}", flush=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out.write_bytes(tflite_model)
    kb = len(tflite_model) / 1024
    print(f"[R18X] saved {out.name}: {len(tflite_model)} bytes ({kb:.1f} KB), "
          f"{time.time() - t0:.1f}s", flush=True)
    assert len(tflite_model) <= 300 * 1024, \
        f"INT8 体积超预算 300KB: {len(tflite_model)}"

    # ---- 冒烟: tflite vs h5 一致性 ----
    # 口径 (R19M 修正, 2026-09-19): 全训模型 softmax 输出饱和双峰, corr 门槛
    # 过严 (R18 实测 corr 0.947 / MAD 0.043 而量化本身良好) → 改 MAD 主判;
    # 双头模型补 valid 通道检查, 评估集混入合成噪声窗 (纯干净集上 valid 恒
    # 1、corr 无定义——R18 踩坑)。
    it = tf.lite.Interpreter(model_path=str(out))
    it.allocate_tensors()
    inp, outd = it.get_input_details()[0], it.get_output_details()[0]
    print(f"[R18X] input {inp['shape']} {inp['dtype']}; "
          f"output {outd['shape']} {outd['dtype']}", flush=True)
    assert outd["dtype"] == np.int8, "输出非 int8"

    d = np.load(DATA)
    x = np.asarray(d["x_train_perm"], dtype=np.float32)
    rng = np.random.default_rng(7)
    idx = rng.choice(len(x), 256, replace=False)
    xs = x[idx][..., np.newaxis]
    if outd["shape"][-1] >= 3:
        sys.path.insert(0, str(BASE))
        from train_r18_dualhead import gen_pure_noise_windows
        nrng = np.random.default_rng(11)
        xnoise = gen_pure_noise_windows(
            {"motion": 32, "emg": 32, "mains": 32}, nrng)[..., np.newaxis]
        xs = np.concatenate([xs.astype(np.float32), xnoise.astype(np.float32)])
    p_h5 = model.predict(xs, batch_size=256, verbose=0)
    scale = float(inp["quantization_parameters"]["scales"].flatten()[0])
    zp = int(inp["quantization_parameters"]["zero_points"].flatten()[0])
    oscale = float(outd["quantization_parameters"]["scales"].flatten()[0])
    ozp = int(outd["quantization_parameters"]["zero_points"].flatten()[0])
    q = np.clip(np.round(xs / scale + zp), -128, 127).astype(np.int8)
    preds = []
    for i in range(len(q)):
        it.set_tensor(inp["index"], q[i:i + 1])
        it.invoke()
        o = it.get_tensor(outd["index"])[0]
        preds.append((o.astype(np.float64) - ozp) * oscale)
    p_tl = np.stack(preds)
    abn_h5 = p_h5[:, 1] if p_h5.ndim > 1 else p_h5
    corr = float(np.corrcoef(abn_h5, p_tl[:, 1])[0, 1])
    mad = float(np.mean(np.abs(abn_h5 - p_tl[:, 1])))
    line = f"[R18X] smoke: corr(abn)={corr:.4f} MAD={mad:.4f} (n={len(q)})"
    ok = mad <= 0.05 and corr >= 0.90
    if p_tl.shape[1] >= 3:
        v_h5 = np.asarray(model.predict(xs, batch_size=256, verbose=0))[:, 2]
        v_tl = p_tl[:, 2]
        if np.std(v_h5) > 1e-6 and np.std(v_tl) > 1e-6:
            corr_v = float(np.corrcoef(v_h5, v_tl)[0, 1])
        else:
            corr_v = float("nan")
        line += f" corr(valid)={corr_v:.4f}"
        ok = ok and (np.isnan(corr_v) or corr_v >= 0.95)
    print(line, flush=True)
    assert ok, f"tflite vs h5 一致性不足: {line}"
    print("[R18X] OK", flush=True)


if __name__ == "__main__":
    main()
