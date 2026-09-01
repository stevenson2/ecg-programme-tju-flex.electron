#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_event_exp7c_sweep_high_causal.py — exp7c INT8 vs ECGFounder v3b QAT INT8 事件级 1-of-5
================================================================================
在同一 MIT+INCART 患者级测试集上，使用直接反量化 abnormal 概率（去二次 softmax），
比较当前 exp7c INT8 与 ECGFounder v3b QAT INT8 的：
  - 1-of-5 事件级 recall / precision / F1
  - FP/record
用于判断 v3b QAT 是否值得进入最终选型。
"""
import sys, json, time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import OpResolverType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR
from data.dataset import set_npz_suffix, load_mit_incart_merged
from data.patient_split import build_mit_patient_map, build_incart_patient_map, patient_level_split
from eval_aami_breakdown import recover_mit_symbols_per_record, recover_incart_symbols_per_record, align_symbols_to_npz
from eval_exp7c_policy_sweep import (
    reduce_mit_augmentation, evaluate_sequence_set,
    DEFAULT_GT_GAP, DEFAULT_ALERT_COOLDOWN,
)

# 扩展阈值扫描范围到 0.80
import eval_exp7c_policy_sweep as _pol
_pol.THETAS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
from eval_aami_matrix import add_channel_dim

OUT = Path(__file__).resolve().parent / "models" / "deploy_match" / "event_exp7c_sweep_high_causal.json"

MODELS = [
    ("exp7c_int8", MODELS_DIR / "ecg_model_exp7c_int8.tflite"),
]


def predict_direct(path, x):
    """批量 INT8 推理，直接取反量化后的 p(abnormal)（不二次 softmax）。"""
    interp = tf.lite.Interpreter(
        model_path=str(path),
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
        num_threads=1,
    )
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    sig = tuple(int(v) for v in in_d["shape_signature"])
    n_batch = min(512, len(x))
    interp.resize_tensor_input(in_d["index"], (n_batch, sig[1], sig[2]), strict=False)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale = float(in_d["quantization_parameters"]["scales"].flatten()[0])
    in_zp = int(in_d["quantization_parameters"]["zero_points"].flatten()[0])
    out_scale = float(out_d["quantization_parameters"]["scales"].flatten()[0])
    out_zp = int(out_d["quantization_parameters"]["zero_points"].flatten()[0])
    probs = np.zeros(len(x), dtype=np.float32)
    for start in range(0, len(x), n_batch):
        xb = x[start:start + n_batch]
        actual = len(xb)
        if actual < n_batch:
            xb = np.concatenate([xb, np.zeros((n_batch - actual, *x.shape[1:]), dtype=x.dtype)], axis=0)
        xq = np.clip(np.round(xb / in_scale) + in_zp, -128, 127)
        if in_d["dtype"] == np.int8:
            xq = xq.astype(np.int8)
        else:
            xq = xq.astype(np.uint8)
        interp.set_tensor(in_d["index"], xq)
        interp.invoke()
        q = interp.get_tensor(out_d["index"]).astype(np.float32)[:actual]
        p = (q - out_zp) * out_scale
        probs[start:start + actual] = p[:, 1]
        if (start + actual) % 20000 < n_batch:
            print(f"  inference {start+actual}/{len(x)}", flush=True)
    return probs


def main():
    t0 = time.time()
    set_npz_suffix("_deploy_causal")
    data = load_mit_incart_merged()
    beats, labels, rids = data["beats"], data["labels"], data["record_ids"]
    beats, labels, rids, kept_idx = reduce_mit_augmentation(beats, labels, rids)

    per_rec_syms = recover_mit_symbols_per_record()
    incart_dir = Path(__file__).resolve().parent / "data" / "raw" / "incart"
    per_rec_syms.update(recover_incart_symbols_per_record(incart_dir))
    sym_full, _ = align_symbols_to_npz(per_rec_syms, data["record_ids"], 6)
    if sym_full is None:
        raise RuntimeError("symbol alignment failed")
    symbols = sym_full[kept_idx]

    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat for rid, pat in build_incart_patient_map().items()})
    tr_m, va_m, te_m, pstats = patient_level_split(data["record_ids"], pmap)
    te_red = te_m[kept_idx]

    x = add_channel_dim(beats.astype(np.float32))
    y = np.asarray(labels).astype(np.int32)

    print(f"[EVENT] test records={len(np.unique(rids[te_red]))}, beats={int(te_red.sum())}", flush=True)
    results = {}
    for tag, path in MODELS:
        print(f"\n=== {tag} ===", flush=True)
        pp = predict_direct(path, x)
        rr, yy, ss = rids[te_red], y[te_red], symbols[te_red]
        rows = evaluate_sequence_set("test", rr, yy, pp[te_red], ss,
                                     DEFAULT_GT_GAP, DEFAULT_ALERT_COOLDOWN)
        results[tag] = rows
        for r in rows:
            print(f"  theta={r['theta']} policy={r['policy']} "
                  f"event_recall={r['event_recall']} event_precision={r['event_precision']} "
                  f"event_f1={r['event_f1']} fp_per_record={r['fp_per_record']} "
                  f"fp_per_1000={r['fp_per_1000_beats']}", flush=True)

    out = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "direct dequantized abnormal probability; no second softmax; MIT+INCART patient-level test",
        "models": [tag for tag, _ in MODELS],
        "results": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[EVENT] saved {OUT}")


if __name__ == "__main__":
    main()
