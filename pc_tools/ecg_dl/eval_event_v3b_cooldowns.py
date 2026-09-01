#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_event_v3b_cooldowns.py — v3b θ=0.85 下扫描 alert_cooldown
================================================================================
在因果部署链 MIT+INCART 测试序列上，缓存 v3b QAT INT8 概率后，
固定 θ=0.85、gt_gap=5，扫描 alert_cooldown 3/4/5/6/7/8/10，
比较事件级 recall/precision/F1/FP。
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
from eval_exp7c_policy_sweep import reduce_mit_augmentation, evaluate_sequence_set, DEFAULT_GT_GAP
from eval_aami_matrix import add_channel_dim
import eval_exp7c_policy_sweep as pol

OUT = Path(__file__).resolve().parent / "models" / "deploy_match" / "event_v3b_cooldown.json"
MODEL = MODELS_DIR / "ecg_model_exp7c_ecgfounder_v3b_qat_int8.tflite"


def predict_direct(path, x):
    it = tf.lite.Interpreter(model_path=str(path),
                             experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
                             num_threads=1)
    it.allocate_tensors()
    in_d = it.get_input_details()[0]
    out_d = it.get_output_details()[0]
    sig = tuple(int(v) for v in in_d["shape_signature"])
    n_batch = min(512, len(x))
    it.resize_tensor_input(in_d["index"], (n_batch, sig[1], sig[2]), strict=False)
    it.allocate_tensors()
    in_d = it.get_input_details()[0]
    out_d = it.get_output_details()[0]
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
        it.set_tensor(in_d["index"], xq)
        it.invoke()
        q = it.get_tensor(out_d["index"]).astype(np.float32)[:actual]
        p = (q - out_zp) * out_scale
        probs[start:start + actual] = p[:, 1]
        if (start + actual) % 50000 < n_batch:
            print(f"  inference {start + actual}/{len(x)}", flush=True)
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
    symbols = sym_full[kept_idx]

    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat for rid, pat in build_incart_patient_map().items()})
    tr_m, va_m, te_m, pstats = patient_level_split(data["record_ids"], pmap)
    te_red = te_m[kept_idx]

    x = add_channel_dim(beats.astype(np.float32))
    probs = predict_direct(MODEL, x)
    np.save(Path(__file__).resolve().parent / "models" / "deploy_match" / "v3b_causal_probs_full.npy", probs)

    pp = probs[te_red]
    rr, yy, ss = rids[te_red], labels[te_red], symbols[te_red]
    pol.THETAS = [0.85]
    pol.POLICIES = [("1", "5")]  # ignore, evaluate_sequence_set uses module POLICIES? We'll set to [('1','5')] below

    results = []
    for cool in [3, 4, 5, 6, 7, 8, 10]:
        pol.POLICIES = [(1, 5)]  # 1-of-5
        rows = evaluate_sequence_set("test", rr, yy, pp, ss, DEFAULT_GT_GAP, cool)
        r = [x for x in rows if abs(x["theta"] - 0.85) < 1e-9 and x["policy"] == "1-of-5"]
        if r:
            row = r[0]
            results.append({
                "alert_cooldown": cool,
                "event_recall": row["event_recall"],
                "event_precision": row["event_precision"],
                "event_f1": row["event_f1"],
                "false_alarm_blocks": row["false_alarm_blocks"],
                "fp_per_record": row["fp_per_record"],
                "fp_per_1000_beats": row["fp_per_1000_beats"],
            })
            print(f"cooldown={cool}: recall={row['event_recall']:.4f} "
                  f"prec={row['event_precision']:.4f} f1={row['event_f1']:.4f} "
                  f"fp_rec={row['fp_per_record']:.4f}", flush=True)

    json.dump({"theta": 0.85, "gt_gap": DEFAULT_GT_GAP, "results": results,
               "elapsed_s": round(time.time() - t0, 1)}, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[COOLDOWN] saved {OUT}")


if __name__ == "__main__":
    main()
