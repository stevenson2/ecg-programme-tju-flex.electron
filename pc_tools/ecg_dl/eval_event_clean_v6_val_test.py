#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_event_clean_v6_val_test.py — 无泄漏 v6 事件级验证/测试
================================================================================
在 MIT+INCART 因果部署链 reduced 序列上：
  1. 用 v6 QAT INT8 推理并缓存全量概率；
  2. 在患者级 validation split 上细扫 θ/cooldown，选择操作点；
  3. 在患者级 test split 上冻结报告；
  4. 记录混淆矩阵自洽性。
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
import eval_exp7c_policy_sweep as pol

OUT = Path(__file__).resolve().parent / "models" / "deploy_match" / "event_clean_v6_val_test.json"
CACHE = Path(__file__).resolve().parent / "models" / "deploy_match"
MODEL = MODELS_DIR / "ecg_model_exp7c_clean_v6_qat_int8.tflite"
THETAS = [0.70, 0.75, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90]
COOLDOWNS = [5, 6, 8, 10]


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
    tr_red, va_red, te_red = tr_m[kept_idx], va_m[kept_idx], te_m[kept_idx]
    print(f"[V6] val beats={int(va_red.sum())}, test beats={int(te_red.sum())}", flush=True)

    x = add_channel_locally(beats.astype(np.float32))
    probs = predict_direct(MODEL, x)
    np.save(CACHE / "clean_v6_causal_probs_full.npy", probs)

    # 验证集选参数
    pol.THETAS = THETAS
    pol.POLICIES = [(1, 5)]
    val_best = None
    val_rows = []
    rr_v, yy_v, ss_v = rids[va_red], labels[va_red], symbols[va_red]
    for cool in COOLDOWNS:
        rows = evaluate_sequence_set("validation", rr_v, yy_v, probs[va_red], ss_v,
                                     DEFAULT_GT_GAP, cool)
        for r in rows:
            val_rows.append({"cooldown": cool, **r})
            if val_best is None or r["event_f1"] > val_best["event_f1"]:
                val_best = {"cooldown": cool, **r}
    print("[V6] selected on validation:", json.dumps({
        "theta": val_best["theta"], "cooldown": val_best["cooldown"],
        "recall": val_best["event_recall"], "precision": val_best["event_precision"],
        "f1": val_best["event_f1"], "fp_per_record": val_best["fp_per_record"],
    }, ensure_ascii=False), flush=True)

    # 测试冻结
    rr_t, yy_t, ss_t = rids[te_red], labels[te_red], symbols[te_red]
    test_rows = evaluate_sequence_set("test", rr_t, yy_t, probs[te_red], ss_t,
                                      DEFAULT_GT_GAP, val_best["cooldown"])
    test_selected = [r for r in test_rows if r["theta"] == val_best["theta"]]
    print("[V6] test at selected:", json.dumps({
        "theta": val_best["theta"], "cooldown": val_best["cooldown"],
        "recall": test_selected[0]["event_recall"],
        "precision": test_selected[0]["event_precision"],
        "f1": test_selected[0]["event_f1"],
        "fp_per_record": test_selected[0]["fp_per_record"],
    }, ensure_ascii=False), flush=True)

    # 自洽断言：GT events >= matched, alert blocks = matched+FP
    for r in [val_best, test_selected[0]]:
        gt = r.get("gt_events")
        matched = r.get("matched_gt_events")
        pred = r.get("pred_alert_blocks")
        fp = r.get("false_alarm_blocks")
        if gt is not None and matched is not None and pred is not None and fp is not None:
            assert matched <= gt, f"matched > gt: {matched} > {gt}"
            assert pred == matched + fp, f"pred != matched+fp: {pred} != {matched}+{fp}"

    json.dump({
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": str(MODEL),
        "patient_stats": {k: pstats[k] for k in ["n_patients","n_train","n_val","n_test"]},
        "validation_grid": [{
            "theta": x["theta"], "cooldown": x["cooldown"],
            "recall": x["event_recall"], "precision": x["event_precision"],
            "f1": x["event_f1"], "fp_per_record": x["fp_per_record"],
        } for x in val_rows],
        "selected_on_validation": {
            "theta": val_best["theta"], "cooldown": val_best["cooldown"],
            "recall": val_best["event_recall"], "precision": val_best["event_precision"],
            "f1": val_best["event_f1"], "fp_per_record": val_best["fp_per_record"],
            "gt_events": val_best["gt_events"], "matched_gt_events": val_best["matched_gt_events"],
            "pred_alert_blocks": val_best["pred_alert_blocks"],
            "false_alarm_blocks": val_best["false_alarm_blocks"],
        },
        "test_frozen": test_selected[0],
        "elapsed_s": round(time.time() - t0, 1),
    }, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[V6] saved {OUT}")


def add_channel_locally(x):
    return x[..., np.newaxis]


if __name__ == "__main__":
    main()
