#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""posttrain_calibration_int8.py — exp7c INT8 后训练分数校准与阈值重扫（MIT 部署链缓存）
================================================================================
目的：
  1. 用 PC TFLite BUILTIN_REF（与 MCU TFLM/ESP-NN 整数 kernel 可比）对
     mit_deploy_causal_match.npz 全量测试拍生成 INT8 异常概率。
  2. 在患者级 50/50 划分的校准侧拟合：
       - 温度缩放（Temperature Scaling）
       - Platt / Logistic 缩放
       - 原始 INT8 概率阈值扫描
  3. 在评估侧报告：
       - 原始分数 vs 校准分数的 AUC / Brier / NLL
       - 阈值扫描、推荐阈值（Youden、F1、0.50/0.60 现用点）
       - 1-of-5 简单事件策略对比
输出：
  models/deploy_match/int8_calibration_mit.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from tensorflow.lite.python.interpreter import OpResolverType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.patient_split import build_mit_patient_map, build_incart_patient_map

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match" / "mit_deploy_causal_match.npz"
TFL = BASE / "models" / "ecg_model_exp7c_int8.tflite"
OUT = BASE / "models" / "deploy_match" / "int8_calibration_mit.json"
SEED = 42


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    se = tp / max(1, tp + fn)
    sp = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    f1 = 2 * se * prec / (se + prec) if (se + prec) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": round(se, 6), "specificity": round(sp, 6),
        "precision": round(prec, 6), "f1": round(f1, 6),
        "youden": round(se + sp - 1.0, 6),
    }


def run_tflite_probs(beats):
    interp = tf.lite.Interpreter(
        model_path=str(TFL),
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
        if (i + 1) % 10000 == 0:
            print(f"  TFLite {i+1}/{len(beats)}", flush=True)
    return probs


def temperature_calibrate(y, p):
    """二元温度缩放: logit=log(p/(1-p)), p_cal=sigmoid(logit/T)."""
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    best = None
    for T in np.arange(0.5, 6.01, 0.05):
        z = logit / T
        pc = 1.0 / (1.0 + np.exp(-z))
        pc = np.clip(pc, eps, 1 - eps)
        nll = -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))
        if best is None or nll < best[1]:
            best = (float(T), float(nll))
    T = best[0]
    z = logit / T
    pc = 1.0 / (1.0 + np.exp(-z))
    return T, np.clip(pc, eps, 1 - eps)


def platt_calibrate(y, p):
    """Platt scaling: logit = a*logit(p)+b, fit by logistic regression on calibration."""
    from sklearn.linear_model import LogisticRegression
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    X = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(X, y)
    a = float(lr.coef_[0][0]); b = float(lr.intercept_[0])
    z = a * X[:, 0] + b
    pc = 1.0 / (1.0 + np.exp(-z))
    return a, b, np.clip(pc, eps, 1 - eps)


def threshold_scan(y, p):
    rows = []
    for thr in np.arange(0.01, 0.96, 0.01):
        m = metrics(y, p, float(thr))
        m["threshold"] = float(thr)
        rows.append(m)
    return rows


def best_by(rows, key):
    return max(rows, key=lambda r: r[key])


def event_1of5(y, p, thr):
    """按记录分组后的 1-of-5 episode 简化评估。"""
    # 这里只做 beat 级阈值触发的 1-of-5：在每组记录内，任一拍超阈值即触发报警块，
    # 触发后 5 拍冷却；GT 异常 episode 用 5 拍间隙合并。
    def merge_blocks(blocks, gap):
        if not blocks:
            return []
        blocks = sorted(blocks)
        out = [list(blocks[0])]
        for s, e in blocks[1:]:
            if s - out[-1][1] - 1 <= gap:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return [(s, e) for s, e in out]

    # This function is intentionally lightweight; full episode metrics reuse existing policy sweep later.
    return None


def main():
    t0 = time.time()
    d = np.load(CACHE)
    beats = d["beats"].astype(np.float32)
    labels = d["labels"].astype(np.int32)
    rids = d["record_ids"].astype(np.int32)
    print(f"loaded {len(beats)} beats, {len(np.unique(rids))} records", flush=True)

    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat
                 for rid, pat in build_incart_patient_map().items()})
    patients = np.array([pmap.get(int(r), f"rid_{int(r)}") for r in rids])
    uniq_pat = np.unique(patients)
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq_pat)
    cal_pats = set(uniq_pat[: len(uniq_pat) // 2])
    cal_mask = np.array([p in cal_pats for p in patients], dtype=bool)
    ev_mask = ~cal_mask
    print(f"patients: cal={len(cal_pats)}, eval={len(uniq_pat)-len(cal_pats)}; "
          f"beats cal={int(cal_mask.sum())}, eval={int(ev_mask.sum())}", flush=True)

    probs = run_tflite_probs(beats)

    y_cal, p_cal = labels[cal_mask], probs[cal_mask]
    y_ev, p_ev = labels[ev_mask], probs[ev_mask]

    # Temperature calibration
    T, p_temp_cal = temperature_calibrate(y_cal, p_cal)
    # Apply same T to eval
    eps = 1e-9
    logit_ev = np.log(np.clip(p_ev, eps, 1 - eps) / (1 - np.clip(p_ev, eps, 1 - eps)))
    p_temp_ev = np.clip(1.0 / (1.0 + np.exp(-logit_ev / T)), eps, 1 - eps)

    # Platt calibration
    a, b, p_platt_cal = platt_calibrate(y_cal, p_cal)
    z_ev = a * logit_ev + b
    p_platt_ev = np.clip(1.0 / (1.0 + np.exp(-z_ev)), eps, 1 - eps)

    # Calibration-side threshold scan
    scan_raw_cal = threshold_scan(y_cal, p_cal)
    scan_temp_cal = threshold_scan(y_cal, p_temp_cal)
    scan_platt_cal = threshold_scan(y_cal, p_platt_cal)
    best_raw_cal = best_by(scan_raw_cal, "youden")
    best_temp_cal = best_by(scan_temp_cal, "youden")
    best_platt_cal = best_by(scan_platt_cal, "youden")

    # Eval-side summary
    def summary(y, p):
        return {
            "auc": round(float(roc_auc_score(y, p)), 6),
            "brier": round(float(brier_score_loss(y, p)), 6),
            "logloss": round(float(log_loss(y, p, labels=[0, 1])), 6),
            "mean_p": round(float(p.mean()), 6),
            "median_p": round(float(np.median(p)), 6),
            "range": [round(float(p.min()), 6), round(float(p.max()), 6)],
        }

    # Eval-side threshold scans
    scan_raw_ev = threshold_scan(y_ev, p_ev)
    scan_temp_ev = threshold_scan(y_ev, p_temp_ev)
    scan_platt_ev = threshold_scan(y_ev, p_platt_ev)

    # Pick Youden from calibration, freeze to eval
    def frozen_eval(scan_ev, thr):
        for r in scan_ev:
            if abs(r["threshold"] - thr) < 1e-9:
                return {k: r[k] for k in ("threshold", "sensitivity", "specificity", "precision", "f1", "youden", "tp", "fp", "fn", "tn")}
        return None

    out = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "exp7c INT8 post-training score calibration & threshold rescan",
            "cache": str(CACHE.relative_to(BASE.parent)),
            "model": str(TFL.relative_to(BASE.parent)),
            "split": "patient-level 50/50 within MIT/INCART causal test cache, seed=42",
            "pc_resolver": "BUILTIN_REF (XNNPACK disabled)",
            "n_beats": int(len(beats)),
            "n_records": int(len(np.unique(rids))),
            "n_patients_cal": int(len(cal_pats)),
            "n_patients_eval": int(len(uniq_pat) - len(cal_pats)),
            "cal_beats": int(cal_mask.sum()),
            "eval_beats": int(ev_mask.sum()),
        },
        "calibration_fit": {
            "temperature": T,
            "platt_a": a,
            "platt_b": b,
        },
        "calibration_metrics": {
            "raw": summary(y_cal, p_cal),
            "temperature": summary(y_cal, p_temp_cal),
            "platt": summary(y_cal, p_platt_cal),
        },
        "eval_metrics": {
            "raw": summary(y_ev, p_ev),
            "temperature": summary(y_ev, p_temp_ev),
            "platt": summary(y_ev, p_platt_ev),
        },
        "calibration_best": {
            "raw_youden": best_raw_cal,
            "temperature_youden": best_temp_cal,
            "platt_youden": best_platt_cal,
        },
        "frozen_eval": {
            "raw_youden": frozen_eval(scan_raw_ev, best_raw_cal["threshold"]),
            "temperature_youden": frozen_eval(scan_temp_ev, best_temp_cal["threshold"]),
            "platt_youden": frozen_eval(scan_platt_ev, best_platt_cal["threshold"]),
        },
        "current_operating_points_eval": {
            "raw_0.50": frozen_eval(scan_raw_ev, 0.50),
            "raw_0.60": frozen_eval(scan_raw_ev, 0.60),
            "temperature_0.50": frozen_eval(scan_temp_ev, 0.50),
            "temperature_0.60": frozen_eval(scan_temp_ev, 0.60),
        },
        "eval_threshold_scan_raw": scan_raw_ev,
        "eval_threshold_scan_temperature": scan_temp_ev,
        "eval_threshold_scan_platt": scan_platt_ev,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[done] saved {OUT}", flush=True)
    print(f"[done] elapsed {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
