#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phase_eval.py — 相位稠密评估协议 (Round-E0, TH §111)
================================================================================
背景 (§109): 训练/评估全部建立在 R 对齐拍窗上, 而部署流是任意相位盲切窗。
phase_scan 单例实验显示: 正常流 FP 集中在窄相位带 (7.6%), 异常流在特定相位
conf 跌至 0.27。本协议把该扫描升级为全语料标准评估:
  每 case 取 M 个互不重叠的 2s 段 (链后 250Hz) → 每段扫 250 个窗相位 →
  - FP 类指标 (正常/纯噪声 case): 稠密相位窗 FP 率 (对比 1s 步进扫的稀疏估计)
  - TP 类指标 (异常 case): 每段相位中位 conf (检出水平) + 相位稳定性
    = min 段内 (中位 conf) 与 conf<0.5 相位占比 (漏报相位暴露度)
v3-A 基线入 corpus/phase_eval_<label>.json; v3-P (相位抖动增广) 的门槛②
即对比本协议输出。
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import eval_corpus_pc as E


def build_interpreter(tflite_path):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=str(tflite_path))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    scale, zp = inp["quantization"]
    oscale, ozp = out["quantization"]

    def conf_one(w):
        q = np.clip(np.round(w / scale + zp), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], q.reshape(1, 250, 1))
        it.invoke()
        o = it.get_tensor(out["index"])[0]
        return (float(o[1]) - float(ozp)) * float(oscale)
    return conf_one


def eval_case(conf_one, sig500, n_segments=5, settle_s=5.0):
    s250 = E.ai_hp_250(E.chain_shape(np.asarray(sig500, dtype=np.float64)))
    win = 250
    start = int(settle_s * 250)
    usable = len(s250) - start - win
    seg_len = 2 * win                      # 2s 段 → 250 相位 × 窗
    results = []
    for m in range(n_segments):
        seg0 = start + int(m * usable / n_segments)
        seg = s250[seg0: seg0 + seg_len + win]
        if len(seg) < seg_len + win:
            break
        confs = np.asarray([
            conf_one(E.zscore_window(seg[ph: ph + win]).astype(np.float32))
            for ph in range(250)])
        results.append({
            "seg": m,
            "fp_or_det_rate": round(float((confs > 0.5).mean()), 4),
            "conf_median": round(float(np.median(confs)), 4),
            "conf_p10": round(float(np.percentile(confs, 10)), 4),
            "conf_min": round(float(confs.min()), 4),
            "frac_conf_lt_05": round(float((confs < 0.5).mean()), 4),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", default=str(
        BASE / "models" / "deploy_match" / "ecg_model_v3a_int8.tflite"))
    ap.add_argument("--label", default="v3a_int8")
    ap.add_argument("--segments", type=int, default=5)
    args = ap.parse_args()

    corpus = np.load(BASE / "corpus" / "corpus_500hz.npy")
    prov = json.loads((BASE / "corpus" / "replay_corpus_provenance.json")
                      .read_text(encoding="utf-8"))
    conf_one = build_interpreter(args.tflite)

    t0 = time.time()
    out_cases = []
    for c in prov["cases"]:
        segs = eval_case(conf_one, corpus[c["index"]], args.segments)
        det_rates = [s["fp_or_det_rate"] for s in segs]
        medians = [s["conf_median"] for s in segs]
        out_cases.append({
            "index": c["index"], "segment": c["segment"], "name": c["name"],
            "params": c["params"], "segments": segs,
            "dense_rate_mean": round(statistics.mean(det_rates), 4),
            "conf_median_min_across_segs": round(min(medians), 4),
        })
        print("[%2d] %-32s dense_rate=%.3f conf_med_min=%.3f" % (
            c["index"], c["name"],
            out_cases[-1]["dense_rate_mean"],
            out_cases[-1]["conf_median_min_across_segs"]), flush=True)

    out = {
        "tool": "phase_eval.py", "label": args.label,
        "protocol": ("每 case 5 个互不重叠 2s 段 × 250 相位稠密窗; "
                     "dense_rate_mean = 全相位窗 conf>0.5 占比 (对比 1s 步进稀疏扫)"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_s": round(time.time() - t0, 1),
        "cases": out_cases,
    }
    (BASE / "corpus" / ("phase_eval_%s.json" % args.label)).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-> corpus/phase_eval_%s.json" % args.label)


if __name__ == "__main__":
    main()
