#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_corpus_pc.py — PC 侧噪声语料评估 (M3 门槛2 / M4 曲面, TH §106/§107)
================================================================================
对 corpus_500hz.npy 逐 case 复刻板上 AI 输入链并推理:
  500Hz 语料 → comb×2 + HP0.05/LP40 (因果) → 2:1 抽取 → 因果 HP0.5@250
  (ai_hp_coeffs_fs250.txt 修正系数, DF2T 流式与固件一致) → 1s 窗逐窗 z-score
  → 模型 (Keras h5 或 TFLite) → conf 序列。
指标与 board_corpus_eval.py 同 schema (raw_rate/conf_median/conf_mean),
可直接对照板上结果 (PC↔板一致性由 R16 已验证过, 本脚本作 M3 快速门槛 +
M4 全网格扩展用; 板上跑 30 case 子集)。

泄漏纪律: 语料载波 = MIT-100/106 (train 患者, 见 provenance 披露);
结论口径 = 同载波噪声敏感性自对照, 不作泛化证据。无任何训练侧取数。
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from noise_lib import chain_shape   # comb×2 + HP0.05/LP40 + 抽取 (500→250)

# 因果 HP 0.5Hz @250Hz (ai_hp_coeffs_fs250.txt, DF2T)
AI_HP_B0 = 0.99115359510166301
AI_HP_B1 = -1.982307190203326
AI_HP_B2 = 0.99115359510166301
AI_HP_A1 = -1.9822289297925284
AI_HP_A2 = 0.98238545061412508


def ai_hp_250(x):
    """DF2T 流式二阶高通 (与固件 ecg_ai.cpp biquad 同构)。"""
    w1 = w2 = 0.0
    out = np.empty_like(x)
    for i, v in enumerate(x):
        w = float(v) - AI_HP_A1 * w1 - AI_HP_A2 * w2
        out[i] = AI_HP_B0 * w + AI_HP_B1 * w1 + AI_HP_B2 * w2
        w2, w1 = w1, w
    return out


def zscore_window(w):
    mu = w.mean()
    sd = w.std()
    if sd < 1e-6:
        return np.zeros_like(w)
    return (w - mu) / sd


def corpus_windows(sig500, settle_s=5.0):
    """复刻板上 AI 窗流: 链 → HP → 1s 窗 z-score; 丢弃前 settle_s 秒。"""
    s250 = chain_shape(np.asarray(sig500, dtype=np.float64))
    s250 = ai_hp_250(s250)
    win = 250
    n_settle = int(settle_s * 250)
    out = []
    for i in range(n_settle, len(s250) - win + 1, win):
        out.append(zscore_window(s250[i:i + win]))
    return np.asarray(out, dtype=np.float32)


def predict_h5(model, windows):
    return model.predict(windows[..., np.newaxis], batch_size=512, verbose=0)[:, 1]


def predict_tflite(tflite_path, windows):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    scale, zp = inp["quantization"]
    confs = []
    for w in windows:
        if inp["dtype"] == np.int8:
            q = np.clip(np.round(w / scale + zp), -128, 127).astype(np.int8)
            interp.set_tensor(inp["index"], q[None, ..., None])
        else:
            interp.set_tensor(inp["index"], w[None, ..., None].astype(np.float32))
        interp.invoke()
        o = interp.get_tensor(out["index"])[0]
        o_scale, o_zp = out["quantization"]
        # 注意: int8 输出与 zp 都可能是 numpy 整型, 直接相减会 int8 回绕
        # (127-(-128)=255 -> -1), 必须先转 float (2026-09-12 踩坑)
        confs.append((float(o[1]) - float(o_zp)) * float(o_scale))
    return np.asarray(confs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", default="", help="Keras h5 模型路径 (与 --tflite 二选一)")
    ap.add_argument("--tflite", default="", help="TFLite 模型路径 (INT8 部署口径)")
    ap.add_argument("--label", default="pc_model")
    ap.add_argument("--corpus", default=str(BASE / "corpus" / "corpus_500hz.npy"))
    ap.add_argument("--provenance",
                    default=str(BASE / "corpus" / "replay_corpus_provenance.json"))
    ap.add_argument("--settle-s", type=float, default=5.0)
    args = ap.parse_args()

    if bool(args.h5) == bool(args.tflite):
        print("!! --h5 与 --tflite 恰选其一")
        return 1

    corpus = np.load(args.corpus)
    prov = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    cases = prov["cases"]
    assert len(cases) == len(corpus), (len(cases), len(corpus))

    if args.h5:
        import tensorflow as tf
        model = tf.keras.models.load_model(args.h5, compile=False)
        predict = lambda w: predict_h5(model, w)
    else:
        predict = lambda w: predict_tflite(args.tflite, w)

    results = []
    for c in cases:
        wins = corpus_windows(corpus[c["index"]], args.settle_s)
        confs = predict(wins)
        r = {
            "index": c["index"], "segment": c["segment"], "name": c["name"],
            "params": c["params"], "windows": int(len(confs)),
            "raw_rate": round(float((confs > 0.5).mean()), 4),
            "conf_median": round(float(np.median(confs)), 4),
            "conf_mean": round(float(confs.mean()), 4),
        }
        results.append(r)
        print("[%2d] seg%-2d %-32s raw=%.3f conf=%.3f n=%d" % (
            c["index"], c["segment"], c["name"], r["raw_rate"],
            r["conf_median"], r["windows"]), flush=True)

    out = {
        "tool": "eval_corpus_pc.py",
        "model_label": args.label,
        "model_path": args.h5 or args.tflite,
        "settle_s": args.settle_s,
        "chain": "comb×2 + HP0.05/LP40 + 2:1 + causal HP0.5@250 (ai_hp_coeffs_fs250)",
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    stem = "corpus_pc_%s_%s" % (args.label, __import__("time").strftime("%Y%m%d_%H%M"))
    (BASE / "corpus").mkdir(exist_ok=True)
    (BASE / "corpus" / (stem + ".json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-> corpus/%s.json" % stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
