#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_r21_ptb_corpus.py — W0 基线补测: 模型在 PTB 扩容语料上的 raw/det 数字
================================================================================
对 build_r21_ptb_corpus.py 生成的 6 个新 case (seg41..46) + corpus1 的
ludb_normal (PTB test s0010, 既有 TP 硬线) 计算:
  raw_rate / gated_raw / valid_mean (单头模型 valid≡1, gated=raw)
并给出 ptb_tp_group (7 case 含 s0010) 的组均/最小值 —— 判据数值在
R21 预注册冻结后由 eval_r21_gated.py 判定; 本脚本只做基线披露
(v3-A h5 + 板上 INT8 双口径)。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from eval_corpus_pc import corpus_windows, predict_tflite


def predict_h5_full(model, windows):
    """返回完整输出 (N,C): C=2 单头 / C=3 双头。"""
    return model.predict(windows[..., np.newaxis], batch_size=512, verbose=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="", help="models/ 下的 h5 文件名")
    ap.add_argument("--tflite", default="", help="tflite 路径 (单头 INT8)")
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    assert bool(args.h5) != bool(args.tflite), "--h5/--tflite 恰选其一"

    corpus = np.load(BASE / "corpus" / "ptb_corpus_500hz.npy")
    prov = json.loads((BASE / "corpus" / "ptb_corpus_provenance.json")
                      .read_text(encoding="utf-8"))
    c1 = np.load(BASE / "corpus" / "corpus_500hz.npy")
    p1 = json.loads((BASE / "corpus" / "replay_corpus_provenance.json")
                    .read_text(encoding="utf-8"))
    ludb_idx = next(c["index"] for c in p1["cases"] if c["name"] == "ludb_normal")

    if args.h5:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(BASE / "models" / args.h5),
                                           compile=False)

        def run(wins):
            pred = predict_h5_full(model, wins)
            if pred.ndim == 2 and pred.shape[-1] >= 3:
                return pred[:, 1], pred[:, 2]
            return pred[:, 1], np.ones(len(pred))
    else:
        tl = Path(args.tflite)
        if not tl.is_absolute():
            tl = BASE / tl
        # 单头 INT8 (v3-A 部署口径): 复用 eval_corpus_pc.predict_tflite
        def run(wins):
            conf = predict_tflite(tl, wins)
            return conf, np.ones(len(conf))

    t0 = time.time()
    results = []
    for c in prov["cases"]:
        conf, valid = run(corpus_windows(corpus[c["index"]]))
        r = {
            "name": c["name"], "segment": c["segment"],
            "diagnosis": c["params"]["diagnosis_tag"],
            "carrier": c["params"]["carrier"],
            "windows": int(len(conf)),
            "raw_rate": round(float((conf > 0.5).mean()), 4),
            "gated_raw": round(float(((conf > 0.5) & (valid > 0.5)).mean()), 4),
            "valid_mean": round(float(valid.mean()), 4),
            "conf_mean": round(float(conf.mean()), 4),
        }
        results.append(r)
        print("[seg%d] %-18s raw=%.3f conf=%.3f valid=%.3f (%s)" % (
            r["segment"], r["name"], r["raw_rate"], r["conf_mean"],
            r["valid_mean"], r["diagnosis"]), flush=True)

    # ludb_normal (s0010, 既有 TP 硬线) 同链复测, 入组
    conf, valid = run(corpus_windows(c1[ludb_idx]))
    r = {
        "name": "ludb_normal", "segment": ludb_idx + 3,
        "diagnosis": "MI (s0010, existing hard line)",
        "carrier": "PTB_patient001/s0010_re",
        "windows": int(len(conf)),
        "raw_rate": round(float((conf > 0.5).mean()), 4),
        "gated_raw": round(float(((conf > 0.5) & (valid > 0.5)).mean()), 4),
        "valid_mean": round(float(valid.mean()), 4),
        "conf_mean": round(float(conf.mean()), 4),
    }
    results.append(r)
    print("[seg%d] %-18s raw=%.3f conf=%.3f valid=%.3f (s0010 existing)" % (
        r["segment"], r["name"], r["raw_rate"], r["conf_mean"],
        r["valid_mean"]), flush=True)

    gated = [x["gated_raw"] for x in results]
    group = {
        "ptb_tp_group_mean": round(float(np.mean(gated)), 4),
        "ptb_tp_group_min": round(float(np.min(gated)), 4),
        "ptb_tp_group_n": len(gated),
    }
    print(json.dumps(group, ensure_ascii=False), flush=True)

    out = {
        "tool": "eval_r21_ptb_corpus.py", "label": args.label,
        "h5": args.h5 or "", "tflite": args.tflite or "",
        "group": group, "results": results,
        "note": ("W0 基线披露 (判据冻结前的测量); 单头 v3-A gated≡raw; "
                 "组含 6 新 case + ludb_normal(s0010)"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_s": round(time.time() - t0, 1),
    }
    stem = "r21_ptb_corpus_%s" % args.label
    (BASE / "corpus" / (stem + ".json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> corpus/{stem}.json", flush=True)


if __name__ == "__main__":
    main()
