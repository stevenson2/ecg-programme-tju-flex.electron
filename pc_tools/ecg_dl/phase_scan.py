#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phase_scan.py — D0: 部署流 FP 机制判别: 窗相位 (R 对齐) 依赖性实验 (TH §109)
================================================================================
假设: 训练拍窗全部与标注 R 波对齐, 而连续流 (板上回放/部署) 每秒盲切窗,
大多数窗不与 R 对齐 → 模型在非对齐相位大面积误报 ("相位失配"), 与数据库
域无关 (或复合)。
实验: 对语料 case (链后 250Hz) 的同一段信号做 0..249 采样平移, 每个相位
切一个 250 点窗 → 推理 → conf(相位) 曲线。若 FP 集中在特定相位带 → 实锤。
对照: clean_normal (MIT-100, train 患者, 连续流 FP 低) 与 ptb001_normal
(MIT... 即语料 ludb_normal, PTB test, FP 0.93)。
输出: corpus/phase_scan.json + research/m4_phase_scan.png (不改历史章节,
结论写入 §109)。
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import eval_corpus_pc as E


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", default=str(
        BASE / "models" / "deploy_match" / "ecg_model_v3a_int8.tflite"))
    args = ap.parse_args()

    corpus = np.load(BASE / "corpus" / "corpus_500hz.npy")
    prov = json.loads((BASE / "corpus" / "replay_corpus_provenance.json")
                      .read_text(encoding="utf-8"))
    names = [c["name"] for c in prov["cases"]]
    i_norm = names.index("clean_normal")
    i_ptb = names.index("ludb_normal")   # 实为 PTB test 患者 s0010_re (D0 改判)

    interp = E.predict_tflite  # not used directly; need interpreter handle
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=args.tflite)
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

    results = {}
    for tag, ci in (("clean_normal_mit100", i_norm), ("ptb_test_s0010", i_ptb)):
        sig250 = E.chain_shape(np.asarray(corpus[ci], dtype=np.float64))
        sig250 = E.ai_hp_250(sig250)
        # 取稳态中部 6s 段, 对 250 个相位各切一个窗 (窗内 1s)
        seg = sig250[1250: 1250 + 250 + 250]   # 2s → 平移 0..249 各出一窗
        confs = []
        for ph in range(250):
            w = E.zscore_window(seg[ph: ph + 250])
            confs.append(conf_one(w.astype(np.float32)))
        confs = np.asarray(confs)
        fp = float((confs > 0.5).mean())
        # 相位带分解: 25 个 10-采样 bin 的 FP 率
        bins = [(confs[k * 10:(k + 1) * 10] > 0.5).mean() for k in range(25)]
        results[tag] = {
            "n_phases": 250, "fp_rate": round(fp, 4),
            "conf_min": round(float(confs.min()), 4),
            "conf_max": round(float(confs.max()), 4),
            "conf_median": round(float(np.median(confs)), 4),
            "fp_by_bin_of10": [round(float(b), 2) for b in bins],
            "conf_curve": [round(float(c), 4) for c in confs],
        }
        print(f"{tag}: FP={fp:.3f} conf[{confs.min():.3f},{confs.max():.3f}]",
              flush=True)

    (BASE / "corpus" / "phase_scan.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 4.2))
        for tag, st in results.items():
            ax.plot(st["conf_curve"], label="%s (FP %.2f)" % (tag, st["fp_rate"]))
        ax.axhline(0.5, ls=":", c="k", lw=0.8)
        ax.set_xlabel("window phase (samples @250Hz, 0..249)")
        ax.set_ylabel("v3-A INT8 abnormal confidence")
        ax.set_title("D0 phase scan: same 1s segment, 250 window phases")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_png = REPO_DIR = BASE.parent.parent.parent / "research" / "m4_phase_scan.png"
        fig.savefig(out_png, dpi=140)
        print("->", out_png)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
