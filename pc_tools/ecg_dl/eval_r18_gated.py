#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_r18_gated.py — R18 门槛2 gated 口径评估 (预注册 §1-H2 判据)
================================================================================
对 eval_corpus_pc 同链窗流, 输出双头模型的三个量:
  raw_rate      = mean(conf_abn > 0.5)                    (v3-A 口径)
  gated_raw     = mean(conf_abn > 0.5 & conf_valid > 0.5) (部署口径: AI 窗
                 计数需 valid 通过; v3-A valid≡1 → gated=raw 可比)
  valid_mean    = mean(conf_valid)
组级判定 (预注册冻结阈值, h5 口径):
  pure_motion gated ≤ 0.30;  abnormal+noise@10dB gated 组均 ≥ 0.56;
  normal+noise gated 组均 ≤ 0.1827;  clean_normal gated ≤ 0.21。
单头模型 (输出 dim 2) 自动退化为 raw=gated (valid≡1)。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from eval_corpus_pc import corpus_windows

# 预注册冻结判据 (v3-A 锚点: 2026-09-18 复测 JSON)
CRITERIA = {
    "pure_motion_gated_max": 0.30,
    "abnormal_noise10_gated_min": 0.56,     # 0.660 - 0.10
    "normal_noise_gated_max": 0.1827,       # 0.1627 + 0.02
    "clean_normal_gated_max": 0.21,         # 0.16 + 0.05
}
ABN_NOISE10 = ("abnormal_baseline_wander_snr10", "abnormal_emg_snr10",
               "abnormal_motion_snr10", "abnormal_mains_snr10")
NORMAL_NOISE = tuple(f"normal_{t}_snr{s}" for t in
                     ("baseline_wander", "respiratory", "mains", "emg", "motion")
                     for s in (0, 10, 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="models/ 下的 h5 文件名")
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    import tensorflow as tf
    model = tf.keras.models.load_model(str(BASE / "models" / args.h5),
                                       compile=False)

    corpus = np.load(BASE / "corpus" / "corpus_500hz.npy")
    prov = json.loads((BASE / "corpus" / "replay_corpus_provenance.json")
                      .read_text(encoding="utf-8"))

    t0 = time.time()
    results = []
    for c in prov["cases"]:
        wins = corpus_windows(corpus[c["index"]])
        pred = model.predict(wins[..., np.newaxis], batch_size=512, verbose=0)
        if pred.ndim == 2 and pred.shape[-1] >= 3:
            conf, valid = pred[:, 1], pred[:, 2]
        else:
            conf, valid = pred[:, 1], np.ones(len(pred))
        r = {
            "name": c["name"], "params": c["params"], "windows": int(len(conf)),
            "raw_rate": round(float((conf > 0.5).mean()), 4),
            "gated_raw": round(float(((conf > 0.5) & (valid > 0.5)).mean()), 4),
            "valid_mean": round(float(valid.mean()), 4),
        }
        results.append(r)
        print("[%2d] %-32s raw=%.3f gated=%.3f valid=%.3f" % (
            c["index"], c["name"], r["raw_rate"], r["gated_raw"],
            r["valid_mean"]), flush=True)

    by = {r["name"]: r for r in results}
    groups = {
        "pure_motion": by["pure_motion"]["gated_raw"],
        "abnormal_noise10_mean": float(np.mean([by[n]["gated_raw"]
                                                for n in ABN_NOISE10])),
        "normal_noise_mean": float(np.mean([by[n]["gated_raw"]
                                            for n in NORMAL_NOISE])),
        "clean_normal": by["clean_normal"]["gated_raw"],
        "mains_snr20": by["normal_mains_snr20"]["gated_raw"],
    }
    verdict = {
        "pure_motion": groups["pure_motion"] <= CRITERIA["pure_motion_gated_max"],
        "abnormal_noise10": groups["abnormal_noise10_mean"]
        >= CRITERIA["abnormal_noise10_gated_min"],
        "normal_noise": groups["normal_noise_mean"]
        <= CRITERIA["normal_noise_gated_max"],
        "clean_normal": groups["clean_normal"] <= CRITERIA["clean_normal_gated_max"],
    }
    # 机制披露: 载波窗 valid 高 / 纯噪声窗 valid 低
    mech = {
        "clean_normal_valid": by["clean_normal"]["valid_mean"],
        "clean_abnormal_valid": by["clean_abnormal"]["valid_mean"],
        "pure_noise_valid_mean": float(np.mean(
            [by[n]["valid_mean"] for n in ("pure_motion", "pure_emg")])),
        "ludb_ptb_valid": by["ludb_normal"]["valid_mean"],
    }

    out = {
        "tool": "eval_r18_gated.py", "label": args.label, "h5": args.h5,
        "criteria": CRITERIA, "groups": {k: round(v, 4)
                                         for k, v in groups.items()},
        "verdict": verdict, "gate2_pass": bool(all(verdict.values())),
        "mechanism_disclosure": {k: round(v, 4) for k, v in mech.items()},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_s": round(time.time() - t0, 1),
        "results": results,
    }
    stem = "r18_gated_%s" % args.label
    (BASE / "corpus" / (stem + ".json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"groups": out["groups"], "verdict": verdict,
                      "gate2_pass": out["gate2_pass"], "mech": mech},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"-> corpus/{stem}.json", flush=True)


if __name__ == "__main__":
    main()
