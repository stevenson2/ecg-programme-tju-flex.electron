#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_r21_gated.py — R21 门槛2 + ptb_tp 组判据评估 (预注册 R21M_PREREG §3.2/§3.3)
================================================================================
对 eval_corpus_pc 同链窗流评估候选 (双头) 与同 seed B0 基线 (单头, gated≡raw):

门槛2 六项 (承 R19M 冻结值):
  pure_motion gated ≤0.30; abn10 组均 ≥0.56; normal_noise 组均 ≤0.1827;
  clean_normal ≤0.21; clean_abnormal gated ≥0.60; ludb gated ≥0.90。

ptb_tp 组 (W0 扩容, 7 case = 6 新 PTB test + ludb(s0010), 对同 seed B0 h5 配对):
  (a) 组均 gated ≥ 0.70 (绝对下限, 锚 v3-A h5 实测 0.7543−0.05);
  (b) 锚保真: 逐 case Δraw=cand.raw−b0.raw, 组均 ≥ −0.05 且单 case ≥ −0.15;
  (c) 门不压制: 逐 case Δgate=cand.gated−cand.raw ≥ −0.05 (全部 case)。

种子聚合 (≥2/3) 由调用侧 (汇总脚本/人工按 JSON) 执行, 本工具单模型出数。
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

# 预注册冻结判据 (R21M_PREREG §3.2/§3.3)
CRITERIA = {
    "pure_motion_gated_max": 0.30,
    "abnormal_noise10_gated_min": 0.56,
    "normal_noise_gated_max": 0.1827,
    "clean_normal_gated_max": 0.21,
    "clean_abnormal_gated_min": 0.60,
    "ludb_gated_min": 0.90,
    "ptb_tp_group_mean_min": 0.70,
    "ptb_tp_dRaw_group_mean_min": -0.05,
    "ptb_tp_dRaw_case_min": -0.15,
    "ptb_tp_gate_suppress_case_min": -0.05,
}
ABN_NOISE10 = ("abnormal_baseline_wander_snr10", "abnormal_emg_snr10",
               "abnormal_motion_snr10", "abnormal_mains_snr10")
NORMAL_NOISE = tuple(f"normal_{t}_snr{s}" for t in
                     ("baseline_wander", "respiratory", "mains", "emg", "motion")
                     for s in (0, 10, 20))


def eval_cases(model, corpus, cases_meta):
    """按 cases_meta (含 index/name) 评估, 返回逐 case dict。"""
    results = []
    for c in cases_meta:
        wins = corpus_windows(corpus[c["index"]])
        pred = model.predict(wins[..., np.newaxis], batch_size=512, verbose=0)
        if pred.ndim == 2 and pred.shape[-1] >= 3:
            conf, valid = pred[:, 1], pred[:, 2]
        else:
            conf, valid = pred[:, 1], np.ones(len(pred))
        r = {
            "name": c["name"], "windows": int(len(conf)),
            "raw_rate": round(float((conf > 0.5).mean()), 4),
            "gated_raw": round(float(((conf > 0.5) & (valid > 0.5)).mean()), 4),
            "valid_mean": round(float(valid.mean()), 4),
        }
        results.append(r)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="候选 h5 (models/ 下)")
    ap.add_argument("--b0", required=True, help="同 seed B0 基线 h5 (models/ 下)")
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    import tensorflow as tf
    cand = tf.keras.models.load_model(str(BASE / "models" / args.h5),
                                      compile=False)
    b0 = tf.keras.models.load_model(str(BASE / "models" / args.b0),
                                    compile=False)

    c1 = np.load(BASE / "corpus" / "corpus_500hz.npy")
    p1 = json.loads((BASE / "corpus" / "replay_corpus_provenance.json")
                    .read_text(encoding="utf-8"))
    ptb = np.load(BASE / "corpus" / "ptb_corpus_500hz.npy")
    pptb = json.loads((BASE / "corpus" / "ptb_corpus_provenance.json")
                      .read_text(encoding="utf-8"))

    t0 = time.time()
    print("== 候选 ==", flush=True)
    rc = eval_cases(cand, c1, p1["cases"])
    print("== B0 (同 seed 基线) ==", flush=True)
    rb0 = eval_cases(b0, c1, p1["cases"])
    for r in rc:
        print("[c1] %-32s raw=%.3f gated=%.3f valid=%.3f" % (
            r["name"], r["raw_rate"], r["gated_raw"], r["valid_mean"]),
            flush=True)

    by_c = {r["name"]: r for r in rc}
    by_b = {r["name"]: r for r in rb0}

    # ---- 门槛2 六项 ----
    groups = {
        "pure_motion": by_c["pure_motion"]["gated_raw"],
        "abnormal_noise10_mean": float(np.mean([by_c[n]["gated_raw"]
                                                for n in ABN_NOISE10])),
        "normal_noise_mean": float(np.mean([by_c[n]["gated_raw"]
                                            for n in NORMAL_NOISE])),
        "clean_normal": by_c["clean_normal"]["gated_raw"],
        "clean_abnormal": by_c["clean_abnormal"]["gated_raw"],
        "ludb": by_c["ludb_normal"]["gated_raw"],
    }
    verdict2 = {
        "pure_motion": groups["pure_motion"] <= CRITERIA["pure_motion_gated_max"],
        "abnormal_noise10": groups["abnormal_noise10_mean"]
        >= CRITERIA["abnormal_noise10_gated_min"],
        "normal_noise": groups["normal_noise_mean"]
        <= CRITERIA["normal_noise_gated_max"],
        "clean_normal": groups["clean_normal"] <= CRITERIA["clean_normal_gated_max"],
        "clean_abnormal": groups["clean_abnormal"]
        >= CRITERIA["clean_abnormal_gated_min"],
        "ludb": groups["ludb"] >= CRITERIA["ludb_gated_min"],
    }

    # ---- ptb_tp 组 (6 新 case + ludb) ----
    print("== PTB 组 (候选 + B0) ==", flush=True)
    ptb_meta = [{"index": c["index"], "name": c["name"]}
                for c in pptb["cases"]]
    ludb_idx = next(c["index"] for c in p1["cases"]
                    if c["name"] == "ludb_normal")
    ptb_meta.append({"index": ludb_idx, "name": "ludb_normal"})
    # ptb corpus 与 corpus1 分属两个 npy → 逐 case 选数组
    rpc = eval_cases(cand, ptb, ptb_meta[:-1]) + \
        eval_cases(cand, c1, [ptb_meta[-1]])
    rpb = eval_cases(b0, ptb, ptb_meta[:-1]) + \
        eval_cases(b0, c1, [ptb_meta[-1]])
    ptb_cases = []
    for rcc, rbb in zip(rpc, rpb):
        d_raw = round(rcc["raw_rate"] - rbb["raw_rate"], 4)
        d_gate = round(rcc["gated_raw"] - rcc["raw_rate"], 4)
        ptb_cases.append({"name": rcc["name"], "cand_raw": rcc["raw_rate"],
                          "cand_gated": rcc["gated_raw"],
                          "cand_valid": rcc["valid_mean"],
                          "b0_raw": rbb["raw_rate"],
                          "d_raw": d_raw, "d_gate": d_gate})
        print("[ptb] %-18s cand raw=%.3f gated=%.3f valid=%.3f | b0 raw=%.3f "
              "| Δraw=%+.3f Δgate=%+.3f" % (
                  rcc["name"], rcc["raw_rate"], rcc["gated_raw"],
                  rcc["valid_mean"], rbb["raw_rate"], d_raw, d_gate),
              flush=True)

    grp_mean = float(np.mean([x["cand_gated"] for x in ptb_cases]))
    d_raw_mean = float(np.mean([x["d_raw"] for x in ptb_cases]))
    d_raw_min = float(np.min([x["d_raw"] for x in ptb_cases]))
    gate_sup_min = float(np.min([x["d_gate"] for x in ptb_cases]))
    ptb_group = {
        "group_mean_gated": round(grp_mean, 4),
        "d_raw_mean": round(d_raw_mean, 4),
        "d_raw_min": round(d_raw_min, 4),
        "gate_suppress_min": round(gate_sup_min, 4),
    }
    verdict_ptb = {
        "a_group_mean": grp_mean >= CRITERIA["ptb_tp_group_mean_min"],
        "b_anchor": (d_raw_mean >= CRITERIA["ptb_tp_dRaw_group_mean_min"]
                     and d_raw_min >= CRITERIA["ptb_tp_dRaw_case_min"]),
        "c_no_suppress": gate_sup_min >= CRITERIA["ptb_tp_gate_suppress_case_min"],
    }

    out = {
        "tool": "eval_r21_gated.py", "label": args.label,
        "h5": args.h5, "b0": args.b0,
        "criteria": CRITERIA,
        "gate2": {"groups": {k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in groups.items()},
                  "verdict": verdict2, "pass": bool(all(verdict2.values()))},
        "ptb_tp_group": {"stats": ptb_group, "cases": ptb_cases,
                         "verdict": verdict_ptb,
                         "pass": bool(all(verdict_ptb.values()))},
        "mech": {
            "pure_noise_valid_mean": float(np.mean(
                [by_c[n]["valid_mean"] for n in ("pure_motion", "pure_emg")])),
            "clean_carrier_valid_mean": float(np.mean(
                [by_c[n]["valid_mean"] for n in
                 ("clean_normal", "clean_abnormal", "ludb_normal")])),
        },
        "results_corpus1": rc,
        "results_b0_corpus1": rb0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_s": round(time.time() - t0, 1),
    }
    out["gate_all_pass"] = bool(out["gate2"]["pass"] and out["ptb_tp_group"]["pass"])
    stem = "r21_gated_%s" % args.label
    (BASE / "corpus" / (stem + ".json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gate2": out["gate2"]["groups"], "v2": verdict2,
                      "ptb": ptb_group, "v_ptb": verdict_ptb,
                      "ALL_PASS": out["gate_all_pass"]},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"-> corpus/{stem}.json", flush=True)


if __name__ == "__main__":
    main()
