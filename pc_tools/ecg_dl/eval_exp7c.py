#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_exp7c.py — exp7c (真实数据微调) 修正后因果链口径 MIT/PTB AUC 防回归评估
================================================================================
与 eval_exp7b.py 同缓存测试拍 (mit/ptb_deploy_causal_match.npz), 同口径评估
exp7c + exp7b 对照。锚点 (不可篡改): exp7b MIT 0.8768 / PTB (JSON 内), 
exp6-SGD 因果链 MIT 0.9090 / PTB 0.7621, D0 上限 MIT 0.8942 / PTB 0.8232。
产出: models/deploy_match/retrain_exp7c_eval.json
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_deploy_match import compute_mit_domain_test_records, compute_ptb_domain_test_records  # noqa: F401 (导入补丁)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
CACHE_DIR = MODEL_DIR / "deploy_match"
MIT_CACHE = CACHE_DIR / "mit_deploy_causal_match.npz"
PTB_CACHE = CACHE_DIR / "ptb_deploy_causal_match.npz"
OUT_JSON = CACHE_DIR / "retrain_exp7c_eval.json"
THRESHOLDS = [0.35, 0.5, 0.6, 0.65, 0.8]

EVAL_MODELS = [
    ("exp7c", "best_resnet_large_exp7c.h5"),
    ("exp7b", "best_resnet_large_exp7b.h5"),
]

ANCHORS = {
    "mit": {"exp7b": 0.8768149164122789, "exp6_sgd_causal": 0.9090, "d0": 0.8942},
    "ptb": {"exp7b": None, "exp6_sgd_causal": 0.7621, "d0": 0.8232},
}


def main():
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

    d = np.load(MIT_CACHE)
    mit_beats, mit_labels = d["beats"], d["labels"]
    d = np.load(PTB_CACHE)
    ptb_beats, ptb_labels = d["beats"], d["labels"]
    print(f"[test] MIT {len(mit_beats)} 拍 (abn={int((mit_labels==1).sum())}), "
          f"PTB {len(ptb_beats)} 拍 (abn={int((ptb_labels==1).sum())})")

    results = {}
    for name, fname in EVAL_MODELS:
        path = MODEL_DIR / fname
        if not path.exists():
            print(f"[skip] {fname} missing")
            continue
        print(f"[eval] {name} ...")
        model = tf.keras.models.load_model(str(path), compile=False)
        results[name] = {}
        for dom, (beats, labels) in [("mit", (mit_beats, mit_labels)),
                                      ("ptb", (ptb_beats, ptb_labels))]:
            prob = model.predict(beats.astype(np.float32)[..., np.newaxis],
                                 batch_size=512, verbose=0)[:, 1]
            auc = float(roc_auc_score(labels, prob))
            r = {"auc": auc, "n_beats": int(len(labels)),
                 "n_abnormal": int((labels == 1).sum())}
            for thr in THRESHOLDS:
                pred = (prob >= thr).astype(int)
                p, rec, f1, _ = precision_recall_fscore_support(
                    labels, pred, average="binary", zero_division=0)
                r[f"thr_{thr}"] = {"recall": float(rec), "precision": float(p),
                                   "f1": float(f1)}
            results[name][dom] = r
            print(f"  {name} {dom}: AUC={auc:.4f} (n={len(labels)}, abn={int((labels==1).sum())})")
        del model
        tf.keras.backend.clear_session()

    # exp7b PTB 锚点从本评估自身读取 (同缓存同口径, 自洽)
    ANCHORS["ptb"]["exp7b"] = results.get("exp7b", {}).get("ptb", {}).get("auc")

    verdicts = {}
    if "exp7c" in results and "exp7b" in results:
        for dom in ["mit", "ptb"]:
            ec = results["exp7c"][dom]["auc"]
            eb = results["exp7b"][dom]["auc"]
            verdicts[dom] = {
                "exp7c_auc": ec, "exp7b_auc": eb, "delta_vs_exp7b": ec - eb,
                "vs_exp6_sgd_causal": ec - ANCHORS[dom]["exp6_sgd_causal"],
                "vs_d0": ec - ANCHORS[dom]["d0"],
                "pass": "no-significant-regression" if ec >= eb - 0.02 else "REGRESSED",
            }
            print(f"[verdict] {dom}: exp7c {ec:.4f} vs exp7b {eb:.4f} "
                  f"(Δ={ec-eb:+.4f}) {verdicts[dom]['pass']}")

    out = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "purpose": "exp7c 真实数据微调后防回归评估 (同 exp7b 缓存测试拍同口径)",
            "chain": "corrected_deployment_chain (D3 + causal HP 0.5Hz @250Hz)",
            "models": ["exp7c", "exp7b"],
            "thresholds": THRESHOLDS,
            "anchors": ANCHORS,
        },
        "results": results,
        "verdicts": verdicts,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[done] {OUT_JSON.name}")


if __name__ == "__main__":
    main()
