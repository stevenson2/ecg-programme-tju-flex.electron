#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_clean_baseline_v5.py — v5 SVDB 扩训模型的患者级测试集评估 (TH §104)
================================================================================
与 §99/§100/§103 完全同口径: θ=0.5、1-of-5、cooldown=5、同一测试缓存 npz。
输出: models/deploy_match/clean_baseline_eval_v5.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_clean_test import make_predictor, beat_metrics, event_metrics

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
MIT_NPZ = CACHE / "mit_deploy_causal_match.npz"
PTB_NPZ = CACHE / "ptb_deploy_causal_match.npz"
MODEL = MODELS / "best_resnet_large_clean_baseline_v5.h5"
OUT = CACHE / "clean_baseline_eval_v5.json"


def main():
    t0 = time.time()
    if not MODEL.exists():
        raise SystemExit(f"model missing: {MODEL}")
    dmi, dpt = np.load(MIT_NPZ), np.load(PTB_NPZ)
    mi_beats = np.asarray(dmi["beats"], dtype=np.float32)
    mi_labels = np.asarray(dmi["labels"]).astype(np.int32)
    mi_rids = np.asarray(dmi["record_ids"])
    pt_beats = np.asarray(dpt["beats"], dtype=np.float32)
    pt_labels = np.asarray(dpt["labels"]).astype(np.int32)
    pt_rids = np.asarray(dpt["record_ids"])

    predict = make_predictor("h5", MODEL)
    p_mi, p_pt = predict(mi_beats), predict(pt_beats)

    results = {}
    for domain, labels, probs, rids in (
            ("mit_incart", mi_labels, p_mi, mi_rids),
            ("ptb", pt_labels, p_pt, pt_rids)):
        beat = beat_metrics(labels, probs)
        evt = event_metrics(rids, labels, probs, f"{domain}_test")
        assert beat["tp"] + beat["fp"] + beat["tn"] + beat["fn"] == beat["n_beats"]
        if evt.get("alert_blocks") is not None and evt.get("gt_events") is not None:
            assert evt.get("matched_pred_blocks", 0) <= evt["alert_blocks"]
            assert evt.get("matched_gt_events", 0) <= evt["gt_events"]
        results[domain] = {"beat": beat, "event": evt}
        print(f"[EVAL5] {domain}: AUC={beat['auc']:.4f} F1={beat['f1']:.4f} "
              f"Sens={beat['sensitivity']:.4f} Prec={beat['precision']:.4f} | "
              f"evF1={evt.get('event_f1')} FP/rec={evt.get('fp_per_record')} "
              f"blocks={evt.get('alert_blocks')}", flush=True)

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "TH §104 clean_baseline_v5 (SVDB 扩训) 患者级测试集评估; "
                   "模型未见任何测试数据, SVDB 取数仅限其训练患者 (48/15/15 独立划分)",
        "model": str(MODEL.relative_to(BASE)),
        "reference_policy": {"theta": 0.50, "policy": "1-of-5", "cooldown": 5},
        "results": results,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[EVAL5] saved {OUT} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
