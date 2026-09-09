#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_clean_baseline_v3.py — clean_baseline_v3 模型的患者级测试集评估 (TH §100)
================================================================================
与 eval_clean_baseline.py (§99) 完全同口径: 指标定义、θ=0.5、1-of-5、
cooldown=5、测试缓存 npz 一律不变; 仅模型路径与输出 JSON 按配置参数化。

用法: python3 eval_clean_baseline_v3.py [v3|v3_b|v3_c]  (默认 v3)
输出: models/deploy_match/clean_baseline_eval_<suffix>.json
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

ANCHORS = {
    "v2_clean_baseline_mit_incart": {"auc": 0.851, "event_f1": 0.643, "beat_f1": 0.281},
    "v4_clean_mit_incart": {"auc": 0.848, "event_f1": 0.697, "beat_f1": 0.498},
    "ptb_clean_deployed_int8": {"auc": 0.900, "event_f1": 0.898},
}

GATES = {
    "mit_incart": {"auc": 0.848, "beat_f1": 0.40, "fp_per_record": 1.5,
                   "event_f1": 0.64, "sensitivity": 0.90, "alert_blocks_min": 3},
    "ptb": {"auc": 0.70, "event_f1": 0.75},
    "train_best_val_auc": 0.80,
    "real_holdout_mean_prob": 0.10,
}


def main():
    suffix = sys.argv[1] if len(sys.argv) > 1 else "v3"
    if suffix not in ("v3", "v3_b", "v3_c"):
        raise SystemExit(f"unknown suffix {suffix!r}, expected v3|v3_b|v3_c")
    model = MODELS / f"best_resnet_large_clean_baseline_{suffix}.h5"
    out = CACHE / f"clean_baseline_eval_{suffix}.json"

    t0 = time.time()
    dmi = np.load(MIT_NPZ)
    dpt = np.load(PTB_NPZ)
    mi_beats = np.asarray(dmi["beats"], dtype=np.float32)
    mi_labels = np.asarray(dmi["labels"]).astype(np.int32)
    mi_rids = np.asarray(dmi["record_ids"])
    pt_beats = np.asarray(dpt["beats"], dtype=np.float32)
    pt_labels = np.asarray(dpt["labels"]).astype(np.int32)
    pt_rids = np.asarray(dpt["record_ids"])
    print(f"[EVAL3:{suffix}] MIT+INCART test: {len(mi_beats)} beats "
          f"(abn={int((mi_labels == 1).sum())}, records={len(np.unique(mi_rids))})", flush=True)
    print(f"[EVAL3:{suffix}] PTB test: {len(pt_beats)} beats "
          f"(abn={int((pt_labels == 1).sum())}, records={len(np.unique(pt_rids))})", flush=True)

    predict = make_predictor("h5", model)
    p_mi = predict(mi_beats)
    p_pt = predict(pt_beats)

    results = {}
    for domain, labels, probs, rids in (
            ("mit_incart", mi_labels, p_mi, mi_rids),
            ("ptb", pt_labels, p_pt, pt_rids)):
        beat = beat_metrics(labels, probs)
        evt = event_metrics(rids, labels, probs, f"{domain}_test")
        # ---- AGENTS §8 合理性断言 (与 §99 修正版一致) ----
        assert beat["tp"] + beat["fp"] + beat["tn"] + beat["fn"] == beat["n_beats"], \
            f"{domain}: 混淆矩阵与样本总数不自洽"
        if evt.get("alert_blocks") is not None and evt.get("gt_events") is not None:
            assert evt.get("matched_pred_blocks", 0) <= evt["alert_blocks"], \
                f"{domain}: 匹配报警块数超过报警块总数 (口径异常)"
            assert evt.get("matched_gt_events", 0) <= evt["gt_events"], \
                f"{domain}: 匹配 GT 事件数超过 GT 事件总数 (口径异常)"
        results[domain] = {"beat": beat, "event": evt}
        print(f"[EVAL3:{suffix}] {domain}: AUC={beat['auc']:.4f} F1={beat['f1']:.4f} "
              f"Sens={beat['sensitivity']:.4f} Prec={beat['precision']:.4f}", flush=True)
        if evt:
            print(f"[EVAL3:{suffix}] {domain}: evF1={evt.get('event_f1')} "
                  f"evRec={evt.get('event_recall')} evPrec={evt.get('event_precision')} "
                  f"FP/rec={evt.get('fp_per_record')} blocks={evt.get('alert_blocks')}",
                  flush=True)

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": f"TH §100 clean_baseline_v3 ({suffix}) 患者级测试集评估; "
                   "模型未见任何测试数据, full=诚实口径",
        "model": str(model.relative_to(BASE)),
        "reference_policy": {"theta": 0.50, "policy": "1-of-5", "cooldown": 5},
        "test_sets": {
            "mit_incart": {"beats": int(len(mi_beats)),
                           "records": [int(r) for r in np.unique(mi_rids)]},
            "ptb": {"beats": int(len(pt_beats)),
                    "records": [int(r) for r in np.unique(pt_rids)]},
        },
        "anchors": ANCHORS,
        "acceptance_gates": GATES,
        "results": results,
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[EVAL3:{suffix}] saved {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
