#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_clean_baseline.py — 干净基线模型的患者级测试集评估 (TH §99)
================================================================================
对象: models/best_resnet_large_clean_baseline.h5 (从零训练, 从未见过任何
测试患者数据) —— 因此 full 测试集口径即诚实口径, 无需剔除子集。

测试集 (与 §96/§98 同口径, 患者级):
  MIT+INCART: models/deploy_match/mit_deploy_causal_match.npz (INCART rid +100000)
  PTB       : models/deploy_match/ptb_deploy_causal_match.npz

指标: 拍级 AUC / θ=0.5 混淆矩阵 / Sens / Prec / F1; 事件级 θ=0.5 1-of-5
cooldown=5 (与 §96 参考操作点一致)。合理性断言按 AGENTS §8 内嵌。

输出: models/deploy_match/clean_baseline_eval.json
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
MODEL = MODELS / "best_resnet_large_clean_baseline.h5"
MIT_NPZ = CACHE / "mit_deploy_causal_match.npz"
PTB_NPZ = CACHE / "ptb_deploy_causal_match.npz"
OUT = CACHE / "clean_baseline_eval.json"

ANCHORS = {
    "v4_clean_mit_incart": {"auc": 0.848, "event_f1": 0.697, "beat_f1": 0.498},
    "ptb_clean_deployed_int8": {"auc": 0.900, "event_f1": 0.898},
}


def main():
    t0 = time.time()
    dmi = np.load(MIT_NPZ)
    dpt = np.load(PTB_NPZ)
    mi_beats = np.asarray(dmi["beats"], dtype=np.float32)
    mi_labels = np.asarray(dmi["labels"]).astype(np.int32)
    mi_rids = np.asarray(dmi["record_ids"])
    pt_beats = np.asarray(dpt["beats"], dtype=np.float32)
    pt_labels = np.asarray(dpt["labels"]).astype(np.int32)
    pt_rids = np.asarray(dpt["record_ids"])
    print(f"[EVAL] MIT+INCART test: {len(mi_beats)} beats "
          f"(abn={int((mi_labels == 1).sum())}, records={len(np.unique(mi_rids))})", flush=True)
    print(f"[EVAL] PTB test: {len(pt_beats)} beats "
          f"(abn={int((pt_labels == 1).sum())}, records={len(np.unique(pt_rids))})", flush=True)

    predict = make_predictor("h5", MODEL)
    p_mi = predict(mi_beats)
    p_pt = predict(pt_beats)

    results = {}
    for domain, labels, probs, rids in (
            ("mit_incart", mi_labels, p_mi, mi_rids),
            ("ptb", pt_labels, p_pt, pt_rids)):
        beat = beat_metrics(labels, probs)
        evt = event_metrics(rids, labels, probs, f"{domain}_test")
        # ---- AGENTS §8 合理性断言 ----
        assert beat["tp"] + beat["fp"] + beat["tn"] + beat["fn"] == beat["n_beats"], \
            f"{domain}: 混淆矩阵与样本总数不自洽"
        if evt.get("alert_blocks") is not None and evt.get("gt_events") is not None:
            # 一个报警块可匹配多个 GT 事件, 故只断言子集关系
            assert evt.get("matched_pred_blocks", 0) <= evt["alert_blocks"], \
                f"{domain}: 匹配报警块数超过报警块总数 (口径异常)"
            assert evt.get("matched_gt_events", 0) <= evt["gt_events"], \
                f"{domain}: 匹配 GT 事件数超过 GT 事件总数 (口径异常)"
        results[domain] = {"beat": beat, "event": evt}
        print(f"[EVAL] {domain}: AUC={beat['auc']:.4f} F1={beat['f1']:.4f} "
              f"Sens={beat['sensitivity']:.4f} Prec={beat['precision']:.4f}", flush=True)
        if evt:
            print(f"[EVAL] {domain}: evF1={evt.get('event_f1')} "
                  f"evRec={evt.get('event_recall')} evPrec={evt.get('event_precision')} "
                  f"FP/rec={evt.get('fp_per_record')}", flush=True)

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "TH §99 干净基线 (从零训练) 患者级测试集评估; 模型未见任何测试数据, full=诚实口径",
        "model": str(MODEL.relative_to(BASE)),
        "reference_policy": {"theta": 0.50, "policy": "1-of-5", "cooldown": 5},
        "test_sets": {
            "mit_incart": {"beats": int(len(mi_beats)),
                           "records": [int(r) for r in np.unique(mi_rids)]},
            "ptb": {"beats": int(len(pt_beats)),
                    "records": [int(r) for r in np.unique(pt_rids)]},
        },
        "anchors": ANCHORS,
        "results": results,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[EVAL] saved {OUT} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
