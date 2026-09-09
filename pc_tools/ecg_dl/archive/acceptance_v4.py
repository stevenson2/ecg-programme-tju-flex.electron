#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""acceptance_v4.py — v4 模型双操作点完整验收门 (TH §103)
================================================================================
θ=0.5 (规范口径) 与 θ=0.95 (§101 选中操作点) 各跑一遍 §100 验收门。
拍级 = 原始概率混淆矩阵; 事件级/FP-rec = 对应策略 (θ, 1-of-5, cooldown);
AUC/真 AFE 留出/val AUC 为阈值无关量 (读训练 JSON)。
输出: models/deploy_match/acceptance_v4.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_clean_test import make_predictor
from sweep_val_policy_v3 import pooled_metrics, record_groups

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
MODEL = MODELS / "best_resnet_large_clean_baseline_v4.h5"
MIT_NPZ = CACHE / "mit_deploy_causal_match.npz"
PTB_NPZ = CACHE / "ptb_deploy_causal_match.npz"
TRAIN_JSON = CACHE / "train_clean_baseline_v4.json"
OUT = CACHE / "acceptance_v4.json"

OPS = {
    "theta0.5_k1of5_cd5": {"theta": 0.5, "k": 1, "n": 5, "cooldown": 5},
    "theta0.95_k1of5_cd20": {"theta": 0.95, "k": 1, "n": 5, "cooldown": 20},
}

GATES = {
    "mit_incart_auc_min": 0.848,
    "mit_incart_beat_f1_min": 0.40,
    "mit_incart_fp_per_record_max": 1.5,
    "mit_incart_event_f1_min": 0.64,
    "mit_incart_beat_sensitivity_min": 0.90,
    "ptb_auc_min": 0.70,
    "ptb_event_f1_min": 0.75,
    "alert_blocks_min": 3,
    "real_holdout_mean_prob_max": 0.10,
    "real_holdout_frac_gt_0.5_max": 0.0,
    "train_best_val_auc_min": 0.80,
}


def raw_beat_metrics(labels, probs, theta):
    pred = (probs > theta).astype(np.int32)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    f1 = 2 * sens * prec / (sens + prec) if (sens and prec) else None
    return {"theta": theta, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": sens, "precision": prec, "f1": f1,
            "auc": float(roc_auc_score(labels, probs)),
            "n_beats": int(len(labels))}


def gate_check(beat_mi, evt_mi, beat_pt, evt_pt, hold_mean, hold_frac, best_val):
    checks = {
        "mit_incart_auc": (beat_mi["auc"] >= GATES["mit_incart_auc_min"],
                           beat_mi["auc"]),
        "mit_incart_beat_f1": (beat_mi["f1"] >= GATES["mit_incart_beat_f1_min"],
                               beat_mi["f1"]),
        "mit_incart_fp_per_record": (
            evt_mi["fp_per_record"] <= GATES["mit_incart_fp_per_record_max"],
            evt_mi["fp_per_record"]),
        "mit_incart_event_f1": (
            evt_mi["event_f1"] >= GATES["mit_incart_event_f1_min"],
            evt_mi["event_f1"]),
        "mit_incart_beat_sensitivity": (
            beat_mi["sensitivity"] >= GATES["mit_incart_beat_sensitivity_min"],
            beat_mi["sensitivity"]),
        "ptb_auc": (beat_pt["auc"] >= GATES["ptb_auc_min"], beat_pt["auc"]),
        "ptb_event_f1": (evt_pt["event_f1"] >= GATES["ptb_event_f1_min"],
                         evt_pt["event_f1"]),
        "collapse_detector": (evt_mi["alert_blocks"] >= GATES["alert_blocks_min"],
                              evt_mi["alert_blocks"]),
        "real_afe_holdout": (
            hold_mean < GATES["real_holdout_mean_prob_max"]
            and hold_frac <= GATES["real_holdout_frac_gt_0.5_max"],
            {"mean_prob": hold_mean, "frac_gt_0.5": hold_frac}),
        "train_best_val_auc": (best_val >= GATES["train_best_val_auc_min"],
                               best_val),
    }
    failed = [k for k, (ok, _v) in checks.items() if not ok]
    return {
        "checks": {k: {"passed": ok, "value": v} for k, (ok, v) in checks.items()},
        "failed": failed,
        "overall": "PASS" if not failed else "FAIL",
    }


def main():
    t0 = time.time()
    predict = make_predictor("h5", MODEL)
    sets = {}
    for name, path in (("mit_incart", MIT_NPZ), ("ptb", PTB_NPZ)):
        d = np.load(path)
        b = np.asarray(d["beats"], dtype=np.float32)
        l = np.asarray(d["labels"]).astype(np.int32)
        r = np.asarray(d["record_ids"])
        sets[name] = (record_groups(r), l, predict(b))

    tr = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    hold_mean = tr["results"]["real_holdout_mean_prob"]
    hold_frac = tr["results"]["real_holdout_frac_gt_0.5"]
    best_val = tr["results"]["best_val_auc"]

    verdicts = {}
    for key, op in OPS.items():
        beat = {n: raw_beat_metrics(l, p, op["theta"])
                for n, (_g, l, p) in sets.items()}
        evt = {n: pooled_metrics(g, l, p, op["theta"], op["k"], op["n"],
                                 op["cooldown"])
               for n, (g, l, p) in sets.items()}
        verdicts[key] = {
            "policy": op,
            "beat_raw_theta": beat,
            "event_policy": evt,
            "gates": gate_check(beat["mit_incart"], evt["mit_incart"],
                                beat["ptb"], evt["ptb"],
                                hold_mean, hold_frac, best_val),
        }
        v = verdicts[key]["gates"]
        print(f"[ACC4] {key}: overall={v['overall']} failed={v['failed']}",
              flush=True)

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "TH §103: v4 硬负例模型双操作点验收门 (测试集从未参与选择)",
        "model": str(MODEL.relative_to(BASE)),
        "gates": GATES,
        "threshold_independent": {
            "real_holdout_mean_prob": hold_mean,
            "real_holdout_frac_gt_0.5": hold_frac,
            "train_best_val_auc": best_val,
        },
        "verdicts": verdicts,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[ACC4] saved {OUT} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
