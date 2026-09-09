#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""acceptance_theta95_v4.py — v4 冻结操作点完整验收门 (TH §103)
================================================================================
与 acceptance_theta95_v3.py (§102) 完全同门口径, 仅换模型为
clean_baseline_v4; 操作点冻结 (θ=0.95 / 1-of-5 / cooldown=20, 不重扫):

  阈值相关门:
    - 拍级混淆矩阵 = 原始概率 > 0.95
    - 事件级 / FP/rec / 报警块 = 策略 (0.95, 1-of-5, cooldown=20)
  阈值无关门:
    - clean AUC (两域)、真 AFE 留出、训练 best val AUC (读 v4 训练 JSON)

测试集在训练/挖掘/调参全程零接触, 本脚本是唯一评估入口。
输出: models/deploy_match/acceptance_theta95_v4.json
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
OUT = CACHE / "acceptance_theta95_v4.json"

THETA = 0.95
POLICY = {"theta": 0.95, "k": 1, "n": 5, "cooldown": 20}

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

    beat = {n: raw_beat_metrics(l, p, THETA) for n, (_g, l, p) in sets.items()}
    evt = {n: pooled_metrics(g, l, p, POLICY["theta"], POLICY["k"],
                             POLICY["n"], POLICY["cooldown"])
           for n, (g, l, p) in sets.items()}

    tr = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    hold_mean = tr["results"]["real_holdout_mean_prob"]
    hold_frac = tr["results"]["real_holdout_frac_gt_0.5"]
    best_val_auc = tr["results"]["best_val_auc"]

    mi_b, mi_e = beat["mit_incart"], evt["mit_incart"]
    pt_b, pt_e = beat["ptb"], evt["ptb"]
    checks = {
        "mit_incart_auc": (mi_b["auc"] >= GATES["mit_incart_auc_min"],
                           mi_b["auc"], f">={GATES['mit_incart_auc_min']}"),
        "mit_incart_beat_f1": (mi_b["f1"] >= GATES["mit_incart_beat_f1_min"],
                               mi_b["f1"], f">={GATES['mit_incart_beat_f1_min']}"),
        "mit_incart_fp_per_record": (
            mi_e["fp_per_record"] <= GATES["mit_incart_fp_per_record_max"],
            mi_e["fp_per_record"], f"<={GATES['mit_incart_fp_per_record_max']}"),
        "mit_incart_event_f1": (
            mi_e["event_f1"] >= GATES["mit_incart_event_f1_min"],
            mi_e["event_f1"], f">={GATES['mit_incart_event_f1_min']}"),
        "mit_incart_beat_sensitivity": (
            mi_b["sensitivity"] >= GATES["mit_incart_beat_sensitivity_min"],
            mi_b["sensitivity"], f">={GATES['mit_incart_beat_sensitivity_min']}"),
        "ptb_auc": (pt_b["auc"] >= GATES["ptb_auc_min"],
                    pt_b["auc"], f">={GATES['ptb_auc_min']}"),
        "ptb_event_f1": (pt_e["event_f1"] >= GATES["ptb_event_f1_min"],
                         pt_e["event_f1"], f">={GATES['ptb_event_f1_min']}"),
        "collapse_detector": (mi_e["alert_blocks"] >= GATES["alert_blocks_min"],
                              mi_e["alert_blocks"], f">={GATES['alert_blocks_min']}"),
        "real_afe_holdout": (
            hold_mean < GATES["real_holdout_mean_prob_max"]
            and hold_frac <= GATES["real_holdout_frac_gt_0.5_max"],
            {"mean_prob": hold_mean, "frac_gt_0.5": hold_frac},
            "mean<0.10 且 frac=0"),
        "train_best_val_auc": (best_val_auc >= GATES["train_best_val_auc_min"],
                               best_val_auc, f">={GATES['train_best_val_auc_min']}"),
    }
    passed = [k for k, (ok, _v, _c) in checks.items() if ok]
    failed = [k for k, (ok, _v, _c) in checks.items() if not ok]

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "TH §103: clean_baseline_v4 在冻结操作点 (θ=0.95, 1-of-5, "
                   "cooldown=20) 上的完整验收门; 操作点继承 §101/§102, 不重扫; "
                   "测试集从未参与训练/挖掘/选择",
        "model": str(MODEL.relative_to(BASE)),
        "operating_point": POLICY,
        "metric_caliber": {
            "beat": f"原始概率 > {THETA} 的混淆矩阵 (与 §102 同源)",
            "event": "策略 (0.95, 1-of-5, cooldown=20), evaluate_sequence_set 池化语义",
            "threshold_independent": "clean AUC / 真 AFE 留出 / best val AUC",
        },
        "gates": GATES,
        "beat_raw_theta": beat,
        "event_policy": evt,
        "threshold_independent": {
            "real_holdout_mean_prob": hold_mean,
            "real_holdout_frac_gt_0.5": hold_frac,
            "train_best_val_auc": best_val_auc,
        },
        "checks": {k: {"passed": ok, "value": v, "criterion": c}
                   for k, (ok, v, c) in checks.items()},
        "verdict": {"passed": passed, "failed": failed,
                    "overall": "PASS" if not failed else "FAIL"},
        "supplementary": {
            "mit_incart_beat_sens_after_k_of_n": evt["mit_incart"]["beat_recall"],
            "note": "事件级拍召回 (1-of-5 平滑后) 另列, 非验收门口径",
        },
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    for k, (ok, v, c) in checks.items():
        print(f"[ACC4] {'PASS' if ok else 'FAIL'} {k}: {v} ({c})", flush=True)
    print(f"[ACC4] overall={report['verdict']['overall']} "
          f"failed={failed} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
