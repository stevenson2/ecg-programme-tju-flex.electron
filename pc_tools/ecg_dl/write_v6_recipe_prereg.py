#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""write_v6_recipe_prereg.py -- pre-register v6a fast recipe.

v6a = fine-tune v3 A on the 100k balanced hard-normal-heavy mined dataset.
This writes models/deploy_match/v6_recipe_prereg.json.  No training here.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_distill_pilot import GATE_CONSTS, FROZEN, BASELINE_REF

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
OUT = CACHE / "v6_recipe_prereg.json"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "v6a 快速微调配方预注册",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": (
            "Fast recipe check: fine-tune v3 A on 100k balanced mined dataset "
            "(50k abnormal + 50k hard-normal >0.5) to see whether the new "
            "hard-normal pool can reduce val over-alarm without breaking recall/calibration."
        ),
        "recipe": {
            "name": "v6a",
            "arch": "v3 A (62,834 params)",
            "init": "load best_resnet_large_clean_baseline_v3.h5, then fine-tune all layers",
            "data": {
                "npz": "models/deploy_match/v6a_train_100k.npz",
                "n": 100000,
                "normal": 50000,
                "abnormal": 50000,
                "normal_source": "hard-normal pool (v3 A score > 0.5) + stratified source quotas",
                "note": "no test/val data; all from train-safe mined pool",
            },
            "optimizer": "Adam + CosineDecay(3e-4, 40*steps)",
            "batch_size": 128,
            "epochs_requested": 40,
            "early_stopping": {
                "monitor": "val_auc",
                "patience": 8,
                "restore_best": True,
            },
            "val_for_early_stopping": "v3 A val split (distill_pilot_teacher_targets.npz x_val/y_val), 4040 beats",
            "loss": "sparse categorical crossentropy (hard labels only, no KD)",
            "augmentation": "none for speed",
        },
        "gates": GATE_CONSTS,
        "frozen_operating_point": FROZEN,
        "baseline_references_from_disk": BASELINE_REF,
        "calibration_protection": {
            "real_afe_holdout_not_worse": {
                "criterion": "mean_prob <= max(0.10, baseline_mean + 0.02) AND frac_gt_0.5 <= baseline_frac_gt_0.5"
            },
            "theta05_false_alarm_not_worse": {
                "criterion": "val pooled normal frac>0.5 increase <= +0.02; pure-normal record alert blocks <= baseline"
            },
            "frozen_event_recall": {
                "criterion": "event_recall drop <= 0.02 at theta=0.95/1-of-5/cooldown=20"
            },
            "no_strategy_only": {
                "rule": "PASS requires calibration/theta=0.5/frozen-recall reports present"
            },
        },
        "verdict_semantics": {
            "PASS": "primary + guardrail A/B/C + calibration protection all pass",
            "PARTIAL": "primary passes but at least one guardrail/calibration gate breaks",
            "UNPROVEN": "primary does not pass or recipe impossible; not called disproven",
        },
        "outputs": {
            "model": "models/deploy_match/best_v6a_finetune_v3a_100k.h5",
            "train_json": "models/deploy_match/train_v6a_finetune_v3a_100k.json",
            "train_log": "models/deploy_match/train_v6a_finetune_v3a_100k.log",
            "eval_json": "models/deploy_match/eval_v6a_finetune_v3a_100k.json",
        },
        "zero_test_contact": True,
        "known_limitations": [
            "fine-tune from v3 A may be faster but could preserve existing biases",
            "100k is a fast subset; if signal appears, expand to 200k-400k",
            "val double-use unchanged (early stopping + gates)",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[V6PRE] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
