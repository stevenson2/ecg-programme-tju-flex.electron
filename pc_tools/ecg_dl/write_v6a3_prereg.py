#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""write_v6a3_prereg.py -- preregister targeted hard-normal fine-tune.

v6a3 = fine-tune v3 A on original v3-A train + 50k hard-normal only
(no external abnormal shift).  This is the most targeted false-alarm fix.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_distill_pilot import GATE_CONSTS, FROZEN, BASELINE_REF

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
OUT = CACHE / "v6a3_prereg.json"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "v6a3 定向 hard-normal 微调预注册",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": (
            "v6a1/v6a2 showed that adding 50k external abnormal beats shifts the "
            "decision boundary badly. v6a3 keeps the original v3-A train abnormal "
            "distribution and only adds 50k hard-normal beats from the mined "
            "train-safe pool."
        ),
        "recipe": {
            "name": "v6a3",
            "arch": "v3 A (62,834 params)",
            "init": "v3 A weights, all layers trainable",
            "data": "v6a3_train_hardnorm_anchor.npz: 50k hard normal (>0.5) + original v3-A train 5,942 (2,500 abnormal + 3,442 normal)",
            "optimizer": "Adam lr=1e-5",
            "batch": 128,
            "epochs_requested": 20,
            "early_stopping": {"monitor": "val_auc", "patience": 5, "restore_best": True},
            "loss": "sparse categorical crossentropy",
            "augmentation": "none",
        },
        "gates": GATE_CONSTS,
        "frozen_operating_point": FROZEN,
        "baseline_references_from_disk": BASELINE_REF,
        "calibration_protection": {
            "real_afe_holdout_not_worse": "same as v6 prereg",
            "theta05_false_alarm_not_worse": "same as v6 prereg",
            "frozen_event_recall_drop_max": 0.02,
        },
        "verdict_semantics": {
            "PASS": "primary + guardrail A/B/C + calibration pass",
            "PARTIAL": "primary passes but a guardrail/calibration breaks",
            "UNPROVEN": "primary does not pass",
        },
        "zero_test_contact": True,
        "outputs": {
            "model": "models/deploy_match/best_v6a3_hardnorm_anchor_100k.h5",
            "train_json": "models/deploy_match/train_v6a3_hardnorm_anchor_100k.json",
            "log": "models/deploy_match/train_v6a3_hardnorm_anchor_100k.log",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[V6A3PRE] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
