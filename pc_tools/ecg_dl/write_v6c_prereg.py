#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""write_v6c_prereg.py -- preregister v6c MIT/INCART-only balanced from-scratch.

v6a/v6b showed that cross-source mined data or extreme class imbalance hurts
val AUC. v6c keeps only MIT/INCART source (same domain family as val), balanced
16k hard normal + 16k abnormal, and trains v3 A from scratch.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_distill_pilot import GATE_CONSTS, FROZEN, BASELINE_REF

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
OUT = CACHE / "v6c_prereg.json"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "v6c MIT/INCART-only balanced from-scratch 预注册",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Test whether a same-domain, balanced MIT/INCART hard-normal pool can train v3 A from scratch with good val AUC.",
        "recipe": {
            "name": "v6c",
            "arch": "v3 A (62,834 params), from scratch",
            "data": "v6c_train_mitincart_balanced32k.npz (16k hard normal source0/1 + 16k abnormal source0/1)",
            "optimizer": "Adam + CosineDecay(3e-4, 40*steps)",
            "batch": 128,
            "epochs_requested": 40,
            "early_stopping": {"monitor": "val_auc", "patience": 8, "restore_best": True},
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
            "model": "models/deploy_match/best_v6c_scratch_mitincart_balanced32k.h5",
            "train_json": "models/deploy_match/train_v6c_scratch_mitincart_balanced32k.json",
            "log": "models/deploy_match/train_v6c_scratch_mitincart_balanced32k.log",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[V6CPRE] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
