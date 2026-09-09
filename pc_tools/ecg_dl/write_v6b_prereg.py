#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""write_v6b_prereg.py -- preregister v6b from-scratch on the 100k mined pool.

v6a fine-tune variants all degraded val AUC. v6b tests whether training v3 A
from scratch on the same 100k balanced mined pool can learn a new representation
that includes hard-normal beats without inheriting v3 A's boundary.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_distill_pilot import GATE_CONSTS, FROZEN, BASELINE_REF

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
OUT = CACHE / "v6b_prereg.json"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "v6b 从零训练 100k 挖掘池预注册",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "From-scratch v3 A on the same 100k mined dataset to see if the hard-normal pool can produce a new, more stable representation.",
        "recipe": {
            "name": "v6b",
            "arch": "v3 A (62,834 params), built from scratch, no v3 A weights",
            "data": "v6a_train_100k.npz (50k hard normal + 50k external abnormal)",
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
            "model": "models/deploy_match/best_v6b_scratch_v3a_100k.h5",
            "train_json": "models/deploy_match/train_v6b_scratch_v3a_100k.json",
            "log": "models/deploy_match/train_v6b_scratch_v3a_100k.log",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[V6BPRE] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
