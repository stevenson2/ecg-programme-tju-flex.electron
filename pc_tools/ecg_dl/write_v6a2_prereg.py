#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""write_v6a2_prereg.py -- preregister v6a2 low-LR + original-train anchor.

v6a1 (same recipe at 3e-4, no anchor) catastrophically forgot v3-A val AUC
during first epochs (baseline ~0.77 -> ~0.56). v6a2 tests whether a much lower
LR plus the original 5,942 v3-A train beats as an anchor prevents forgetting.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_distill_pilot import GATE_CONSTS, FROZEN, BASELINE_REF

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
OUT = CACHE / "v6a2_prereg.json"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "v6a2 低LR+锚点微调预注册",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "After v6a1 at LR=3e-4 collapsed val AUC, test LR=1e-5 + original v3-A train anchor on the same 100k hard-normal dataset.",
        "recipe": {
            "name": "v6a2",
            "arch": "v3 A (62,834 params)",
            "init": "v3 A weights, all layers trainable",
            "data": "v6a_train_100k.npz (50k normal hard >0.5 + 50k abnormal) + original v3-A train 5,942 beats as anchor",
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
            "model": "models/deploy_match/best_v6a2_lowlr_anchor_100k.h5",
            "train_json": "models/deploy_match/train_v6a2_lowlr_anchor_100k.json",
            "log": "models/deploy_match/train_v6a2_lowlr_anchor_100k.log",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[V6A2PRE] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
