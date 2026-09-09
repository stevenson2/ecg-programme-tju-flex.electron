#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""write_v6a4_prereg.py -- preregister v6a4 MIT/INCART-only hard-normal fine-tune.

v6a3 (all-source hard normals) dropped val AUC to ~0.63. v6a4 restricts hard
normals to MIT/INCART train-safe beats, the same domain family as the val split.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_distill_pilot import GATE_CONSTS, FROZEN, BASELINE_REF

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
OUT = CACHE / "v6a4_prereg.json"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "v6a4 MIT/INCART-only hard-normal 微调预注册",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Restrict hard-normal additions to MIT/INCART train beats to avoid domain shift from SVDB/AFDB/VFDB.",
        "recipe": {
            "name": "v6a4",
            "arch": "v3 A (62,834 params)",
            "init": "v3 A weights, all layers trainable",
            "data": "v6a4_train_mitincart_hardnorm_anchor.npz (16k MIT/INCART hard normal + original v3-A train 5,942)",
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
            "model": "models/deploy_match/best_v6a4_mitincart_hardnorm_anchor.h5",
            "train_json": "models/deploy_match/train_v6a4_mitincart_hardnorm_anchor.json",
            "log": "models/deploy_match/train_v6a4_mitincart_hardnorm_anchor.log",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[V6A4PRE] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
