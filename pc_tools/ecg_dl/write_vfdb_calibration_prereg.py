#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""write_vfdb_calibration_prereg.py -- pre-registration for the VFDB/CUDB
calibration-protected expansion route.

This writes models/deploy_match/vfdb_calibration_prereg.json.  It deliberately
does NOT train anything.  The file is the lock for any future v6-style training
that mixes v3-A train data with the VFDB/CUDB train split.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_distill_pilot import GATE_CONSTS, FROZEN, BASELINE_REF

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
OUT = CACHE / "vfdb_calibration_prereg.json"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "VFDB/CUDB 校准保护扩训预注册 (v6 candidate)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "AUDIT_ONLY_NO_TRAINING_YET",
        "purpose": (
            "Audit/plan for adding VFDB+CUDB as a new data source to the clean "
            "v3-A pipeline, with explicit calibration protection. The raw audit, "
            "deploy-causal preprocessing, and record-level split audit are done; "
            "training is NOT approved by this file."
        ),
        "data": {
            "raw_dir": "/home/devcontainers/vfdb",
            "raw_audit_json": "models/deploy_match/vfdb_raw_audit.json",
            "preprocessed_npz": "models/deploy_match/vfdb_processed_deploy_causal.npz",
            "preprocess_audit_json": "models/deploy_match/vfdb_preprocess_audit.json",
            "split_audit_json": "models/deploy_match/vfdb_split_audit.json",
            "note": (
                "VFDB 22 records + CUDB 35 records. After strict 250-point "
                "deploy-causal extraction and provisional rhythm labels: "
                "29 records with usable beats, 67,249 beats (N=26,331, A=40,918, "
                "excluded=8,325 discarded from arrays). CUDB XQRS detection "
                "failed on 28/35 records; only 7 CUDB records contributed beats."
            ),
            "record_split": {
                "rule": "patient_level_split(seed=42, 60/20/20); each record = one patient (conservative)",
                "train_records": 19,
                "val_records": 5,
                "test_records": 5,
                "train_beats": 43676,
                "train_normal": 17941,
                "train_abnormal": 25735,
                "val_beats": 14663,
                "val_normal": 4471,
                "val_abnormal": 10192,
                "test_beats": 8910,
                "test_normal": 3919,
                "test_abnormal": 4991,
                "zero_overlap_audited": True,
            },
        },
        "label_map": {
            "vfdb_normal_notes": ["N", "NSR"],
            "vfdb_abnormal": "all non-normal, non-excluded rhythm notes (VT,VF,VFL,VFIB,AFIB,AF,SVTA,NOD,BI,HGEA,VER,SBR,PM,B,...)",
            "vfdb_exclude": ["NOISE", "ASYS", "EMPTY_NOTE"],
            "cudb": "pre-'[' normal; '['->']' VF abnormal; post-']' excluded",
            "note": "provisional map already used for feasibility arrays; final decision locked here if training is approved",
        },
        "training_plan_if_approved": {
            "architecture": "v3 A = build_ecg_resnet_lite_large(dropout=0.4), 62,834 params",
            "seed": 42,
            "data_mix": (
                "v3-A train split (existing distill_pilot_teacher_targets.npz "
                "train x/y) + VFDB/CUDB train split only. VFDB/CUDB val/test "
                "are NEVER used for training or early-stopping."
            ),
            "optimizer": "Adam + CosineDecay(3e-4, 80*steps), batch 32, epochs 80, patience 20, monitor val_auc, restore best",
            "loss": "hard-label CE, no distillation, no teacher targets",
            "calibration": {
                "method": "post-hoc isotonic or temperature scaling fitted on val/probabilities (monotone; ranking/AUC-invariant)",
                "fit_data": "val split only; never test; never VFDB test split",
                "required_report": "pre/post calibration Brier or ECE + theta=0.5 metrics",
            },
        },
        "gates": GATE_CONSTS,
        "frozen_operating_point": FROZEN,
        "baseline_references_from_disk": BASELINE_REF,
        "calibration_protection": {
            "real_afe_holdout_not_worse": {
                "criterion": (
                    "mean_prob <= max(0.10, baseline_mean + 0.02) AND "
                    "frac_gt_0.5 <= baseline_frac_gt_0.5"
                ),
                "baseline_ref": BASELINE_REF,
            },
            "theta05_false_alarm_not_worse": {
                "criterion": (
                    "pooled val normal beat frac>0.5 increase <= +0.02 vs baseline; "
                    "pure-normal record alert blocks <= baseline count (from p1/val "
                    "per-record blocks), reported as relative indicator"
                ),
                "note": "baseline values loaded at evaluation time from disk; never hard-coded without proof",
            },
            "frozen_event_recall": {
                "criterion": "drop <= 0.02 vs baseline at theta=0.95/1-of-5/cooldown=20",
            },
            "no_strategy_only": {
                "rule": "A verdict cannot be PASS if calibration/theta=0.5/frozen-recall reports are missing.",
            },
        },
        "verdict_semantics": {
            "PASS": "primary reduction gate + guardrail A/B/C + all calibration protection gates pass",
            "PARTIAL": "primary passes but at least one guardrail/calibration gate breaks",
            "UNPROVEN": "primary does not pass or data/split impossible; not called disproven",
        },
        "zero_test_contact": True,
        "tests_holdout": "the existing 40-beat real AFE holdout remains untouched in training; used only in gates",
        "known_limitations": [
            "VFDB/CUDB are not healthy-population SVEB Holter data like SVDB; VFDB N segments come from patients with malignant arrhythmias, so 'normal' here is normal-sinus morphology in a pathological population.",
            "CUDB XQRS failure on 28/35 records reduces usablity; current usable set is mostly VFDB.",
            "label map is provisional and must be re-confirmed before any training.",
            "val double-use remains (early stopping + gates); numbers are relative indicators.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[VFPRE] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
