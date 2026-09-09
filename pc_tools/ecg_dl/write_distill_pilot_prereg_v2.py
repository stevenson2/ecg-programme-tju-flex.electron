#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""write_distill_pilot_prereg_v2.py -- write the pre-registration file for the
TH106+ recall-preserving distill-pilot variant (alpha=0.2 + teacher tail
sharpening T=0.7 pre-packing; KD loss remains T=1.0).

Run BEFORE any v2 training/evaluation.  No test contact.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_distill_pilot import GATE_CONSTS, FROZEN, BASELINE_REF

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
OUT = CACHE / "distill_pilot_prereg_v2.json"

VARIANT = {
    "name": "alpha0.2_teacher_tail_sharpen_t0.7",
    "alpha": 0.2,
    "teacher_tail_pre_sharpen_temperature": 0.7,
    "kd_temperature": 1.0,
    "tail_mode": "sharpen",
    "rationale": (
        "The v1 pilot (alpha=0.5, T=1) cleared the theta=0.95 high-confidence "
        "false-alarm tail but collapsed global confidence: frozen event recall "
        "fell to 0.33-0.50 and real-holdout guardrail C also broke on one of "
        "three seeds. This v2 variant keeps the hard-label anchor strong "
        "(alpha=0.2) and pre-sharpens the teacher soft probabilities with a "
        "monotone logit temperature T=0.7 before the standard KL packing, so "
        "the soft target does not drag the abnormal tail back toward 0.5. "
        "The KD loss itself stays at T=1.0; the transform only changes the "
        "teacher scale, not the ranking (AUC-invariant), and is fitted/derived "
        "from the existing Stage-1 package (no val/test label use, no new data)."
    ),
}


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "TH106+ v2 recall-preserving distill variant prereg",
        "gates": GATE_CONSTS,
        "frozen_operating_point": FROZEN,
        "baseline_references_from_disk": BASELINE_REF,
        "variant": VARIANT,
        "verdict_semantics": {
            "PASS": "primary + guardrail A/B/C all pass",
            "PARTIAL": "primary passes but at least one guardrail breaks",
            "UNPROVEN": "primary does not pass (relative reduction below the "
                        "30% pre-registered bar, or negative), recorded as "
                        "unproven rather than disproven; single-seed noise is "
                        "not treated as disproof",
        },
        "zero_test_contact": True,
        "known_limitations": [
            "val double-use: early stopping + all gates share val; numbers are relative indicators",
            "teacher domain offset: tile view 0.8893 vs true deploy 10s 0.7794",
            "pseudo-target contains noise (LORO AUC 0.88, not oracle)",
            "variant is pre-registered; only PASS/PARTIAL/UNPROVEN verdicts are allowed",
        ],
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Pre-register TH106+ v2 recall-preserving distill variant",
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[DP2PRE] prereg v2 saved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
