#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_distill_pilot_models.py -- evaluate one or more Stage-2 distill-pilot
models against the frozen v3-A baseline, using the registered TH106 gates.

This is a new, non-frozen evaluator.  It reuses the same evaluation primitives
as eval_distill_pilot.py (val_sequences, pooled_metrics, per-record blocks) and
the same validity rules: val-side only, zero test contact.

Usage:
  python3 eval_distill_pilot_models.py \
      --model best_resnet_large_distill_pilot_seed42.h5:seed42 \
      --model best_resnet_large_distill_pilot_seed43.h5:seed43 \
      --prereg distill_pilot_prereg.json \
      --out distill_pilot_multiseed_eval.json
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_clean_test import make_predictor
from sweep_val_policy_v3 import val_sequences
from eval_distill_pilot import (
    evaluate_model, summarize_per_record, FROZEN, BASELINE_REF,
)

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
MODEL_BASE = MODELS / "best_resnet_large_clean_baseline_v3.h5"
TARGETS_NPZ = CACHE / "distill_pilot_teacher_targets.npz"


def load_prereg(name):
    path = CACHE / name
    if not path.exists():
        raise SystemExit(f"[DPM] prereg missing: {path}")
    reg = json.loads(path.read_text(encoding="utf-8"))
    return reg


def compute_gates(base, pilot, holdout_pilot, prereg):
    gates_cfg = prereg["gates"]
    b_frac = base["pooled_norm_frac_gt_0.95"]
    p_frac = pilot["pooled_norm_frac_gt_0.95"]
    rel_red = (b_frac - p_frac) / b_frac
    b_auc = base["mit_incart"]["auc"]
    p_auc = pilot["mit_incart"]["auc"]
    b_ev = base["mit_incart"]["frozen"]["event_recall"]
    p_ev = pilot["mit_incart"]["frozen"]["event_recall"]
    p_ab = pilot["mit_incart"]["frozen"]["alert_blocks"]
    ho = holdout_pilot

    primary = rel_red >= gates_cfg["primary_relative_reduction_min"]
    gA = (b_auc - p_auc) <= gates_cfg["guardrail_auc_max_drop"]
    gB = ((b_ev - p_ev) <= gates_cfg["guardrail_event_recall_max_drop"]
          and p_ab >= gates_cfg["guardrail_alert_blocks_min"])
    gC = (ho["mean_prob"] < gates_cfg["real_holdout_mean_prob_max"]
          and ho["frac_gt_0.5"] <= gates_cfg["real_holdout_frac_gt_0.5_max"])
    if primary and gA and gB and gC:
        verdict = "PASS"
    elif primary:
        verdict = "PARTIAL"
    else:
        verdict = "UNPROVEN"

    gates = {
        "primary_pooled_norm_frac_gt_0.95": {
            "baseline": b_frac, "pilot": p_frac,
            "relative_reduction": rel_red, "passed": primary,
            "criterion": f">={gates_cfg['primary_relative_reduction_min']} relative reduction"},
        "guardrail_A_mit_auc": {
            "baseline": b_auc, "pilot": p_auc, "drop": b_auc - p_auc,
            "passed": gA,
            "criterion": f"drop<={gates_cfg['guardrail_auc_max_drop']}"},
        "guardrail_B_frozen_event_recall": {
            "baseline_event_recall": b_ev, "pilot_event_recall": p_ev,
            "pilot_alert_blocks": p_ab, "passed": gB,
            "criterion": f"drop<={gates_cfg['guardrail_event_recall_max_drop']} "
                         f"and alert_blocks>={gates_cfg['guardrail_alert_blocks_min']}"},
        "guardrail_C_real_holdout": {
            "pilot": ho, "passed": gC,
            "criterion": f"mean_prob<{gates_cfg['real_holdout_mean_prob_max']} "
                         f"and frac_gt_0.5<={gates_cfg['real_holdout_frac_gt_0.5_max']}"},
    }
    return gates, verdict


def selfcheck_base(base):
    band_mi = base["mit_only"]["band_0.005_count"]
    band_in = base["incart_only"]["band_0.005_count"]
    mit_tol = band_mi / base["mit_only"]["n_norm"] + 1e-6
    inc_tol = band_in / base["incart_only"]["n_norm"] + 1e-6
    checks = {
        "mit_frac_gt_0.95_vs_p1": (
            base["mit_only"]["frac_gt_0.95"],
            abs(base["mit_only"]["frac_gt_0.95"]
                - BASELINE_REF["p1_mit_frac_gt_0.95"]) <= mit_tol,
            f"tol={mit_tol:.6f}"),
        "incart_frac_gt_0.95_vs_p1": (
            base["incart_only"]["frac_gt_0.95"],
            abs(base["incart_only"]["frac_gt_0.95"]
                - BASELINE_REF["p1_incart_frac_gt_0.95"]) <= inc_tol,
            f"tol={inc_tol:.6f}"),
        "mit_auc_vs_p1": (
            base["mit_incart"]["auc"],
            abs(base["mit_incart"]["auc"] - BASELINE_REF["p1_mit_auc"]) < 1e-3,
            "tol=1e-3"),
        "ptb_auc_vs_p1": (
            base["ptb"]["auc"],
            abs(base["ptb"]["auc"] - BASELINE_REF["p1_ptb_auc"]) < 1e-3,
            "tol=1e-3"),
        "frozen_event_recall_vs_sweep": (
            base["mit_incart"]["frozen"]["event_recall"],
            abs(base["mit_incart"]["frozen"]["event_recall"]
                - BASELINE_REF["sweep_mit_event_recall"]) < 1e-3,
            "tol=1e-3"),
        "frozen_alert_blocks_vs_sweep": (
            base["mit_incart"]["frozen"]["alert_blocks"],
            abs(base["mit_incart"]["frozen"]["alert_blocks"]
                - BASELINE_REF["sweep_mit_alert_blocks"]) <= 2,
            "tol=2"),
    }
    ok = all(v[1] for v in checks.values())
    return checks, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="path relative to models/deploy_match:label")
    ap.add_argument("--prereg", default="distill_pilot_prereg.json")
    ap.add_argument("--out", default="distill_pilot_multiseed_eval.json")
    ap.add_argument("--skip-selfcheck", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    prereg = load_prereg(args.prereg)
    print(f"[DPM] prereg: {args.prereg} "
          f"gates_primary_min={prereg['gates']['primary_relative_reduction_min']} "
          f"gates_event_recall_drop_max="
          f"{prereg['gates']['guardrail_event_recall_max_drop']}", flush=True)

    d = np.load(TARGETS_NPZ)
    x_ho = np.asarray(d["x_real_holdout"], dtype=np.float32)
    predict_base = make_predictor("h5", MODEL_BASE)
    sets_base, _ = val_sequences(predict_base)
    base = evaluate_model(predict_base, sets_base)

    checks, ok = selfcheck_base(base)
    for k, (v, good, tol) in checks.items():
        print(f"[DPM] selfcheck {k}: {v} ({tol}) "
              f"{'PASS' if good else 'FAIL'}", flush=True)
    if not ok and not args.skip_selfcheck:
        raise SystemExit("[DPM] base selfcheck failed; no adjudication")

    holdout_base_probs = predict_base(x_ho)
    holdout_base = {"mean_prob": float(holdout_base_probs.mean()),
                    "frac_gt_0.5": float((holdout_base_probs > 0.5).mean()),
                    "n": int(len(x_ho))}

    rows = []
    per_model_jsons = []
    for spec in args.model:
        if ":" not in spec:
            raise SystemExit(f"[DPM] --model expects path:label, got {spec}")
        model_rel, label = spec.split(":", 1)
        model_path = CACHE / model_rel
        print(f"[DPM] evaluating {label} <- {model_path}", flush=True)
        predict_pilot = make_predictor("h5", model_path)
        sets_pilot, _ = val_sequences(predict_pilot)
        pilot = evaluate_model(predict_pilot, sets_pilot)
        p_ho = predict_pilot(x_ho)
        holdout_pilot = {"mean_prob": float(p_ho.mean()),
                         "frac_gt_0.5": float((p_ho > 0.5).mean()),
                         "n": int(len(x_ho))}
        gates, verdict = compute_gates(base, pilot, holdout_pilot, prereg)
        row = {
            "label": label,
            "model": str(model_rel),
            "verdict": verdict,
            "gates": gates,
            "secondary": {
                "alert_blocks_ratio_pilot_over_baseline":
                    pilot["mit_incart"]["frozen"]["alert_blocks"]
                    / base["mit_incart"]["frozen"]["alert_blocks"],
                "p1_caliber_summary": {
                    "baseline": summarize_per_record(
                        base["mit_incart"]["p1_caliber_per_record"]),
                    "pilot": summarize_per_record(
                        pilot["mit_incart"]["p1_caliber_per_record"]),
                },
            },
        }
        rows.append(row)
        per_model = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "label": label,
            "model": str(model_rel),
            "purpose": "TH106+ per-seed distill-pilot evaluation (new script)",
            "frozen_operating_point": FROZEN,
            "prereg": args.prereg,
            "gates": gates,
            "verdict": verdict,
            "baseline_metrics": copy.deepcopy(base),
            "pilot_metrics": copy.deepcopy(pilot),
            "real_afe_holdout": {"baseline": holdout_base,
                                 "pilot": holdout_pilot},
            "verdict_semantics": prereg["verdict_semantics"],
        }
        # Drop heavy per-record tables from the per-model JSON; the full
        # secondary summary is still in `rows` above.
        per_model_path = CACHE / f"distill_pilot_eval_{label}.json"
        per_model["baseline_metrics"]["mit_incart"].pop("p1_caliber_per_record", None)
        per_model["baseline_metrics"]["ptb"].pop("p1_caliber_per_record", None)
        per_model["pilot_metrics"]["mit_incart"].pop("p1_caliber_per_record", None)
        per_model["pilot_metrics"]["ptb"].pop("p1_caliber_per_record", None)
        per_model_path.write_text(json.dumps(per_model, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        per_model_jsons.append(str(per_model_path.relative_to(CACHE)))
        for k, g in gates.items():
            print(f"[DPM] {label} gate {k}: passed={g['passed']}", flush=True)
        print(f"[DPM] {label} verdict={verdict}", flush=True)
        print(f"[DPM] {label} primary rel_red="
              f"{gates['primary_pooled_norm_frac_gt_0.95']['relative_reduction']:.4f} "
              f"A_drop={gates['guardrail_A_mit_auc']['drop']:.4f} "
              f"B_ev={gates['guardrail_B_frozen_event_recall']['pilot_event_recall']:.4f} "
              f"B_blocks={gates['guardrail_B_frozen_event_recall']['pilot_alert_blocks']} "
              f"C_mean={holdout_pilot['mean_prob']:.4f}",
              flush=True)

    # Summary table + mean/std.
    summary = {"seeds": [r["label"] for r in rows]}
    key_paths = [
        ("primary_rel_red", "gates", "primary_pooled_norm_frac_gt_0.95", "relative_reduction"),
        ("pilot_frac_gt_0.95", "gates", "primary_pooled_norm_frac_gt_0.95", "pilot"),
        ("pilot_auc", "gates", "guardrail_A_mit_auc", "pilot"),
        ("auc_drop", "gates", "guardrail_A_mit_auc", "drop"),
        ("pilot_event_recall", "gates", "guardrail_B_frozen_event_recall", "pilot_event_recall"),
        ("baseline_event_recall", "gates", "guardrail_B_frozen_event_recall",
         "baseline_event_recall"),
        ("pilot_alert_blocks", "gates", "guardrail_B_frozen_event_recall", "pilot_alert_blocks"),
        ("pilot_holdout_mean", "gates", "guardrail_C_real_holdout", "pilot", "mean_prob"),
    ]
    for key, *path in key_paths:
        vals = []
        for r in rows:
            node = r
            for p in path:
                node = node[p]
            vals.append(float(node))
        summary[key] = {
            "values": vals,
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        }

    # Actual event-recall drop (baseline - pilot) for convenience.
    ev_drops = []
    for r in rows:
        gb = r["gates"]["guardrail_B_frozen_event_recall"]
        ev_drops.append(float(gb["baseline_event_recall"] - gb["pilot_event_recall"]))
    summary["event_recall_drop"] = {
        "values": ev_drops,
        "mean": float(np.mean(ev_drops)) if ev_drops else None,
        "std": float(np.std(ev_drops, ddof=1)) if len(ev_drops) > 1 else None,
    }

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "TH106+ multi-seed / variant robustness check of distill pilot",
        "prereg": args.prereg,
        "zero_test_contact": True,
        "frozen_operating_point": FROZEN,
        "baseline_model": str(MODEL_BASE.relative_to(BASE)),
        "selfcheck": {k: {"value": v[0], "ok": v[1], "tol": v[2]}
                      for k, v in checks.items()},
        "rows": rows,
        "summary": summary,
        "per_model_files": per_model_jsons,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = CACHE / args.out
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[DPM] saved {out_path} + per-model files: {per_model_jsons}", flush=True)


if __name__ == "__main__":
    main()
