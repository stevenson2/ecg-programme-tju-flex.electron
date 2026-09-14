# -*- coding: utf-8 -*-
"""eval_gate_model_r18.py — A1 fallback gate evaluation (frozen prereg R18).

- Threshold is selected on the validation split only:
    among thresholds with E_A<=0.10, maximize Sn_A;
    if none, choose min E_A then max Sn_A.
- Test is evaluated once at that frozen threshold.
- Outputs JSON with AUC / Sn_A / E_A for MIT+INCART and PTB test.

Usage:
  python3 eval_gate_model_r18.py --model runs/r18_gate/r18_gate_..._seed42.h5 \
      --seed 42 --out models/gate/r18_gate_eval_seed42.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data.dataset as dataset
from data.patient_split import (
    build_mit_patient_map, build_incart_patient_map, build_ptb_patient_map,
    patient_level_split,
)


def _select_threshold(y_val, p_val):
    """Select threshold on validation only per frozen prereg."""
    best = None
    fallback = None
    for thr in np.arange(0.02, 0.99, 0.01):
        pred = p_val >= thr
        abn = int(y_val.sum())
        norm = len(y_val) - abn
        sn = float((pred & (y_val == 1)).sum() / max(abn, 1))
        ea = float((pred & (y_val == 0)).sum() / max(norm, 1))
        if ea <= 0.10:
            key = (sn, -ea)
            if best is None or key > best[0]:
                best = (key, float(thr), sn, ea)
        key = (-ea, sn)
        if fallback is None or key > fallback[0]:
            fallback = (key, float(thr), sn, ea)
    if best is not None:
        return {"theta": round(best[1], 3), "val_Sn_A": round(best[2], 4),
                "val_E_A": round(best[3], 4), "gate_passed": True}
    return {"theta": round(fallback[1], 3), "val_Sn_A": round(fallback[2], 4),
            "val_E_A": round(fallback[3], 4), "gate_passed": False}


def _metrics(y, p, thr):
    pred = p >= thr
    abn = int(y.sum())
    norm = len(y) - abn
    return {
        "n": int(len(y)),
        "n_abn": abn,
        "n_normal": norm,
        "auc": round(float(roc_auc_score(y, p)), 4),
        "theta": round(float(thr), 3),
        "Sn_A": round(float((pred & (y == 1)).sum() / max(abn, 1)), 4),
        "E_A": round(float((pred & (y == 0)).sum() / max(norm, 1)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    dataset.set_npz_suffix("_deploy")

    # Validation split = MIT+INCART patient-level seed 42 (no PTB beat test leak).
    a = dataset.load_mit_incart_merged()
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat
                 for rid, pat in build_incart_patient_map().items()})
    tr_m, va_m, te_m, pstats = patient_level_split(a["record_ids"], pmap)
    x_val = a["beats"][va_m].astype(np.float32)[..., np.newaxis]
    y_val = a["labels"][va_m].astype(np.int32)
    x_test = a["beats"][te_m].astype(np.float32)[..., np.newaxis]
    y_test = a["labels"][te_m].astype(np.int32)

    p_val = model.predict(x_val, batch_size=512, verbose=0)[:, 1]
    sel = _select_threshold(y_val, p_val)

    ptb = dataset.load_ptb_data()
    _, _, te_m_ptb, _ = patient_level_split(ptb["record_ids"],
                                            build_ptb_patient_map())
    x_ptb = ptb["beats"][te_m_ptb].astype(np.float32)[..., np.newaxis]
    y_ptb = ptb["labels"][te_m_ptb].astype(np.int32)
    p_ptb = model.predict(x_ptb, batch_size=512, verbose=0)[:, 1]

    out = {
        "script": "eval_gate_model_r18.py",
        "model": str(args.model),
        "seed": args.seed,
        "frozen_prereg": "runs/R18_A1_GATE_PREREG.md",
        "threshold_selection": sel,
        "patient_stats": pstats,
        "test_mit_incart": _metrics(y_test, model.predict(
            x_test, batch_size=512, verbose=0)[:, 1], sel["theta"]),
        "test_ptb": _metrics(y_ptb, p_ptb, sel["theta"]),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))
    print("Saved", out_path)


if __name__ == "__main__":
    main()