#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mine_hard_normal_clean.py -- clean hard-normal mining after data-quality fix.

Uses only:
  0: MIT-BIH      (MLII)
  1: INCART Lead II (rebuilt from p_signal[:,1])
  2: PTB          (ii)
  3: SVDB         (channel 0 = MLII per project record)
Excludes AFDB/VFDB/CUDB because their Lead II identity is not confirmed.

Also applies a beat-completeness filter:
  - the max-absolute-deviation peak must be near the 250-pt window center
    (within CENTER_TOL samples).

No training; no test contact.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.split_guard import get_guard
from eval_clean_test import make_predictor

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
ECG_DATA = Path("/home/devcontainers/ecg_data")
MODEL_BASE = BASE / "models" / "best_resnet_large_clean_baseline_v3.h5"
OUT_SCORES = CACHE / "beat_pool_clean_normal_scores.npy"
OUT_META = CACHE / "beat_pool_clean_normal_meta.npz"
OUT_AUDIT = CACHE / "hard_normal_clean_audit.json"
CENTER = 125
CENTER_TOL = 40


def load_source(code):
    if code == 0:
        tag = "mit_bih"
        b = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy", mmap_mode="r")
        l = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_labels.npy", mmap_mode="r")
        r = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_record_ids.npy", mmap_mode="r")
        mask = np.asarray(get_guard(tag).train_mask, dtype=bool)
        return b, l, r, mask
    if code == 1:
        # INCART Lead II rebuilt arrays (R-peak-refined). Use record-level
        # train mask because the refined beat count may differ by a few beats
        # from the legacy arrays.
        d = np.load(CACHE / "incart_lead2_processed_deploy_causal.npz")
        b, l, r = d["beats"], d["labels"], d["record_ids"]
        train_rids = np.asarray(get_guard("incart").train_record_ids(), dtype=np.int64)
        mask = np.isin(np.asarray(r, dtype=np.int64), train_rids)
        return b, l, r, mask
    if code == 2:
        tag = "ptb"
        b = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy", mmap_mode="r")
        l = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_labels.npy", mmap_mode="r")
        r = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_record_ids.npy", mmap_mode="r")
        mask = np.asarray(get_guard(tag).train_mask, dtype=bool)
        return b, l, r, mask
    if code == 3:
        tag = "svdb"
        b = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy", mmap_mode="r")
        l = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_labels.npy", mmap_mode="r")
        r = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_record_ids.npy", mmap_mode="r")
        mask = np.asarray(get_guard(tag).train_mask, dtype=bool)
        return b, l, r, mask
    raise ValueError(code)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    predict = make_predictor("h5", MODEL_BASE)

    all_scores, all_source, all_rec, all_local, all_offset, all_complete = [], [], [], [], [], []
    per_source = []

    for code, name in [(0, "MIT-BIH"), (1, "INCART-LeadII"), (2, "PTB"), (3, "SVDB")]:
        b, l, r, mask = load_source(code)
        l = np.asarray(l)
        r = np.asarray(r)
        idx = np.flatnonzero(mask & (l == 0))
        if len(idx) == 0:
            per_source.append({"name": name, "source_code": code, "n_normal": 0,
                               "n_complete": 0, "n_hard_complete": 0})
            continue
        X = np.asarray(b)[idx].astype(np.float32)
        scores = []
        for i0 in range(0, len(X), 512):
            scores.append(predict(X[i0:i0 + 512]))
        scores = np.concatenate(scores).astype(np.float32)

        # Completeness: peak of |deviation from mean| near window center.
        Xf = X.astype(np.float64)
        centered = Xf - Xf.mean(axis=1, keepdims=True)
        peak_pos = np.abs(centered).argmax(axis=1)
        offset = np.abs(peak_pos - CENTER)
        complete = offset <= CENTER_TOL

        all_scores.append(scores)
        all_source.append(np.full(len(scores), code, dtype=np.int32))
        all_rec.append(r[idx].astype(np.int32))
        all_local.append(idx.astype(np.int32))
        all_offset.append(offset.astype(np.int32))
        all_complete.append(complete.astype(np.int32))

        per_source.append({
            "name": name,
            "source_code": code,
            "n_normal": int(len(scores)),
            "n_complete": int(complete.sum()),
            "n_score_gt_0.5": int((scores > 0.5).sum()),
            "n_score_gt_0.95": int((scores > 0.95).sum()),
            "n_hard_gt_0.5_complete": int(((scores > 0.5) & complete).sum()),
            "n_hard_gt_0.95_complete": int(((scores > 0.95) & complete).sum()),
            "frac_gt_0.95": float((scores > 0.95).mean()),
        })
        print(f"[HNC] {name}: normal={len(scores)} complete={int(complete.sum())} "
              f"hard>0.95={int((scores>0.95).sum())} "
              f"hard>0.95&complete={int(((scores>0.95)&complete).sum())}", flush=True)

    scores_all = np.concatenate(all_scores).astype(np.float32)
    source_all = np.concatenate(all_source).astype(np.int32)
    rec_all = np.concatenate(all_rec).astype(np.int32)
    local_all = np.concatenate(all_local).astype(np.int32)
    offset_all = np.concatenate(all_offset).astype(np.int32)
    complete_all = np.concatenate(all_complete).astype(np.int32)

    np.save(OUT_SCORES, scores_all)
    np.savez_compressed(OUT_META, source=source_all, record_id=rec_all,
                        local_beat_index=local_all, peak_offset=offset_all,
                        complete=complete_all)

    overall = {
        "n_train_normal_total": int(len(scores_all)),
        "n_complete": int(complete_all.sum()),
        "n_gt_0.5": int((scores_all > 0.5).sum()),
        "n_gt_0.95": int((scores_all > 0.95).sum()),
        "n_gt_0.5_complete": int(((scores_all > 0.5) & (complete_all == 1)).sum()),
        "n_gt_0.95_complete": int(((scores_all > 0.95) & (complete_all == 1)).sum()),
        "frac_gt_0.5": float((scores_all > 0.5).mean()),
        "frac_gt_0.95": float((scores_all > 0.95).mean()),
        "mean": float(scores_all.mean()),
        "p95": float(np.quantile(scores_all, 0.95)),
    }
    audit = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Clean hard-normal mining (MIT/INCART-LeadII/PTB/SVDB, completeness-filtered)",
        "model": str(MODEL_BASE.relative_to(BASE)),
        "center_tol": CENTER_TOL,
        "overall": overall,
        "per_source": per_source,
        "outputs": {
            "scores": str(OUT_SCORES.relative_to(BASE)),
            "meta": str(OUT_META.relative_to(BASE)),
            "audit": str(OUT_AUDIT.relative_to(BASE)),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[HNC] saved {OUT_SCORES.name}, {OUT_META.name}, {OUT_AUDIT.name}", flush=True)
    print(f"[HNC] totals: normal={len(scores_all)} complete={int(complete_all.sum())} "
          f"hard>0.95={int((scores_all>0.95).sum())} "
          f"hard>0.95&complete={int(((scores_all>0.95)&(complete_all==1)).sum())}",
          flush=True)


if __name__ == "__main__":
    main()
