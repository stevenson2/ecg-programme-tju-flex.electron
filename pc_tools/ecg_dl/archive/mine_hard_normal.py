#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mine_hard_normal.py -- Step 2: score all train-safe normal beats with v3 A
and audit the high-confidence false-alarm ("hard normal") pool.

Parses the same train-safe sources as mine_beat_pool.py, keeps only
train-mask normal beats, runs v3 A inference, and writes:
  - beat_pool_normal_scores.npy
  - beat_pool_normal_meta.npz
  - hard_normal_audit.json

No test/val contact. No training.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.split_guard import get_guard
from data.patient_split import patient_level_split, SEED as SPLIT_SEED
from eval_clean_test import make_predictor

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
ECG_DATA = Path("/home/devcontainers/ecg_data")
MODEL_BASE = BASE / "models" / "best_resnet_large_clean_baseline_v3.h5"
OUT_SCORES = CACHE / "beat_pool_normal_scores.npy"
OUT_META = CACHE / "beat_pool_normal_meta.npz"
OUT_AUDIT = CACHE / "hard_normal_audit.json"


def split_mask_for_source(name, beats_path, labels_path, rids_path):
    b = np.load(beats_path, mmap_mode="r")
    l = np.load(labels_path, mmap_mode="r")
    r = np.load(rids_path, mmap_mode="r")
    if name == "MIT-BIH":
        tag = "mit_bih"
    elif name == "INCART":
        tag = "incart"
    elif name == "PTB":
        tag = "ptb"
    elif name == "SVDB":
        tag = "svdb"
    else:
        raise ValueError(name)
    mask = np.asarray(get_guard(tag).train_mask, dtype=bool)
    return b, l, r, mask


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    predict = make_predictor("h5", MODEL_BASE)

    # Load split audit for VFDB/CUDB.
    vfdb_split = json.loads((CACHE / "vfdb_split_audit.json").read_text(encoding="utf-8"))
    vfdb_train_ids = np.asarray(vfdb_split["train_record_ids"], dtype=np.int64)

    # AFDB split.
    a = np.load(CACHE / "afdb_processed_deploy_causal.npz")
    ar = np.asarray(a["record_ids"]).astype(np.int64)
    amap = {int(rid): f"afdb_{int(rid)}" for rid in np.unique(ar)}
    afdb_train_mask = np.asarray(patient_level_split(
        np.asarray(ar, dtype=np.int64), amap, seed=SPLIT_SEED)[0], dtype=bool)

    sources = []

    # 0 MIT, 1 INCART, 2 PTB, 3 SVDB
    for code, name, tag in [
        (0, "MIT-BIH", "mit_bih"),
        (1, "INCART", "incart"),
        (2, "PTB", "ptb"),
        (3, "SVDB", "svdb"),
    ]:
        b, l, r, mask = split_mask_for_source(
            name,
            ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy",
            ECG_DATA / f"{tag}_processed_deploy_causal_labels.npy",
            ECG_DATA / f"{tag}_processed_deploy_causal_record_ids.npy",
        )
        norm_mask = mask & (np.asarray(l) == 0)
        idx = np.flatnonzero(norm_mask)
        sources.append((code, name, b, np.asarray(l), np.asarray(r), idx))

    # VFDB/CUDB
    v = np.load(CACHE / "vfdb_processed_deploy_causal.npz")
    vb, vl, vr = v["beats"], v["labels"], v["record_ids"]
    vm = np.isin(np.asarray(vr, dtype=np.int64), vfdb_train_ids)
    sources.append((4, "VFDB+CUDB", vb, np.asarray(vl), np.asarray(vr),
                    np.flatnonzero(vm & (np.asarray(vl) == 0))))

    # AFDB
    sources.append((5, "AFDB", a["beats"], np.asarray(a["labels"]),
                    ar, np.flatnonzero(afdb_train_mask & (np.asarray(a["labels"]) == 0))))

    all_scores = []
    all_source = []
    all_rids = []
    all_local_idx = []
    per_source = []

    for code, name, beats, labels, rids, idx in sources:
        if len(idx) == 0:
            per_source.append({"name": name, "source_code": code, "n_normal": 0,
                               "frac_gt_0.5": None, "frac_gt_0.95": None})
            continue
        # Predict in batches (512) on normal beats only.
        X = np.asarray(beats)[idx].astype(np.float32)
        scores = []
        for i0 in range(0, len(X), 512):
            p = predict(X[i0:i0 + 512])
            scores.append(p)
        scores = np.concatenate(scores).astype(np.float32)
        all_scores.append(scores)
        all_source.append(np.full(len(scores), code, dtype=np.int32))
        all_rids.append(np.asarray(rids)[idx].astype(np.int32))
        all_local_idx.append(idx.astype(np.int32))
        per_source.append({
            "name": name,
            "source_code": code,
            "n_normal": int(len(scores)),
            "mean": float(scores.mean()),
            "p50": float(np.quantile(scores, 0.50)),
            "p90": float(np.quantile(scores, 0.90)),
            "p95": float(np.quantile(scores, 0.95)),
            "p99": float(np.quantile(scores, 0.99)),
            "frac_gt_0.5": float((scores > 0.5).mean()),
            "frac_gt_0.9": float((scores > 0.9).mean()),
            "frac_gt_0.95": float((scores > 0.95).mean()),
            "n_gt_0.5": int((scores > 0.5).sum()),
            "n_gt_0.95": int((scores > 0.95).sum()),
        })
        print(f"[HN] {name}: normal={len(scores)} mean={scores.mean():.4f} "
              f"p95={np.quantile(scores,0.95):.4f} frac>0.5={(scores>0.5).mean():.4f} "
              f"frac>0.95={(scores>0.95).mean():.4f}", flush=True)

    scores_all = np.concatenate(all_scores).astype(np.float32)
    source_all = np.concatenate(all_source).astype(np.int32)
    rid_all = np.concatenate(all_rids).astype(np.int32)
    local_idx_all = np.concatenate(all_local_idx).astype(np.int32)

    np.save(OUT_SCORES, scores_all)
    np.savez_compressed(OUT_META, source=source_all, record_id=rid_all,
                        local_beat_index=local_idx_all)

    audit = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "v3 A scoring of all train-safe normal beats (hard-normal mining audit)",
        "model": str(MODEL_BASE.relative_to(BASE)),
        "n_train_normal_total": int(len(scores_all)),
        "overall": {
            "mean": float(scores_all.mean()),
            "p50": float(np.quantile(scores_all, 0.50)),
            "p75": float(np.quantile(scores_all, 0.75)),
            "p90": float(np.quantile(scores_all, 0.90)),
            "p95": float(np.quantile(scores_all, 0.95)),
            "p99": float(np.quantile(scores_all, 0.99)),
            "max": float(scores_all.max()),
            "frac_gt_0.5": float((scores_all > 0.5).mean()),
            "frac_gt_0.9": float((scores_all > 0.9).mean()),
            "frac_gt_0.95": float((scores_all > 0.95).mean()),
            "n_gt_0.5": int((scores_all > 0.5).sum()),
            "n_gt_0.9": int((scores_all > 0.9).sum()),
            "n_gt_0.95": int((scores_all > 0.95).sum()),
        },
        "per_source": per_source,
        "outputs": {
            "scores": str(OUT_SCORES.relative_to(BASE)),
            "meta": str(OUT_META.relative_to(BASE)),
            "audit": str(OUT_AUDIT.relative_to(BASE)),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[HN] saved {OUT_SCORES.name}, {OUT_META.name}, {OUT_AUDIT.name}", flush=True)
    print(f"[HN] totals: normal={len(scores_all)} "
          f"frac>0.5={(scores_all>0.5).mean():.4f} "
          f"n>0.5={int((scores_all>0.5).sum())} "
          f"n>0.95={int((scores_all>0.95).sum())}", flush=True)


if __name__ == "__main__":
    main()
