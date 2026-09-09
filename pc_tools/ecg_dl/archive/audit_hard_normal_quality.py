#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_hard_normal_quality.py -- sample-by-source label-quality audit for
v3-A hard-normal beats (score > 0.95).

For each source, this script:
  - selects up to 200 hard-normal beats,
  - loads the actual beat from the source arrays,
  - records basic waveform statistics,
  - records provenance/annotation note where available,
  - writes a metadata JSON under models/deploy_match/.

No training; no val/test contact.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.split_guard import get_guard
from data.patient_split import patient_level_split, SEED as SPLIT_SEED

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
ECG_DATA = Path("/home/devcontainers/ecg_data")
OUT = CACHE / "hard_normal_quality_audit.json"
SAMPLE_PER_SOURCE = 200
SEED = 42


def load_source_arrays(code):
    if code == 0:
        tag = "mit_bih"
        b = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy", mmap_mode="r")
        l = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_labels.npy", mmap_mode="r")
        r = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_record_ids.npy", mmap_mode="r")
        mask = np.asarray(get_guard(tag).train_mask, dtype=bool)
        return b, l, r, mask
    if code == 1:
        tag = "incart"
        b = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy", mmap_mode="r")
        l = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_labels.npy", mmap_mode="r")
        r = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_record_ids.npy", mmap_mode="r")
        mask = np.asarray(get_guard(tag).train_mask, dtype=bool)
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
    if code == 4:
        d = np.load(CACHE / "vfdb_processed_deploy_causal.npz")
        b, l, r = d["beats"], d["labels"], d["record_ids"]
        split = json.loads((CACHE / "vfdb_split_audit.json").read_text(encoding="utf-8"))
        train_ids = np.asarray(split["train_record_ids"], dtype=np.int64)
        mask = np.isin(np.asarray(r, dtype=np.int64), train_ids)
        return b, l, r, mask
    if code == 5:
        d = np.load(CACHE / "afdb_processed_deploy_causal.npz")
        b, l, r = d["beats"], d["labels"], d["record_ids"]
        r = np.asarray(r, dtype=np.int64)
        pmap = {int(rid): f"afdb_{int(rid)}" for rid in np.unique(r)}
        mask = np.asarray(patient_level_split(r, pmap, seed=SPLIT_SEED)[0], dtype=bool)
        return b, l, r, mask
    raise ValueError(code)


def beat_stats(beat):
    x = np.asarray(beat, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "ptp": float(x.max() - x.min()),
        "rms": float(np.sqrt(np.mean(x**2))),
    }


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    scores = np.load(CACHE / "beat_pool_normal_scores.npy")
    meta = np.load(CACHE / "beat_pool_normal_meta.npz")
    src_all = meta["source"]
    rec_all = meta["record_id"]
    loc_all = meta["local_beat_index"]

    source_names = {0: "MIT-BIH", 1: "INCART", 2: "PTB", 3: "SVDB", 4: "VFDB+CUDB", 5: "AFDB"}
    per_source = []

    for code in range(6):
        cand = np.flatnonzero((src_all == code) & (scores > 0.95))
        n_use = min(SAMPLE_PER_SOURCE, len(cand))
        if n_use == 0:
            per_source.append({"source_code": code, "name": source_names[code],
                               "n_hard_total": int(len(cand)), "n_sampled": 0,
                               "samples": []})
            continue
        pick = rng.choice(cand, n_use, replace=False)
        b, l, r, mask = load_source_arrays(code)
        rows = []
        r_arr = np.asarray(r)
        for gi in pick:
            loc = int(loc_all[gi])
            beat = np.asarray(b[loc], dtype=np.float32)
            rows.append({
                "global_index": int(gi),
                "source": source_names[code],
                "source_code": code,
                "record_id": int(r_arr[loc]),
                "local_beat_index": loc,
                "v3a_score": float(scores[gi]),
                "stats": beat_stats(beat),
            })
        per_source.append({
            "source_code": code,
            "name": source_names[code],
            "n_hard_total": int(len(cand)),
            "n_sampled": len(rows),
            "samples": rows,
        })
        print(f"[HNQ] {source_names[code]}: hard_total={len(cand)} sampled={len(rows)}",
              flush=True)

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Label-quality / sample audit for v3-A hard-normal pool (score>0.95)",
        "sample_per_source": SAMPLE_PER_SOURCE,
        "seed": SEED,
        "per_source": per_source,
        "note": (
            "MIT/INCART/PTB/SVDB labels are AAMI beat-level normal labels; "
            "VFDB/CUDB and AFDB 'normal' are rhythm-segment labels from the "
            "annotation files. This file is for manual waveform/source review, "
            "not a final label decision."
        ),
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[HNQ] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
