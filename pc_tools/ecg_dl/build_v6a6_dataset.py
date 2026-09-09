#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_v6a6_dataset.py -- v6a6 dataset: original v3-A train + 10k clean
MIT/INCART hard-normal (no SVDB/PTB).
"""
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
ECG_DATA = Path("/home/devcontainers/ecg_data")
SCORES = CACHE / "beat_pool_clean_normal_scores.npy"
META = CACHE / "beat_pool_clean_normal_meta.npz"
VAL = CACHE / "distill_pilot_teacher_targets.npz"
OUT = CACHE / "v6a6_train_10k_mitincart_hardnorm_anchor.npz"
SEED = 42
QUOTAS = {0: 4000, 1: 6000}


def load_source(code):
    if code == 0:
        b = np.load(ECG_DATA / "mit_bih_processed_deploy_causal_beats.npy", mmap_mode="r")
        return b
    if code == 1:
        d = np.load(CACHE / "incart_lead2_processed_deploy_causal.npz")
        return d["beats"]
    raise ValueError(code)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    scores = np.load(SCORES)
    meta = np.load(META)
    src = meta["source"]
    loc = meta["local_beat_index"]
    complete = meta["complete"]

    selected_global = []
    for code, quota in QUOTAS.items():
        cand = np.flatnonzero((src == code) & (complete == 1) & (scores > 0.95))
        n_use = min(quota, len(cand))
        if n_use <= 0:
            print(f"[V6A6] source {code}: quota={quota} available={len(cand)} -> 0")
            continue
        pick = rng.choice(cand, n_use, replace=False)
        selected_global.append(pick)
        print(f"[V6A6] source {code}: quota={quota} available={len(cand)} -> {n_use}")
    sel = np.concatenate(selected_global)

    xs, srcs, recs, locs, scs = [], [], [], [], []
    for code in sorted(set(src[sel].tolist())):
        m = sel[src[sel] == code]
        b = load_source(code)
        loc_use = loc[m].astype(np.int64)
        xs.append(np.asarray(b)[loc_use].astype(np.float32))
        srcs.append(np.full(len(loc_use), code, dtype=np.int32))
        recs.append(np.zeros(len(loc_use), dtype=np.int32))
        locs.append(loc_use.astype(np.int32))
        scs.append(scores[m].astype(np.float32))

    x_hard = np.concatenate(xs)
    src_hard = np.concatenate(srcs)
    loc_hard = np.concatenate(locs)
    score_hard = np.concatenate(scs)

    v = np.load(VAL)
    x_orig = np.asarray(v["x_train_perm"], dtype=np.float32)
    y_orig = np.asarray(v["y_train"], dtype=np.int32)

    x = np.concatenate([x_hard, x_orig], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(len(x_hard), dtype=np.int32), y_orig], axis=0).astype(np.int32)
    source = np.concatenate([src_hard, np.full(len(y_orig), 7, dtype=np.int32)]).astype(np.int32)
    record = np.zeros(len(y), dtype=np.int32)
    local = np.concatenate([loc_hard, np.arange(len(y_orig), dtype=np.int32)]).astype(np.int32)
    score = np.concatenate([score_hard, np.full(len(y_orig), -1.0, dtype=np.float32)]).astype(np.float32)

    np.savez_compressed(OUT, x=x, y=y, source=source,
                        record_id=record, local_beat_index=local,
                        hard_normal_score=score)
    print(f"[V6A6] saved {OUT.name}: x={x.shape} n={len(y)} "
          f"normal={int((y==0).sum())} abnormal={int((y==1).sum())} "
          f"elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
