#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_v6c_dataset.py -- v6c: MIT/INCART-only balanced 32k dataset.

Uses only the MIT/INCART portions of the v6a mined pool, balanced 16k hard
normal + 16k abnormal, to avoid the domain shift from SVDB/AFDB/VFDB and the
extreme class imbalance of the original v3-A train.
"""
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
V6A = CACHE / "v6a_train_100k.npz"
OUT = CACHE / "v6c_train_mitincart_balanced32k.npz"
SEED = 42


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    d = np.load(V6A)
    x = np.asarray(d["x"], dtype=np.float32)
    y = np.asarray(d["y"], dtype=np.int32)
    src = np.asarray(d["source"], dtype=np.int32)
    rec = np.asarray(d["record_id"], dtype=np.int32)
    loc = np.asarray(d["local_beat_index"], dtype=np.int32)
    score = np.asarray(d["hard_normal_score"], dtype=np.float32)

    keep = (src == 0) | (src == 1)
    x = x[keep]; y = y[keep]; src = src[keep]; rec = rec[keep]; loc = loc[keep]; score = score[keep]
    norm_idx = np.flatnonzero(y == 0)
    abn_idx = np.flatnonzero(y == 1)
    n = min(len(norm_idx), len(abn_idx))
    norm_pick = rng.choice(norm_idx, n, replace=False)
    abn_pick = rng.choice(abn_idx, n, replace=False)
    idx = np.concatenate([norm_pick, abn_pick])
    rng.shuffle(idx)

    x_out = x[idx].astype(np.float32)
    y_out = y[idx].astype(np.int32)
    src_out = src[idx].astype(np.int32)
    rec_out = rec[idx].astype(np.int32)
    loc_out = loc[idx].astype(np.int32)
    score_out = score[idx].astype(np.float32)

    np.savez_compressed(OUT, x=x_out, y=y_out, source=src_out,
                        record_id=rec_out, local_beat_index=loc_out,
                        hard_normal_score=score_out)
    print(f"[V6C] saved {OUT.name}: x={x_out.shape} n={len(y_out)} "
          f"norm={int((y_out==0).sum())} abn={int((y_out==1).sum())} "
          f"elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
