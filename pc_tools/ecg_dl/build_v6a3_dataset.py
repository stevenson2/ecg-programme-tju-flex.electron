#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_v6a3_dataset.py -- v6a3 dataset: original v3-A train + 50k hard normal.

No external abnormal beats are added; the abnormal distribution stays exactly
the v3-A training distribution.  This tests targeted hard-normal addition as a
false-alarm fix without shifting the abnormal decision boundary.
"""
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
V6A = CACHE / "v6a_train_100k.npz"
VAL = CACHE / "distill_pilot_teacher_targets.npz"
OUT = CACHE / "v6a3_train_hardnorm_anchor.npz"


def main():
    t0 = time.time()
    d = np.load(V6A)
    x_hard = np.asarray(d["x"], dtype=np.float32)
    y_hard = np.asarray(d["y"], dtype=np.int32)
    # 50k normal entries are rows y == 0; their score metadata also in file.
    src_hard = np.asarray(d["source"], dtype=np.int32)
    rec_hard = np.asarray(d["record_id"], dtype=np.int32)
    loc_hard = np.asarray(d["local_beat_index"], dtype=np.int32)
    score_hard = np.asarray(d["hard_normal_score"], dtype=np.float32)
    norm_mask = y_hard == 0
    x_hard = x_hard[norm_mask]
    src_hard = src_hard[norm_mask]
    rec_hard = rec_hard[norm_mask]
    loc_hard = loc_hard[norm_mask]
    score_hard = score_hard[norm_mask]

    v = np.load(VAL)
    x_orig = np.asarray(v["x_train_perm"], dtype=np.float32)
    y_orig = np.asarray(v["y_train"], dtype=np.int32)

    x = np.concatenate([x_hard, x_orig], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(len(x_hard), dtype=np.int32), y_orig], axis=0).astype(np.int32)
    source = np.concatenate([src_hard,
                             np.full(len(y_orig), 7, dtype=np.int32)]).astype(np.int32)
    record = np.concatenate([rec_hard,
                             np.zeros(len(y_orig), dtype=np.int32)]).astype(np.int32)
    local = np.concatenate([loc_hard,
                            np.arange(len(y_orig), dtype=np.int32)]).astype(np.int32)
    score = np.concatenate([score_hard,
                            np.full(len(y_orig), -1.0, dtype=np.float32)]).astype(np.float32)

    np.savez_compressed(OUT, x=x, y=y, source=source,
                        record_id=record, local_beat_index=local,
                        hard_normal_score=score)
    print(f"[V6A3] saved {OUT.name}: x={x.shape} n={len(y)} "
          f"normal={int((y==0).sum())} abnormal={int((y==1).sum())} "
          f"elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
