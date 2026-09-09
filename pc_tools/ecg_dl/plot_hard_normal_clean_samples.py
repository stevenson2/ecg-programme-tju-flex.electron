#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_hard_normal_clean_samples.py -- preview clean hard-normal samples.

Reads beat_pool_clean_normal_scores/meta and shows only:
  - sources MIT/INCART-LeadII/PTB/SVDB
  - score > 0.95
  - completeness filter passed (peak within CENTER_TOL of 250-pt center)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.split_guard import get_guard

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
ECG_DATA = Path("/home/devcontainers/ecg_data")
OUT = CACHE / "hard_normal_clean_samples_preview.png"
CENTER_TOL = 40


def load_source(code):
    if code == 0:
        tag = "mit_bih"
        b = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy", mmap_mode="r")
        return b
    if code == 1:
        d = np.load(CACHE / "incart_lead2_processed_deploy_causal.npz")
        return d["beats"]
    if code == 2:
        tag = "ptb"
        b = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy", mmap_mode="r")
        return b
    if code == 3:
        tag = "svdb"
        b = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy", mmap_mode="r")
        return b
    raise ValueError(code)


def main():
    scores = np.load(CACHE / "beat_pool_clean_normal_scores.npy")
    meta = np.load(CACHE / "beat_pool_clean_normal_meta.npz")
    src = meta["source"]
    rec = meta["record_id"]
    loc = meta["local_beat_index"]
    complete = meta["complete"]

    names = {0: "MIT-BIH", 1: "INCART-LeadII", 2: "PTB", 3: "SVDB"}
    cand = (scores > 0.95) & (complete == 1)

    fig, axes = plt.subplots(4, 6, figsize=(16, 8), squeeze=False)
    cached = {}
    for row, code in enumerate([0, 1, 2, 3]):
        idxs = np.flatnonzero(cand & (src == code))
        if len(idxs) == 0:
            continue
        if code not in cached:
            cached[code] = load_source(code)
        beats = cached[code]
        for col in range(6):
            ax = axes[row][col]
            if col < len(idxs):
                gi = idxs[col]
                beat = np.asarray(beats[int(loc[gi])], dtype=np.float32)
                ax.plot(beat, linewidth=0.8)
                ax.set_title(f"{names[code]}\nscore={scores[gi]:.3f}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"[HNC] saved clean preview: {OUT}", flush=True)


if __name__ == "__main__":
    main()
