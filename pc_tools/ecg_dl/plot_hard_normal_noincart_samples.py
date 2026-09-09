#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_hard_normal_noincart_samples.py -- preview hard-normal samples after
excluding INCART.  Sources: MIT-BIH, PTB, SVDB; score>0.95; completeness pass.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
ECG_DATA = Path("/home/devcontainers/ecg_data")
OUT = CACHE / "hard_normal_noincart_samples_preview.png"


def load_source(code):
    if code == 0:
        b = np.load(ECG_DATA / "mit_bih_processed_deploy_causal_beats.npy", mmap_mode="r")
        return b
    if code == 2:
        b = np.load(ECG_DATA / "ptb_processed_deploy_causal_beats.npy", mmap_mode="r")
        return b
    if code == 3:
        b = np.load(ECG_DATA / "svdb_processed_deploy_causal_beats.npy", mmap_mode="r")
        return b
    raise ValueError(code)


def main():
    scores = np.load(CACHE / "beat_pool_noincart_normal_scores.npy")
    meta = np.load(CACHE / "beat_pool_noincart_normal_meta.npz")
    src = meta["source"]
    loc = meta["local_beat_index"]
    complete = meta["complete"]

    names = {0: "MIT-BIH", 2: "PTB", 3: "SVDB"}
    cand = (scores > 0.95) & (complete == 1)
    fig, axes = plt.subplots(3, 6, figsize=(16, 6), squeeze=False)
    cached = {}
    for row, code in enumerate([0, 2, 3]):
        idxs = np.flatnonzero(cand & (src == code))
        if code not in cached:
            cached[code] = load_source(code)
        beats = cached[code]
        for col in range(6):
            ax = axes[row][col]
            if col < len(idxs):
                gi = idxs[col]
                ax.plot(np.asarray(beats[int(loc[gi])], dtype=np.float32), linewidth=0.8)
                ax.set_title(f"{names[code]}\nscore={scores[gi]:.3f}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"[NOINC] saved preview: {OUT}", flush=True)


if __name__ == "__main__":
    main()
