#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_hard_normal_samples.py -- preview sampled hard-normal beats.

Reads hard_normal_quality_audit.json and renders one figure per source (or a
single combined grid) so a human can judge whether the "normal" labels look
plausible.  No training.
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
from data.patient_split import patient_level_split, SEED as SPLIT_SEED

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
ECG_DATA = Path("/home/devcontainers/ecg_data")
QUALITY = json.loads((CACHE / "hard_normal_quality_audit.json").read_text(encoding="utf-8"))
OUT = CACHE / "hard_normal_samples_preview.png"


def load_source_arrays(code):
    if code == 0:
        tag = "mit_bih"
        b = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy", mmap_mode="r")
        return b
    if code == 1:
        tag = "incart"
        b = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy", mmap_mode="r")
        return b
    if code == 2:
        tag = "ptb"
        b = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy", mmap_mode="r")
        return b
    if code == 3:
        tag = "svdb"
        b = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy", mmap_mode="r")
        return b
    if code == 4:
        d = np.load(CACHE / "vfdb_processed_deploy_causal.npz")
        return d["beats"]
    if code == 5:
        d = np.load(CACHE / "afdb_processed_deploy_causal.npz")
        return d["beats"]
    raise ValueError(code)


def main():
    sources = [s for s in QUALITY["per_source"] if s["n_sampled"] > 0]
    n_rows = len(sources)
    n_cols = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2.2 * n_rows), squeeze=False)
    cached = {}
    for i, src in enumerate(sources):
        code = src["source_code"]
        if code not in cached:
            cached[code] = load_source_arrays(code)
        beats = cached[code]
        row_axes = axes[i]
        for j in range(n_cols):
            ax = row_axes[j]
            if j < len(src["samples"]):
                loc = src["samples"][j]["local_beat_index"]
                beat = np.asarray(beats[loc], dtype=np.float32)
                ax.plot(beat, linewidth=0.8)
                ax.set_title(f"{src['name']}\nscore={src['samples'][j]['v3a_score']:.3f}",
                             fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"[HNQ] saved preview: {OUT}", flush=True)


if __name__ == "__main__":
    main()
