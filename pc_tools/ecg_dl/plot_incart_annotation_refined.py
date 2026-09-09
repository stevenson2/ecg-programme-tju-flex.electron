#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_incart_annotation_refined.py -- compare raw annotation positions,
refined R-peak positions, and the 250-point beat windows used by the corrected
INCART Lead II dataset.
"""
import sys
from pathlib import Path
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import ecg_refs as _REFS  # noqa: E402


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TARGET_FS, BEAT_WINDOW_SAMPLES
from eval_deploy_match import corrected_deployment_chain

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
INCART_DIR = _REFS.INCART_DIR
OUT = CACHE / "incart_annotation_refined_preview.png"
RECORDS = ["I01", "I02"]
WIN_S = 5.0
SEARCH = 14


def refine(ann_samples, fs, filtered):
    refined = []
    for s in ann_samples:
        idx = int(s * TARGET_FS / fs)
        if idx - SEARCH < 0 or idx + SEARCH + 1 > len(filtered):
            refined.append(idx)
            continue
        seg = filtered[idx - SEARCH:idx + SEARCH + 1]
        rel = int(np.argmax(seg))
        refined.append(idx - SEARCH + rel)
    return np.asarray(refined, dtype=np.int64)


def main():
    fig, axes = plt.subplots(len(RECORDS), 1, figsize=(18, 4.5 * len(RECORDS)), squeeze=False)
    for row, rec_name in enumerate(RECORDS):
        rec = wfdb.rdrecord(str(INCART_DIR / rec_name))
        ann = wfdb.rdann(str(INCART_DIR / rec_name), "atr")
        fs = int(rec.fs)
        lead2 = rec.p_signal[:, 1].astype(np.float64)
        filtered = corrected_deployment_chain(lead2, fs)
        start_f = int(10.0 * TARGET_FS)
        end_f = int((10.0 + WIN_S) * TARGET_FS)
        seg = filtered[start_f:end_f]
        t = np.arange(len(seg)) / TARGET_FS

        # Raw annotation positions in this filtered 250 Hz segment.
        raw_pos = []
        for s in ann.sample:
            time_s = s / fs
            if 10.0 <= time_s < 10.0 + WIN_S:
                raw_pos.append(int(time_s * TARGET_FS) - start_f)
        raw_pos = np.asarray(raw_pos, dtype=np.int64)

        # Refined R peaks in same segment.
        refined = refine(np.asarray(ann.sample), fs, filtered)
        ref_pos = np.asarray([r - start_f for r in refined
                              if 10.0 * TARGET_FS <= r < (10.0 + WIN_S) * TARGET_FS],
                             dtype=np.int64)

        ax = axes[row][0]
        ax.plot(t, seg, linewidth=1.0, label=f"{rec_name} filtered Lead II")
        for p in raw_pos:
            ax.axvline(t[p], color="red", linestyle="--", alpha=0.7, linewidth=1.2)
        for p in ref_pos:
            ax.axvline(t[p], color="green", linestyle="-", alpha=0.9, linewidth=1.5)
        # Draw a few 250-pt beat windows around refined peaks (first 3).
        half = BEAT_WINDOW_SAMPLES // 2
        for p in ref_pos[:3]:
            lo = max(0, p - half)
            hi = min(len(seg), p + half)
            ax.axvspan(t[lo], t[hi], color="green", alpha=0.08)
        ax.set_title(f"{rec_name}: red = raw .atr, green = refined R peak, "
                     f"shaded = 250-pt beat window", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f"[ILR] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
