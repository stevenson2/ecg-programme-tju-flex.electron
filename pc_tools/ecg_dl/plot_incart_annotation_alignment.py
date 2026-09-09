#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_incart_annotation_alignment.py -- diagnose INCART beat segmentation.

Shows filtered Lead II (corrected_deployment_chain) with INCART .atr annotation
positions overlaid, so we can see whether the annotation samples correspond to
the R peaks (i.e., whether beat-window centering is correct).
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
from config import TARGET_FS
from eval_deploy_match import corrected_deployment_chain

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
INCART_DIR = _REFS.INCART_DIR
OUT = CACHE / "incart_annotation_alignment.png"
RECORDS = ["I01", "I02"]
WIN_S = 5.0


def main():
    fig, axes = plt.subplots(len(RECORDS), 1, figsize=(16, 4 * len(RECORDS)), squeeze=False)
    for row, rec_name in enumerate(RECORDS):
        rec = wfdb.rdrecord(str(INCART_DIR / rec_name))
        ann = wfdb.rdann(str(INCART_DIR / rec_name), "atr")
        fs = int(rec.fs)
        # Take 10-15 s.
        start_raw = int(10.0 * fs)
        end_raw = int((10.0 + WIN_S) * fs)
        lead2 = rec.p_signal[:, 1].astype(np.float64)
        filtered = corrected_deployment_chain(lead2, fs)
        start_f = int(start_raw * TARGET_FS / fs)
        end_f = int((start_raw + TARGET_FS * WIN_S) * TARGET_FS / fs)
        seg = filtered[start_f:end_f]
        t = np.arange(len(seg)) / TARGET_FS

        ax = axes[row][0]
        ax.plot(t, seg, linewidth=0.9, label=f"{rec_name} filtered Lead II")
        # Convert annotations within the window to 250 Hz time.
        for s in ann.sample:
            time_s = s / fs
            if 10.0 <= time_s < 10.0 + WIN_S:
                ax.axvline(time_s - 10.0, color="red", linestyle="--", alpha=0.7)
        ax.set_title(f"{rec_name}: filtered Lead II with .atr annotation marks", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f"[ILA] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
