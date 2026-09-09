#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_incart_leads.py -- visualize INCART raw multi-lead ECG for lead
identification.

Plots all 12 standard leads for selected INCART records over a short time
window so the user can tell which channel is Lead II.
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

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
INCART_DIR = _REFS.INCART_DIR
OUT = CACHE / "incart_leads_preview.png"
RECORDS = ["I01", "I02", "I10"]
WINDOW_S = 5.0


def main():
    fig, axes = plt.subplots(len(RECORDS), 12, figsize=(24, 3.0 * len(RECORDS)),
                             squeeze=False)
    for row, rec_name in enumerate(RECORDS):
        rec = wfdb.rdrecord(str(INCART_DIR / rec_name))
        fs = int(rec.fs)
        n_win = int(WINDOW_S * fs)
        # Start at 10 s to avoid initial transitions.
        start = int(10.0 * fs)
        seg = rec.p_signal[start:start + n_win]
        t = np.arange(seg.shape[0]) / fs
        for col in range(12):
            ax = axes[row][col]
            lead_name = rec.sig_name[col]
            ax.plot(t, seg[:, col], linewidth=0.8)
            ax.set_title(f"{rec_name} {lead_name}", fontsize=8)
            ax.set_yticks([])
            ax.set_xticks([])
        axes[row][0].set_ylabel(f"{rec_name}", fontsize=9)
    fig.suptitle("INCART raw 12-lead ECG (5 s window, start at 10 s)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f"[IL] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
