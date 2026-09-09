#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_incart_leads_filtered.py -- filtered INCART limb-lead ECG preview.

Applies the project deployment-causal chain (corrected_deployment_chain) to the
selected INCART limb leads and plots them, so the user can visually identify
which lead is the true Lead II-like channel.
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
from eval_deploy_match import corrected_deployment_chain

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
INCART_DIR = _REFS.INCART_DIR
OUT = CACHE / "incart_leads_filtered_preview.png"
RECORDS = ["I01", "I02", "I10"]
LEAD_COLS = [0, 1, 2, 3, 4, 5]  # I, II, III, aVR, aVL, aVF
WINDOW_S = 5.0


def main():
    fig, axes = plt.subplots(len(RECORDS), len(LEAD_COLS), figsize=(20, 3.0 * len(RECORDS)),
                             squeeze=False)
    for row, rec_name in enumerate(RECORDS):
        rec = wfdb.rdrecord(str(INCART_DIR / rec_name))
        fs = int(rec.fs)
        start_raw = int(10.0 * fs)
        end_raw = int((10.0 + WINDOW_S) * fs)
        raw = rec.p_signal[start_raw:end_raw]
        for col, lead_idx in enumerate(LEAD_COLS):
            ax = axes[row][col]
            lead_name = rec.sig_name[lead_idx]
            filtered = corrected_deployment_chain(raw[:, lead_idx].astype(np.float64), fs)
            # Filtered is 250 Hz; keep 5 s from the same time window.
            seg = filtered[:int(WINDOW_S * 250)]
            t = np.arange(len(seg)) / 250.0
            ax.plot(t, seg, linewidth=0.8)
            ax.set_title(f"{rec_name} {lead_name} filtered", fontsize=8)
            ax.set_yticks([])
            ax.set_xticks([])
    fig.suptitle("INCART limb leads after deployment-causal filter (5 s window)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f"[ILF] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
