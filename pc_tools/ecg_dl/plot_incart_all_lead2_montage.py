#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_incart_all_lead2_montage.py -- visual montage of filtered Lead II for
all 75 INCART records, for per-record Lead II quality review.
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
OUT = CACHE / "incart_all_lead2_montage.png"
WIN_S = 5.0


def main():
    fig, axes = plt.subplots(8, 10, figsize=(30, 22), squeeze=False)
    for i in range(1, 76):
        rec_name = f"I{i:02d}"
        row = (i - 1) // 10
        col = (i - 1) % 10
        ax = axes[row][col]
        try:
            rec = wfdb.rdrecord(str(INCART_DIR / rec_name))
            fs = int(rec.fs)
            lead2 = rec.p_signal[:, 1].astype(np.float64)
            filtered = corrected_deployment_chain(lead2, fs)
            start = int(10.0 * TARGET_FS)
            seg = filtered[start:start + int(WIN_S * TARGET_FS)]
            t = np.arange(len(seg)) / TARGET_FS
            ax.plot(t, seg, linewidth=0.7)
            ax.set_title(rec_name, fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, "ERR", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    # Hide unused slots.
    for i in range(76, 81):
        row = (i - 1) // 10
        col = (i - 1) % 10
        axes[row][col].axis("off")
    fig.suptitle("INCART Lead II filtered 5s montage (I01-I75)", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT, dpi=90)
    print(f"[ILM] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
