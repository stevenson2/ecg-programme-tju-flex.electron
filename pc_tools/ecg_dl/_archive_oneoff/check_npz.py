#!/usr/bin/env python3
"""Quick check of original npz counts."""
import numpy as np
from pathlib import Path

for name in ['mit_bih_processed', 'incart_processed', 'ptb_processed']:
    p = Path('data/processed') / f'{name}.npz'
    if p.exists():
        d = np.load(p)
        n = len(d['beats'])
        recs = np.unique(d['record_ids'])
        nN = int((d['labels'] == 0).sum())
        nA = int((d['labels'] == 1).sum())
        print(f'{name}.npz: {n} beats ({nN}N/{nA}A), {len(recs)} records, size={p.stat().st_size/1024/1024:.1f}MB')
    else:
        print(f'{name}.npz: NOT FOUND')
