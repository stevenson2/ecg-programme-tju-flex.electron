#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""resave_uncompressed.py — 将 deploy npz 重存为无压缩格式 (mmap 可加载, 降 RSS)"""
import numpy as np
from pathlib import Path

SRC = Path("/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl/data/processed")
DST = Path.home() / "ecg_data"

for name in ["mit_bih_processed_deploy.npz", "incart_processed_deploy.npz", "ptb_processed_deploy.npz"]:
    d = np.load(SRC / name)
    out = DST / name
    np.savez(out, beats=d["beats"], labels=d["labels"], record_ids=d["record_ids"])
    print(f"{name}: {out.stat().st_size/1024/1024:.0f} MB (uncompressed)")
print("DONE")
