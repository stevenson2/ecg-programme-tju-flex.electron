#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""smoke_mmap.py — mmap 加载冒烟测试 (TUNING_HISTORY 十三章)"""
import os
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("ECG_PROCESSED_DIR", str(Path.home() / "ecg_data"))
import data.dataset as ds

ds.set_npz_suffix("_deploy")
d = ds.load_processed_data()
print("beats type:", type(d["beats"]).__name__, "| is memmap:", isinstance(d["beats"], np.memmap))
i = ds.load_incart_data()
p = ds.load_ptb_data()
print("OK shapes:", d["beats"].shape, i["beats"].shape, p["beats"].shape)
print("SMOKE_OK")
