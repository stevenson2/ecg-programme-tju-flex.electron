#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""smoke_deploy_chain.py — --deploy-chain 数据开关冒烟测试 (阶段 1.5)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import data.dataset as ds

ds.set_npz_suffix("_deploy")
d = ds.load_processed_data()
print("MIT deploy:", d["beats"].shape, d["labels"].shape, "rec_ids unique:", len(set(d["record_ids"].tolist())))
i = ds.load_incart_data()
print("INCART deploy:", i["beats"].shape, "rec_ids unique:", len(set(i["record_ids"].tolist())))
p = ds.load_ptb_data()
print("PTB deploy:", p["beats"].shape, "rec_ids unique:", len(set(p["record_ids"].tolist())))
print("SMOKE_OK")
