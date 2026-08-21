#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RR 对齐 sanity check: 验证 MIT 记录在 npz 中的拍数与 .atr 注解拍数是否匹配"""
import sys
from pathlib import Path
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import set_npz_suffix, load_mit_incart_merged
from collections import defaultdict

set_npz_suffix("_deploy")
mit_inc = load_mit_incart_merged()
rids = mit_inc["record_ids"]

# 每个记录在 npz 中的拍数
groups = defaultdict(list)
for i, rid in enumerate(rids):
    groups[int(rid)].append(i)

# MIT 48 条记录的 .atr 拍数
import wfdb
rec_dir = Path(__file__).resolve().parent / "data" / "raw" / "mit-bih-arrhythmia-database"
print(f"{'记录':<6}{'npz拍数':<10}{'.atr拍数':<10}{'倍数':<6}{'对齐OK'}")
for rec in sorted(groups):
    if rec < 200:  # MIT 记录 100-199 (排除 INCART 200+)
        n_npz = len(groups[rec])
        atr_path = rec_dir / f"{rec}.atr"
        if atr_path.exists():
            ann = wfdb.rdann(str(atr_path.with_suffix("")), 'atr')
            n_atr = len(ann.sample)
            mult = n_npz / n_atr if n_atr > 0 else 0
            ok = abs(mult - 6) < 0.01 if n_npz > 0 else False
            print(f"{rec:<6}{n_npz:<10}{n_atr:<10}{mult:<6.2f}{'✅' if ok else '❌'}")
