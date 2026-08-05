#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_noaug_testset.py — T1-2: 重建未增强 MIT-BIH 测试拍 (训练数据不变)
=========================================================================
任务: 必做清单 T1-2 / solutions.md M5
方法: preprocess.py process_all_records(augment=False) → mit_bih_processed_noaug.npz
   (原始拍 1×, 不含 6× 增强变体; 训练仍用 mit_bih_processed.npz 6× 增强数据)
用法 (WSL): export ECG_PROCESSED_DIR=$HOME/ecg_data; python3 build_noaug_testset.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR
from data.preprocess import process_all_records

OUT = PROCESSED_DIR / "mit_bih_processed_noaug.npz"
print(f"[T1-2] 输出: {OUT}")

result = process_all_records(augment=False)
beats, labels, rids = result["beats"], result["labels"], result["record_ids"]
print(f"[T1-2] 未增强 MIT-BIH: {len(beats)} 拍 "
      f"(N={(labels == 0).sum()}, A={(labels == 1).sum()}, 记录 {len(np.unique(rids))})")

np.savez_compressed(OUT, beats=beats, labels=labels, record_ids=rids)
print(f"[T1-2] ✅ 已保存: {OUT} ({OUT.stat().st_size / 1024 / 1024:.1f} MB)")
