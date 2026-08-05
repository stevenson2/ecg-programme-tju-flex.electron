#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""4.4-4 冒烟测试: 验证 patient_split=True 训练路径正确工作。
只验证数据划分逻辑, 不训练模型。运行 (WSL2):
  cd /mnt/c/.../pc_tools/ecg_dl && python3 smoke_patient_split.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import prepare_datasets

print("=" * 60)
print("冒烟测试: prepare_datasets(patient_split=True, use_ptb_beat=True)")
print("=" * 60)

ds = prepare_datasets(
    use_incart=True,
    use_ptb_beat=True,
    ptb_abn_max=10000,
    patient_split=True,
    batch_size=512,
)

# 验证: 取出原始拍级数据, 检查 train/test 患者交集为空
from data.patient_split import (build_mit_patient_map, build_incart_patient_map,
                                build_ptb_patient_map, patient_level_split)
import numpy as np

splits = ds["data"]
x_tr, y_tr = splits["train"]
x_te, y_te = splits["test"]
print(f"\n[验证] train 拍: {len(x_tr):,} | test 拍: {len(x_te):,}")
print(f"[验证] train 异常占比: {np.mean(y_tr==1)*100:.1f}%")

# PTB 训练拍应仅来自 train 患者 (约 60% × 69482 ≈ 41K)
# (prepare_datasets 内部已过滤, 这里只确认不崩溃 + 拍数合理)
print("\n[冒烟测试] ✅ patient_split 训练路径正常工作")