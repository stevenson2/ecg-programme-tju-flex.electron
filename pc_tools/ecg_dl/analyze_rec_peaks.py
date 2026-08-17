#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""能量包络峰级对照: v6 接受拍 vs 全部 MWI 峰 (rec_latest)"""
import csv
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_heartrate_ludb_v5 import chain_filter_v5
from verify_heartrate_ludb_v5 import QRS_LP25_A1, QRS_LP25_A2, QRS_LP25_B0, QRS_LP25_B1, QRS_LP25_B2
from verify_heartrate_ludb_v5 import QRS_HP8_A1, QRS_HP8_A2, QRS_HP8_B0, QRS_HP8_B1, QRS_HP8_B2

root = Path(__file__).resolve().parents[2]
data = (root / 'rec_latest.ecgr').read_bytes()
n = struct.unpack_from('<I', data, 18)[0]
x = np.frombuffer(data, dtype='<i2', count=n, offset=32).astype(np.float64) / 8000.0
y = chain_filter_v5(x, gain=1.0)
FS_REC = 248.1
t_rec = np.arange(len(y)) / FS_REC
t500 = np.arange(int(len(y) * 500 / FS_REC)) / 500.0
y500 = interp1d(t_rec, y, kind='linear', bounds_error=False, fill_value='extrapolate')(t500)

# QRS 8-25Hz + 能量 + MWI40 (向量化, 与固件近似)
yl = lfilter([QRS_LP25_B0, QRS_LP25_B1, QRS_LP25_B2],
             [1, QRS_LP25_A1, QRS_LP25_A2], y500)
yh = lfilter([QRS_HP8_B0, QRS_HP8_B1, QRS_HP8_B2],
             [1, QRS_HP8_A1, QRS_HP8_A2], yl)
energy = yh ** 2
mwi = np.convolve(energy, np.ones(40) / 40.0, mode='full')[:len(energy)]

# 局部峰
pk_idx = []
for i in range(2, len(mwi) - 1):
    if mwi[i-1] > mwi[i-2] and mwi[i-1] >= mwi[i]:
        pk_idx.append(i - 1)
pk_idx = np.array(pk_idx)

beats = set()
for r in csv.DictReader(open(root / 'rec_latest_diag.csv', encoding='utf-8')):
    if r['is_beat'] == '1':
        beats.add(int(r['idx']))
beats = sorted(beats)

# 每个峰找最近 beat
print('MWI 峰数', len(pk_idx), 'v6 beats', len(beats))
for k, p in enumerate(pk_idx):
    if mwi[p] < 1e-6:
        continue
    nearest_beat = min(beats, key=lambda b: abs(b - p))
    dist = abs(nearest_beat - p)
    # 半高宽
    half = mwi[p] * 0.5
    left = p
    while left > 0 and mwi[left] > half:
        left -= 1
    right = p
    while right < len(mwi) - 1 and mwi[right] > half:
        right += 1
    w = right - left
    mark = 'BEAT' if dist <= 2 else ('near%+d' % (p - nearest_beat))
    print('  pk=%5d %.3fs amp=%.3e w=%3d  %s' % (p, p / 500.0, mwi[p], w, mark))
