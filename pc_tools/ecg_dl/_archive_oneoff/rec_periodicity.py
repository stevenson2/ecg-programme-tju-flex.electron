#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rec_latest: 逐秒质量 + 自相关周期判断"""
import struct
from pathlib import Path

import numpy as np
from scipy.signal import correlate

root = Path(__file__).resolve().parents[2]
data = (root / 'rec_latest.ecgr').read_bytes()
n = struct.unpack_from('<I', data, 18)[0]
x = np.frombuffer(data, dtype='<i2', count=n, offset=32).astype(np.float64) / 8000.0
FS = 248.1

print('逐秒质量:')
for s in range(0, len(x), int(FS)):
    seg = x[s:s + int(FS)]
    if len(seg) < FS * 0.8:
        break
    clip = np.mean(np.abs(seg) > 1.55)
    print(f'  {s/FS:4.1f}s pp={np.ptp(seg):6.3f} std={seg.std():5.3f} clip={clip:.3f}')

# 自相关 (去均值, 每 1s 块归一化后平均, 防幅度漂移)
segs = [x[s:s + int(FS)] - x[s:s + int(FS)].mean() for s in range(0, len(x), int(FS))]
ac = np.zeros(int(2.0 * FS))
cnt = np.zeros(int(2.0 * FS))
for seg in segs:
    if len(seg) < FS * 0.8 or seg.std() < 1e-6:
        continue
    a = correlate(seg, seg, mode='full') / (len(seg) * seg.var())
    a = a[len(seg) - 1:]
    L = min(len(a), len(ac))
    ac[:L] += a[:L]
    cnt[:L] += 1
ac = ac / np.maximum(cnt, 1)
lags = np.arange(len(ac)) / FS
# 0.3-1.5s 内找峰
band = (lags >= 0.30) & (lags <= 1.50)
acb = ac[band].copy()
lb = lags[band]
peaks = []
for i in range(1, len(acb) - 1):
    if acb[i] > acb[i - 1] and acb[i] >= acb[i + 1]:
        peaks.append((lb[i], acb[i]))
peaks.sort(key=lambda t: -t[1])
print('自相关峰 (lag, corr):', [(round(l, 3), round(c, 3)) for l, c in peaks[:8]])
