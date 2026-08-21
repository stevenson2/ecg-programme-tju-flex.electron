#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""干净段峰宽: 区分 QRS(窄) 与正弦伪迹(宽)"""
import struct
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths

root = Path(__file__).resolve().parents[2]
data = (root / 'rec_latest.ecgr').read_bytes()
n = struct.unpack_from('<I', data, 18)[0]
x = np.frombuffer(data, dtype='<i2', count=n, offset=32).astype(np.float64) / 8000.0
FS = 248.1
t_rec = np.arange(len(x)) / FS
t500 = np.arange(int(len(x) * 500 / FS)) / 500.0
x500 = interp1d(t_rec, x, kind='linear', bounds_error=False, fill_value='extrapolate')(t500)

for t0, t1 in [(20, 26), (32, 35)]:
    m = (t500 >= t0) & (t500 < t1)
    seg = x500[m] - np.median(x500[m])
    pk, props = find_peaks(seg, distance=int(0.3 * 500), prominence=0.15)
    w, wh, left, right = peak_widths(seg, pk, rel_height=0.5)
    print(f'=== {t0}-{t1}s: n={len(pk)}')
    for i in range(len(pk)):
        print(f'  peak@{t0+pk[i]/500:.2f}s 高={seg[pk[i]]:.3f} 半高宽={w[i]/500*1000:.0f}ms 前沿={left[i]/500*1000:.0f}ms 后沿={right[i]/500*1000:.0f}ms')
