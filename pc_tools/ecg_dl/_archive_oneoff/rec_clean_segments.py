#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""干净段 20-26s: 原始/滤波峰值时间线, 判断真实 QRS 间距"""
import struct
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

root = Path(__file__).resolve().parents[2]
data = (root / 'rec_latest.ecgr').read_bytes()
n = struct.unpack_from('<I', data, 18)[0]
x = np.frombuffer(data, dtype='<i2', count=n, offset=32).astype(np.float64) / 8000.0
FS = 248.1
t_rec = np.arange(len(x)) / FS
t500 = np.arange(int(len(x) * 500 / FS)) / 500.0
x500 = interp1d(t_rec, x, kind='linear', bounds_error=False, fill_value='extrapolate')(t500)

for t0, t1 in [(20, 26), (32, 35), (44, 48)]:
    m = (t500 >= t0) & (t500 < t1)
    seg = x500[m] - np.median(x500[m])
    print(f'\n=== {t0}-{t1}s raw pp={np.ptp(seg):.3f} std={seg.std():.3f} ===')
    for dist, prom in [(0.30, 0.15), (0.45, 0.25), (0.60, 0.4)]:
        pk, props = find_peaks(seg, distance=int(dist * 500), prominence=prom)
        if len(pk) >= 2:
            rrs = np.diff(pk) / 500.0
            print(f'  dist>={dist}s prom>={prom}: n={len(pk)} RR中位={np.median(rrs):.3f}s -> {60/np.median(rrs):.1f} BPM  峰@ ' +
                  ' '.join('%.2f' % (t0 + p / 500.0) for p in pk))
        else:
            print(f'  dist>={dist}s prom>={prom}: n={len(pk)} 峰不足')
