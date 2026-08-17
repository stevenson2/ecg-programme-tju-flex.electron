#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OVS1 捕获三通道频谱粗看 (100Hz 采样)"""
import re
import numpy as np

rows = []
for l in open('esp_timer_check_afe_ovs1.txt', encoding='utf-8', errors='replace'):
    if re.match(r'^-?\d+\.\d{4},-?\d+\.\d{4},-?\d+\.\d{4},', l):
        p = l.split(',')
        if len(p) >= 9:
            rows.append([float(p[0]), float(p[1]), float(p[2])])
a = np.array(rows)
fs = 100.0
for i, name in enumerate(['clean(raw)', 'noisy(comb)', 'filtered(HP4+LP40)']):
    x = a[:, i]
    x = x - x.mean()
    w = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x * w))
    f = np.fft.rfftfreq(len(x), 1 / fs)
    print(f'{name}: pp={np.ptp(x):.3f} rms={np.sqrt(np.mean(x**2)):.4f}')
    for lo, hi in [(0.5, 2), (2, 5), (5, 10), (10, 20), (20, 40)]:
        band = (f >= lo) & (f < hi)
        print(f'  {lo}-{hi}Hz 能量占比 {np.sum(X[band]**2)/np.sum(X**2)*100:.1f}% 主峰 {f[band][np.argmax(X[band])]:.1f}Hz')
