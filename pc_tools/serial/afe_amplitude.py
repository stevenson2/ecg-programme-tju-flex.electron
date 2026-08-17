#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 AFE 捕获的 100Hz CSV 估计各通道幅度 (V), 用于 VF 尺度换算"""
import re
import statistics

rows = []
for l in open('esp_timer_check_afe.txt', encoding='utf-8', errors='replace'):
    if re.match(r'^-?\d+\.\d{4},-?\d+\.\d{4},-?\d+\.\d{4},', l):
        p = l.split(',')
        if len(p) >= 9:
            rows.append([float(p[0]), float(p[1]), float(p[2])])
print('rows', len(rows))
for i, name in enumerate(['clean', 'noisy', 'filtered']):
    v = [r[i] for r in rows]
    print(f'{name}: min={min(v):.4f} max={max(v):.4f} pp={max(v)-min(v):.4f} '
          f'rms≈{statistics.pstdev(v):.4f} med={statistics.median(v):.4f}')
# 每秒分块 pp
for i, name in enumerate(['clean', 'noisy', 'filtered']):
    secs = []
    for s in range(0, len(rows), 100):
        seg = rows[s:s+100]
        if seg:
            v = [r[i] for r in seg]
            secs.append(max(v) - min(v))
    print(f'{name} 每秒pp: 最小={min(secs):.4f} 中位={statistics.median(secs):.4f} 最大={max(secs):.4f}')
