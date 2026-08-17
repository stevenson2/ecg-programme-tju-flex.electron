#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 捕获: [心率] 行时间线 + CSV bpm 分段"""
import re
import statistics

lines = open('esp_timer_check_afe_v3.txt', encoding='utf-8', errors='replace').read().splitlines()
hr = [l.strip() for l in lines if '[心率]' in l]
print('心率行数', len(hr))
for h in hr[:8]:
    print(' ', h[:120])
print('  ...中间每隔10行:')
for h in hr[::10][:12]:
    print(' ', h[:120])
print('  末尾8行:')
for h in hr[-8:]:
    print(' ', h[:120])

rows = []
for l in lines:
    if re.match(r'^-?\d+\.\d{4},-?\d+\.\d{4},-?\d+\.\d{4},', l):
        p = l.split(',')
        if len(p) >= 9:
            rows.append(p)
bpm = [int(x[3]) for x in rows if x[3].isdigit()]
print('CSV bpm n=%d 中位=%s 分位=%s' % (len(bpm), statistics.median(bpm),
      [round(statistics.quantiles(bpm, n=4)[i],1) for i in range(3)] if bpm else []))
# 每 10s 段 bpm 中位
for s in range(0, len(bpm), 1000):
    seg = bpm[s:s+1000]
    if seg:
        print(f'  {s//1000*10:2d}-{(s//1000+1)*10:2d}s: n={len(seg)} med={statistics.median(seg)} min={min(seg)} max={max(seg)}')
