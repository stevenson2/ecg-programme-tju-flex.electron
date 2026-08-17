#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按录音逐秒质量分组统计 v6 检测 RR"""
import csv
import struct
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
data = (root / 'rec_latest.ecgr').read_bytes()
n = struct.unpack_from('<I', data, 18)[0]
x = np.frombuffer(data, dtype='<i2', count=n, offset=32).astype(np.float64) / 8000.0
FS = 248.1

sec_quality = {}
for s in range(0, len(x), int(FS)):
    seg = x[s:s + int(FS)]
    if len(seg) < FS * 0.8:
        break
    sec_quality[s // int(FS)] = (np.ptp(seg), np.mean(np.abs(seg) > 1.55))

beats = [int(r['idx']) for r in csv.DictReader(open(root / 'rec_latest_diag.csv', encoding='utf-8')) if r['is_beat'] == '1']
# diag idx 是 500Hz 复刻; 换算回录音时间
beat_sec = [i / 500.0 for i in beats]

for label, pred in [
    ('干净秒 (clip==0 且 pp<2.2V)', lambda q: q[1] == 0 and q[0] < 2.2),
    ('削顶/大摆秒 (clip>0 或 pp>=2.2V)', lambda q: q[1] > 0 or q[0] >= 2.2),
]:
    rrs = []
    prev = None
    for t in beat_sec:
        s = int(t)
        if s in sec_quality and pred(sec_quality[s]) and prev is not None:
            rrs.append(t - prev)
        prev = t
    rrs = np.array(rrs)
    if len(rrs):
        print(f'{label}: beats间隔 n={len(rrs)} RR中位={np.median(rrs):.3f}s -> {60/np.median(rrs):.1f} BPM  P10/P90={np.percentile(rrs,10):.3f}/{np.percentile(rrs,90):.3f}')
    else:
        print(label, ': 无数据')
