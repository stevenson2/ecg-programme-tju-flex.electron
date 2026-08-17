#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制 rec_latest 20s 波形 + v6 检测拍标记"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open('rec_latest_diag.csv', encoding='utf-8')))
idx = np.array([int(r['idx']) for r in rows])
sec = np.array([float(r['sec']) for r in rows])
y = np.array([float(r['chain']) for r in rows])
beats = np.array([float(r['sec']) for r in rows if r['is_beat'] == '1'])

fig, axes = plt.subplots(2, 1, figsize=(16, 8))
for ax, t0, t1, title in ((axes[0], 0, 10, 'first 10s'), (axes[1], 10, 20, '10-20s')):
    m = (sec >= t0) & (sec < t1)
    ax.plot(sec[m], y[m], lw=0.7)
    b = beats[(beats >= t0) & (beats < t1)]
    ax.plot(b, np.interp(b, sec, y), 'rv', markersize=7, label='v6 beats')
    ax.set_title(title)
    ax.set_xlabel('sec')
    ax.legend()
fig.tight_layout()
fig.savefig('rec_latest_diag.png', dpi=110)
print('saved rec_latest_diag.png  beats(0-20s)=', len(beats[(beats>=0)&(beats<20)]))
