#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OVS1 捕获: 重建两种 VF 输入路径, 用当前固件 raw 训练 v2 模型打分"""
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import lfilter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
FS = 500

# 当前固件 raw 训练 v2 参数 (第一次 v2 拟合, 已烧录)
mean_ = np.array([0.42513171332031363, 0.24363859527220647, 0.25254772741265197,
                  0.04137449856733724, 43.309992836676216, 4.105229226361016])
std_ = np.array([0.4191231586420349, 0.3631392072566181, 0.12891100945922151,
                 0.004458203265709307, 9.81194160995198, 1.7036455584085064])
coef = np.array([0.6866138077335683, 4.134092762658282, -0.8109653996849735,
                 1.2396861763592981, -1.2619362566532957, 1.3262778596404012])
intercept = 0.24291412543887136
theta = 0.15
SOS = np.array([
    [2.67349040e-05, 5.34698080e-05, 2.67349040e-05, 1.0, -1.81135233, 0.846093824],
    [1.0, 2.0, 1.0, 1.0, -1.87777543, 0.893812835],
    [1.0, -2.0, 1.0, 1.0, -1.86519459, 0.922431828],
    [1.0, -2.0, 1.0, 1.0, -1.95573823, 0.966220895],
])
HP = dict(b=(0.9955669720176472, -1.9911339440352944, 0.9955669720176472),
          a=(1.0, -1.9911142922016536, 0.9911535958689355))
LP = dict(b=(0.046131802093312926, 0.09226360418662585, 0.046131802093312926),
          a=(1.0, -1.3072850288493234, 0.4918122372225752))
SCALE_AFE = 0.763


def score_of(x_mv):
    x = x_mv - x_mv.mean()
    rms = float(np.sqrt(np.mean(x ** 2)))
    med = float(np.median(np.abs(x)))
    xf = sosfiltfilt(SOS, x, padlen=0)
    ratio = float(np.sum(xf ** 2) / (np.sum(x ** 2) + 1e-12))
    zc = float(np.sum(np.diff(np.sign(xf)) != 0)) / len(x)
    d = np.diff(x)
    pk = float(np.sum((d[:-1] > 0) & (d[1:] <= 0))) / 5.0
    zc_raw = float(np.sum(np.diff(np.sign(x)) != 0))
    dom = zc_raw / len(x) * 125.0
    feat = np.array([rms, med, ratio, zc, pk, dom])
    z = intercept + float(coef @ ((feat - mean_) / std_))
    return 1.0 / (1.0 + np.exp(-z)), feat


rows = []
for l in open(ROOT / 'esp_timer_check_afe_ovs1.txt', encoding='utf-8', errors='replace'):
    if re.match(r'^-?\d+\.\d{4},-?\d+\.\d{4},-?\d+\.\d{4},', l):
        p = l.split(',')
        if len(p) >= 9:
            rows.append(float(p[0]))  # clean = raw 去偏置 (500Hz 前链未滤波)
x100 = np.array(rows)
t100 = np.arange(len(x100)) / 100.0
t500 = np.arange(int((len(x100) - 1) * 5) + 1) / 500.0
x500 = interp1d(t100, x100, kind='cubic', bounds_error=False, fill_value='extrapolate')(t500)

# 路径 A: comb(双10抽头) only
k = np.ones(10) / 10.0
comb = np.convolve(np.convolve(x500, k, mode='full')[:len(x500)], k, mode='full')[:len(x500)]
# 路径 B: comb + HP0.5 + LP40
hpf = lfilter(HP['b'], HP['a'], comb)
full = lfilter(LP['b'], LP['a'], hpf)

for name, y in (('A comb-only', comb), ('B comb+HP/LP (当前固件)', full)):
    y250 = y[::2] * SCALE_AFE
    s = []
    for kk in range(0, len(y250) - 1250 + 1, 1250):
        s.append(score_of(y250[kk:kk + 1250]))
    print(name)
    for i, (p, f) in enumerate(s):
        print(f'  win{i:02d} score={p:.4f} suspect={p>=theta}  rms={f[0]:.3f} med={f[1]:.3f} ratio={f[2]:.3f} zcr={f[3]:.4f} pv={f[4]:.1f} dom={f[5]:.2f}')
