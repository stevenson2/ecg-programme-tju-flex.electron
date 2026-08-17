#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 OVS1 捕获 (100Hz CSV filtered 列) 重建 250Hz, 复算 VF v2 特征与分数"""
import json
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
models = ROOT / 'pc_tools' / 'ecg_dl' / 'models'
m = json.load(open(models / 'vf_detect_eval_v2.json', encoding='utf-8'))
mp = m['model_params']
mean_, std_ = [np.array(mp[k]) for k in ('mean', 'std')]
coef, intercept, theta = np.array(mp['coef']), mp['intercept'], mp['theta']

SOS = np.array([
    [2.67349040e-05, 5.34698080e-05, 2.67349040e-05, 1.0, -1.81135233, 0.846093824],
    [1.0, 2.0, 1.0, 1.0, -1.87777543, 0.893812835],
    [1.0, -2.0, 1.0, 1.0, -1.86519459, 0.922431828],
    [1.0, -2.0, 1.0, 1.0, -1.95573823, 0.966220895],
])

rows = []
for l in open(ROOT / 'esp_timer_check_afe_ovs1.txt', encoding='utf-8', errors='replace'):
    if re.match(r'^-?\d+\.\d{4},-?\d+\.\d{4},-?\d+\.\d{4},', l):
        p = l.split(',')
        if len(p) >= 9:
            rows.append(float(p[2]))
x100 = np.array(rows)
print('OVS1 filtered: n=%d pp=%.3f V rms=%.4f' % (len(x100), np.ptp(x100), np.sqrt(np.mean(x100**2))))

# 100Hz → 500Hz 三次样条重建 (信号经 LP40, 100Hz 采样满足 Nyquist)
t100 = np.arange(len(x100)) / 100.0
t500 = np.arange(int((len(x100)-1) * 5) + 1) / 500.0
f = interp1d(t100, x100, kind='cubic', bounds_error=False, fill_value='extrapolate')
x500 = f(t500)
x250 = x500[::2]  # 2:1 抽取

SCALE = 0.763
x_mv = x250 * SCALE


def features(x):
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x**2)))
    med = float(np.median(np.abs(x)))
    xf = sosfiltfilt(SOS, x, padlen=0)
    ratio = float(np.sum(xf**2) / (np.sum(x**2) + 1e-12))
    zc = float(np.sum(np.diff(np.sign(xf)) != 0)) / len(x)
    d = np.diff(x)
    pk = float(np.sum((d[:-1] > 0) & (d[1:] <= 0))) / 5.0
    zc_raw = float(np.sum(np.diff(np.sign(x)) != 0))
    dom = zc_raw / len(x) * 125.0
    return np.array([rms, med, ratio, zc, pk, dom])


print('窗级分数 (5s 不重叠, 固件口径):')
for k in range(0, len(x_mv) - 1250 + 1, 1250):
    x = x_mv[k:k + 1250]
    feat = features(x)
    z = intercept + float(coef @ ((feat - mean_) / std_))
    p = 1.0 / (1.0 + np.exp(-z))
    print(f'  win{k//1250:02d} rms={feat[0]:.3f} med={feat[1]:.3f} ratio={feat[2]:.3f} '
          f'zcr={feat[3]:.4f} pv={feat[4]:.1f} dom={feat[5]:.2f} score={p:.4f} suspect={p>=theta}')
