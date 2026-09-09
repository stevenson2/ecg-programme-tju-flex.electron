#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LUDB 链输出 (HP0.5+LP40, gain1000) 经 VF v2 模型: 正常记录应不触发"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_heartrate_ludb_v5 import chain_filter_v5, load_ludb_record, DEFAULT_DATA_DIR
from eval_vf_detect_v2 import SOS

models = Path(__file__).resolve().parent / 'models'
m = json.load(open(models / 'vf_detect_eval_v2.json', encoding='utf-8'))
mp = m['model_params']
mean_, std_ = [np.array(mp[k]) for k in ('mean', 'std')]
coef, intercept, theta = np.array(mp['coef']), mp['intercept'], mp['theta']
SCALE = 1.0 / 1000.0  # LUDB 链输出数值是 mV×1000 域 → 电极 mV


def score_of(x_mv):
    x = x_mv - x_mv.mean()
    rms = np.sqrt(np.mean(x ** 2))
    med = np.median(np.abs(x))
    xf = sosfiltfilt(SOS, x, padlen=0)
    ratio = np.sum(xf ** 2) / (np.sum(x ** 2) + 1e-12)
    zc = np.sum(np.diff(np.sign(xf)) != 0) / len(x)
    d = np.diff(x)
    pk = np.sum((d[:-1] > 0) & (d[1:] <= 0)) / 5.0
    zc_raw = np.sum(np.diff(np.sign(x)) != 0)
    dom = zc_raw / len(x) * 125.0
    feat = np.array([rms, med, ratio, zc, pk, dom])
    z = intercept + float(coef @ ((feat - mean_) / std_))
    p = 1.0 / (1.0 + np.exp(-z))
    return p, feat


recs = [l.strip().split('/')[-1] for l in open(str(DEFAULT_DATA_DIR.parent / 'RECORDS')).read().splitlines() if l.strip()]
scores, feats = [], []
for rid in recs[:60]:
    sig, _ = load_ludb_record(DEFAULT_DATA_DIR, rid, 'ii')
    y = chain_filter_v5(sig, 1000.0)
    y250 = y[::2]
    x = y250 * SCALE
    for k in range(0, len(x) - 1250 + 1, 1250):
        p, f = score_of(x[k:k + 1250])
        scores.append(p)
        feats.append(f)
scores = np.array(scores)
print(f'LUDB 60 记录滤波链后 VF v2 分数: n={len(scores)} min={scores.min():.4f} '
      f'中位={np.median(scores):.4f} p90={np.percentile(scores,90):.4f} max={scores.max():.4f} '
      f'suspect(>={theta})={np.sum(scores>=theta)} ({100*np.mean(scores>=theta):.1f}%)')
bad = np.where(scores >= theta)[0]
for i in bad[:10]:
    f = feats[i]
    print(f'  win{i}: rms={f[0]:.3f} med={f[1]:.3f} ratio={f[2]:.3f} zcr={f[3]:.4f} pv={f[4]:.1f} dom={f[5]:.2f} score={scores[i]:.4f}')
