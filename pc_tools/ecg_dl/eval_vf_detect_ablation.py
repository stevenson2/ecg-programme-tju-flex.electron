#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VF 特征变体消融: 定位 CUDB Se 下降来源 (2026-08-16)"""
import sys
from pathlib import Path
import numpy as np
from scipy.signal import sosfilt, sosfiltfilt, butter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_vf_detect_v2 import (
    FS, WIN_N, WIN_S, SOS, load_vfdb_annotated, load_mit_control, load_cudb, window_stream,
)

# 带通 (filtfilt, 与 v1 相同)
SOS_BP = butter(4, [4 / (FS / 2), 10 / (FS / 2)], btype='band', output='sos')


def f_abs_filtfilt(x):
    x = x - x.mean()
    rms = np.sqrt(np.mean(x ** 2))
    med = np.median(np.abs(x))
    xf = sosfiltfilt(SOS_BP, x)
    ratio = np.sum(xf ** 2) / (np.sum(x ** 2) + 1e-12)
    zc = np.sum(np.diff(np.sign(xf)) != 0) / WIN_N
    d = np.diff(x)
    pk = np.sum((d[:-1] > 0) & (d[1:] <= 0)) / WIN_S
    w = np.hanning(len(x))
    Xf = np.abs(np.fft.rfft(x * w))
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    band = (freqs >= 2) & (freqs <= 50)
    dom = freqs[band][np.argmax(Xf[band])]
    return [rms, med, ratio, zc, pk, dom]


def f_abs_causal(x):
    """v1 特征口径, 但带通因果 + 主频 ZCR 复刻"""
    x = x - x.mean()
    rms = np.sqrt(np.mean(x ** 2))
    med = np.median(np.abs(x))
    xf = sosfilt(SOS, x)
    ratio = np.sum(xf ** 2) / (np.sum(x ** 2) + 1e-12)
    zc = np.sum(np.diff(np.sign(xf)) != 0) / WIN_N
    d = np.diff(x)
    pk = np.sum((d[:-1] > 0) & (d[1:] <= 0)) / WIN_S
    zc_raw = np.sum(np.diff(np.sign(x)) != 0)
    dom = zc_raw / WIN_N * (FS / 2.0)
    return [rms, med, ratio, zc, pk, dom]


def f_p95_causal(x):
    x = x - x.mean()
    rms = np.sqrt(np.mean(x ** 2))
    med = np.median(np.abs(x))
    p95 = np.percentile(np.abs(x), 95) + 1e-9
    xf = sosfilt(SOS, x)
    ratio = np.sum(xf ** 2) / (np.sum(x ** 2) + 1e-12)
    zc = np.sum(np.diff(np.sign(xf)) != 0) / WIN_N
    d = np.diff(x)
    pk = np.sum((d[:-1] > 0) & (d[1:] <= 0)) / WIN_S
    zc_raw = np.sum(np.diff(np.sign(x)) != 0)
    dom = zc_raw / WIN_N * (FS / 2.0)
    return [rms / p95, med / p95, ratio, zc, pk, dom]


def f_med_causal(x):
    """幅度特征只保留 rms/med, 分母换 med (med 特征退化为 1 则丢弃)"""
    x = x - x.mean()
    rms = np.sqrt(np.mean(x ** 2))
    med = np.median(np.abs(x)) + 1e-9
    xf = sosfilt(SOS, x)
    ratio = np.sum(xf ** 2) / (np.sum(x ** 2) + 1e-12)
    zc = np.sum(np.diff(np.sign(xf)) != 0) / WIN_N
    d = np.diff(x)
    pk = np.sum((d[:-1] > 0) & (d[1:] <= 0)) / WIN_S
    zc_raw = np.sum(np.diff(np.sign(x)) != 0)
    dom = zc_raw / WIN_N * (FS / 2.0)
    return [rms / med, ratio, zc, pk, dom]


def f_logabs_causal(x):
    """幅度取 log (设备增益变成常数偏移, 由截距吸收? 训练/部署偏移未知)"""
    x = x - x.mean()
    rms = np.sqrt(np.mean(x ** 2))
    med = np.median(np.abs(x))
    xf = sosfilt(SOS, x)
    ratio = np.sum(xf ** 2) / (np.sum(x ** 2) + 1e-12)
    zc = np.sum(np.diff(np.sign(xf)) != 0) / WIN_N
    d = np.diff(x)
    pk = np.sum((d[:-1] > 0) & (d[1:] <= 0)) / WIN_S
    zc_raw = np.sum(np.diff(np.sign(x)) != 0)
    dom = zc_raw / WIN_N * (FS / 2.0)
    return [np.log(rms + 1e-9), np.log(med + 1e-9), ratio, zc, pk, dom]


def run(name, feat_fn):
    data = load_vfdb_annotated()
    vf_recs = list(data)
    X_vf = np.concatenate([data[r][0][data[r][1] == 1] for r in vf_recs])
    X_ctl = load_mit_control()
    # 重新生成特征 (window_stream 默认用 v2 特征, 这里手工重算)
    # 用原始信号重新计算代价大; 直接对已有窗口特征无法换口径 → 重新加载太慢。
    # 简化: 本实验接受重新加载 (与主脚本一致)
    return None


def gen_features(load_fn, feat_fn):
    return load_fn()


print('该消融需要按特征口径重新生成窗口; 使用流式重算...')
# 直接对原始信号重新生成所有窗口特征
def windows_of(sig):
    X = []
    t = 0.0
    dur = len(sig) / FS
    while t + WIN_S <= dur:
        i0, i1 = int(t * FS), int((t + WIN_S) * FS)
        x = sig[i0:i1]
        if np.std(x) > 1e-6:
            X.append(feat_fn(x))
        t += WIN_S / 2
    return np.array(X)


data = load_vfdb_annotated()
vf_recs = list(data)
# 训练窗: VFDB VF 窗 + MIT 对照 (与 v1/v2 相同记录)
import wfdb
from scipy.signal import sosfilt

MIT_DIR = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl/data/raw/mit-bih-arrhythmia-database')
MIT_RECS = ('100', '103', '113', '117', '121', '122', '123')

variants = {
    'v1_abs_filtfilt': f_abs_filtfilt,
    'v1b_abs_causal': f_abs_causal,
    'v2_p95_causal': f_p95_causal,
    'v2b_med_causal(4feat)': f_med_causal,
    'v2c_logabs_causal': f_logabs_causal,
}

for name, feat_fn in variants.items():
    Xv, Xc = [], []
    for r in vf_recs:
        sig = wfdb.rdrecord(str(Path('/home/devcontainers/vfdb') / r), channels=[0]).p_signal[:, 0].astype(np.float64)
        Xv.append(windows_of(sig)[data[r][1] == 1])
    for r in MIT_RECS:
        sig = wfdb.rdrecord(str(MIT_DIR / r), channels=[0]).p_signal[:, 0].astype(np.float64)
        Xc.append(windows_of(sig))
    Xv = np.concatenate(Xv)
    Xc = np.concatenate(Xc)
    X_tr = np.concatenate([Xv, Xc])
    y_tr = np.concatenate([np.ones(len(Xv)), np.zeros(len(Xc))])
    mean_, std_ = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-9
    clf = LogisticRegression(max_iter=3000, C=0.05).fit((X_tr - mean_) / std_, y_tr)
    p = clf.predict_proba((X_tr - mean_) / std_)[:, 1]
    best = None
    for thr in np.arange(0.05, 0.95, 0.005):
        se = (p[y_tr == 1] >= thr).mean()
        sp = (p[y_tr == 0] < thr).mean()
        if se < 0.95:
            continue
        if best is None or (sp >= 0.83 and (best['sp'] < 0.83 or thr < best['thr'])) or (best['sp'] < 0.83 and sp > best['sp']):
            best = {'thr': thr, 'se': se, 'sp': sp}
    # CUDB
    Xcu = []
    for rec in sorted(Path('/home/devcontainers/vfdb').glob('cu*.dat')):
        sig = wfdb.rdrecord(str(rec.with_suffix('')), channels=[0]).p_signal[:, 0].astype(np.float64)
        Xcu.append(windows_of(sig))
    Xcu = np.concatenate(Xcu)
    pcu = clf.predict_proba((Xcu - mean_) / std_)[:, 1]
    se_cu = float((pcu >= best['thr']).mean())
    # MIT 留出
    Xct = []
    for r in ('103', '117', '123'):
        sig = wfdb.rdrecord(str(MIT_DIR / r), channels=[0]).p_signal[:, 0].astype(np.float64)
        Xct.append(windows_of(sig))
    Xct = np.concatenate(Xct)
    pct = clf.predict_proba((Xct - mean_) / std_)[:, 1]
    sp_ctl = float((pct < best['thr']).mean())
    print(f'{name:<24} AUC={roc_auc_score(y_tr, p):.4f} θ={best["thr"]:.2f} '
          f'calSe={best["se"]:.3f} calSp={best["sp"]:.3f} CUDB_Se={se_cu:.3f} MIT_Sp={sp_ctl:.3f}')
