#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VF 消融第二组 + AFE 增益估计 (2026-08-16)"""
import sys
from pathlib import Path
import numpy as np
import wfdb
from scipy.signal import sosfilt, sosfiltfilt, butter, resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_vf_detect_v2 import FS, WIN_N, WIN_S, SOS, load_vfdb_annotated, load_mit_control

SOS_BP = butter(4, [4 / (FS / 2), 10 / (FS / 2)], btype='band', output='sos')
VFDB = Path('/home/devcontainers/vfdb')
MIT_DIR = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl/data/raw/mit-bih-arrhythmia-database')


def fft_dom(x):
    w = np.hanning(len(x))
    Xf = np.abs(np.fft.rfft(x * w))
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    band = (freqs >= 2) & (freqs <= 50)
    return float(freqs[band][np.argmax(Xf[band])])


def zcr_dom(x):
    zc = np.sum(np.diff(np.sign(x)) != 0)
    return float(zc) / WIN_N * (FS / 2.0)


def f_abs_causal_fft(x):
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x ** 2)))
    med = float(np.median(np.abs(x)))
    xf = sosfilt(SOS, x)
    ratio = float(np.sum(xf ** 2) / (np.sum(x ** 2) + 1e-12))
    zc = float(np.sum(np.diff(np.sign(xf)) != 0)) / WIN_N
    d = np.diff(x)
    pk = float(np.sum((d[:-1] > 0) & (d[1:] <= 0))) / WIN_S
    return [rms, med, ratio, zc, pk, fft_dom(x)]


def f_abs_filtfilt_zcr(x):
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x ** 2)))
    med = float(np.median(np.abs(x)))
    xf = sosfiltfilt(SOS_BP, x)
    ratio = float(np.sum(xf ** 2) / (np.sum(x ** 2) + 1e-12))
    zc = float(np.sum(np.diff(np.sign(xf)) != 0)) / WIN_N
    d = np.diff(x)
    pk = float(np.sum((d[:-1] > 0) & (d[1:] <= 0))) / WIN_S
    return [rms, med, ratio, zc, pk, zcr_dom(x)]


def f_p95_causal_fft(x):
    x = x - x.mean()
    p95 = float(np.percentile(np.abs(x), 95)) + 1e-9
    rms = float(np.sqrt(np.mean(x ** 2))) / p95
    med = float(np.median(np.abs(x))) / p95
    xf = sosfilt(SOS, x)
    ratio = float(np.sum(xf ** 2) / (np.sum(x ** 2) + 1e-12))
    zc = float(np.sum(np.diff(np.sign(xf)) != 0)) / WIN_N
    d = np.diff(x)
    pk = float(np.sum((d[:-1] > 0) & (d[1:] <= 0))) / WIN_S
    return [rms, med, ratio, zc, pk, fft_dom(x)]


def windows_of(sig, feat_fn):
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


def load_windows(rec, base, feat_fn):
    sig = wfdb.rdrecord(str(base / rec), channels=[0]).p_signal[:, 0].astype(np.float64)
    return windows_of(sig, feat_fn)


def main():
    data = load_vfdb_annotated()
    vf_recs = list(data)

    for name, feat_fn in {
        'v1c_abs_causal_fft': f_abs_causal_fft,
        'v1d_abs_filtfilt_zcr': f_abs_filtfilt_zcr,
        'v2d_p95_causal_fft': f_p95_causal_fft,
    }.items():
        Xv = np.concatenate([load_windows(r, VFDB, feat_fn)[data[r][1] == 1] for r in vf_recs])
        Xc = np.concatenate([load_windows(r, MIT_DIR, feat_fn) for r in
                             ('100', '103', '113', '117', '121', '122', '123')])
        X_tr = np.concatenate([Xv, Xc])
        y_tr = np.concatenate([np.ones(len(Xv)), np.zeros(len(Xc))]).astype(int)
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
        Xcu = np.concatenate([load_windows(r.stem, VFDB, feat_fn) for r in sorted(VFDB.glob('cu*.dat'))])
        pcu = clf.predict_proba((Xcu - mean_) / std_)[:, 1]
        se_cu = float((pcu >= best['thr']).mean())
        Xct = np.concatenate([load_windows(r, MIT_DIR, feat_fn) for r in ('103', '117', '123')])
        pct = clf.predict_proba((Xct - mean_) / std_)[:, 1]
        sp_ctl = float((pct < best['thr']).mean())
        print(f'{name:<22} AUC={roc_auc_score(y_tr, p):.4f} θ={best["thr"]:.2f} '
              f'calSe={best["se"]:.3f} calSp={best["sp"]:.3f} CUDB_Se={se_cu:.3f} MIT_Sp={sp_ctl:.3f}')

    # ---- AFE 增益估计: LUDB 链输出 (mV×1000) 每 1s pp 中位 vs AFE filtered 每 1s pp 中位 ----
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from verify_heartrate_ludb_v5 import chain_filter_v5, DEFAULT_DATA_DIR
    recs = [l.strip().split('/')[-1] for l in open(str(DEFAULT_DATA_DIR.parent / 'RECORDS')).read().splitlines() if l.strip()]
    pps = []
    for rid in recs:
        sig, _ = load_ludb_record_local(DEFAULT_DATA_DIR, rid)
        y = chain_filter_v5(sig, 1000.0)
        for s in range(0, len(y) - 500, 500):
            seg = y[s:s + 500]
            pps.append(float(np.ptp(seg)))
    pps = np.array(pps)
    print(f'LUDB 链输出 (mV×1000 域) 每 1s pp: 中位={np.median(pps):.3f}V P10={np.percentile(pps,10):.3f} P90={np.percentile(pps,90):.3f}')

    # AFE 捕获 filtered 每 1s pp (100Hz CSV → 每 100 行 ≈1s)
    import re
    rows = []
    for l in open('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/esp_timer_check_afe.txt', encoding='utf-8', errors='replace'):
        if re.match(r'^-?\d+\.\d{4},-?\d+\.\d{4},-?\d+\.\d{4},', l):
            p9 = l.split(',')
            if len(p9) >= 9:
                rows.append(float(p9[2]))
    afe = [np.ptp(rows[s:s + 100]) for s in range(0, len(rows) - 100, 100)]
    afe = np.array(afe)
    print(f'AFE filtered 每 1s pp: 中位={np.median(afe):.3f}V P10={np.percentile(afe,10):.3f} P90={np.percentile(afe,90):.3f}')
    g_med = 1000.0 * np.median(afe) / np.median(pps)
    print(f'AFE 增益估计 g = 1000×AFE_pp/LUDB_pp = {g_med:.0f} (输入 mV → ADC V)')


def load_ludb_record_local(data_dir, rid):
    hdr = wfdb.rdheader(str(data_dir / rid))
    lead_idx = hdr.sig_name.index('ii')
    rec = wfdb.rdrecord(str(data_dir / rid), channels=[lead_idx])
    sig = rec.p_signal[:, 0].astype(np.float64)
    ann = wfdb.rdann(str(data_dir / rid), 'ii')
    gold = [int(s) for s, sym in zip(ann.sample, ann.symbol) if sym == 'N']
    return sig, gold


if __name__ == '__main__':
    main()
