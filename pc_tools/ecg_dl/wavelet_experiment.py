#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wavelet_experiment.py — 真实 ECG 上小波 vs 现固件显示链 (量化对比)
输入: ecg_real_052.ecgr (225.68Hz) -> 有理重采样 500Hz (与固件同)
候选链:
  A) 现固件显示链: 梳状(10抽头x2) -> HP4 -> LP40
  B) 小波基线去除 (db4 level7 近似系数置零重构) + LP40
  C) 小波基线去除 + 细节软阈值去噪 (Donoho-Johnstone) + LP40
指标 (测量滤波零相位, 不参与链):
  基线残差 RMS (<0.5Hz 分量), 高频噪声 RMS (40-100Hz), QRS 幅度保留比,
  QRS带(5-25Hz) RMS, 信噪比 = QRS带RMS/噪声RMS
"""
import sys, struct
from pathlib import Path
from fractions import Fraction
import numpy as np
from scipy.signal import resample_poly, butter, filtfilt, lfilter
import pywt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_deploy_match import _comb_filter

DATA = Path(__file__).resolve().parent / "data" / "real"
raw = (DATA / "ecg_real_052.ecgr").read_bytes()
n = struct.unpack_from("<I", raw, 18)[0]
dur = struct.unpack_from("<I", raw, 14)[0]
x = np.frombuffer(raw, dtype="<i2", count=n, offset=32).astype(np.float64) / 8000.0
fs_eff = n / dur
ratio = Fraction(int(round(500.0 / fs_eff * 10000)), 10000).limit_denominator(100000)
s500 = resample_poly(x, ratio.numerator, ratio.denominator)
s500 = s500 - np.mean(s500)
print(f"[data] n={len(s500)} @500Hz ({dur}s)")

# ---- 滤波器组 (系数与固件一致) ----
def biquad_filt(sig, b0, b1, b2, a1, a2):
    return lfilter([b0, b1, b2], [1.0, a1, a2], sig)
HP4 = (0.9650809863447340, -1.9301619726894681, 0.9650809863447340, -1.9289422632520332, 0.9313816821269024)
LP40B, LP40A = butter(2, 40.0 / 250.0, btype="low")

def chain_fw(sig):
    c = _comb_filter(sig)
    return biquad_filt(biquad_filt(c, *HP4), LP40B[0], LP40B[1], LP40B[2], LP40A[1], LP40A[2])

def chain_dwt_baseline(sig, level=7, denoise=False, wname="db4"):
    coeffs = pywt.wavedec(sig, wname, level=level)
    a7 = coeffs[0]; details = list(coeffs[1:])
    a7[:] = 0.0                    # 基线: 近似系数置零 (a7 频带 0~1.95Hz)
    if denoise:
        sigma = np.median(np.abs(details[-1])) / 0.6745
        thr = sigma * np.sqrt(2 * np.log(len(sig)))
        for i in range(len(details)):
            details[i] = pywt.threshold(details[i], thr, mode="soft")
    rec = pywt.waverec([a7] + details, wname)[:len(sig)]
    return biquad_filt(rec, LP40B[0], LP40B[1], LP40B[2], LP40A[1], LP40A[2])

# ---- 测量滤波器 (零相位, 只用于评估) ----
def band_rms(sig, lo, hi, fs=500.0, order=3):
    b, a = butter(order, [lo / (fs/2), hi / (fs/2)], btype="band")
    return float(np.sqrt(np.mean(filtfilt(b, a, sig) ** 2)))
def low_rms(sig, hi, fs=500.0, order=3):
    b, a = butter(order, hi / (fs/2), btype="low")
    return float(np.sqrt(np.mean(filtfilt(b, a, sig) ** 2)))

ref = chain_fw(s500)                      # 对照: 当前固件链
b_w = chain_dwt_baseline(s500, denoise=False)
c_w = chain_dwt_baseline(s500, denoise=True)
# 公平对照: 梳状保留, 仅 HP4 -> 小波基线去除 (隔离基线方案变量)
d_w = chain_dwt_baseline(_comb_filter(s500), denoise=False)
e_w = chain_dwt_baseline(_comb_filter(s500), denoise=True)
qrs_raw = band_rms(s500, 5, 25)

rows = []
for tag, sig in [("A 固件链 梳状+HP4+LP40", ref),
                 ("B 小波基线去除+LP40", b_w),
                 ("C 小波基线+软阈值去噪+LP40", c_w),
                 ("D 梳状+小波基线+LP40", d_w),
                 ("E 梳状+小波基线+软阈值+LP40", e_w)]:
    base = low_rms(sig, 0.5)
    noise = band_rms(sig, 40, 100)
    qrs = band_rms(sig, 5, 25)
    rows.append((tag, base, noise, qrs, qrs / max(noise, 1e-9), qrs / qrs_raw))
    print("%-28s | 基线残差 %.4f mV | 噪声40-100Hz %.4f mV | QRS带RMS %.4f mV | SNR %.1f dB | QRS保留 %.0f%%"
          % (tag, base * 1000, noise * 1000, qrs * 1000, 20 * np.log10(qrs / max(noise, 1e-9)), 100 * qrs / qrs_raw))
print("[note] 基线残差=处理后<0.5Hz剩余能量; 噪声=40-100Hz; QRS保留=相对原始")
