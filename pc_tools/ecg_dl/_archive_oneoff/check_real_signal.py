"""独立测量真实信号质量 (不经固件 SQI): 从 raw_standard.csv 的 clean 列算
QRS 带 (5-25Hz) 能量 vs 噪声带 (40-100Hz), R 峰周期性 (自相关), 峰噪比。"""
import numpy as np
import re
from scipy.signal import butter, filtfilt, resample_poly
from fractions import Fraction

import sys
from pathlib import Path
lines = open(str(Path(__file__).resolve().parent.parent / "serial" / "raw_standard.csv"), encoding="utf-8").read().splitlines()
clean = []
for l in lines:
    l = l.strip()
    m = re.match(r"^(-?\d+\.\d+),", l)
    if m:
        clean.append(float(m.group(1)))
x = np.array(clean, dtype=np.float64)
# CSV 是 100Hz 抽取 (每5帧), 但时间轴按此; 只需相对 SNR 与峰噪比, 不要求精确 fs
fs_csv = 100.0
print(f"[raw] n={len(x)} (~{len(x)/fs_csv:.0f}s @100Hz) pp={x.max()-x.min():.3f}V std={x.std():.4f}V")

def band_rms(sig, lo, hi, fs):
    b, a = butter(3, [lo/(fs/2), hi/(fs/2)], btype="band")
    return float(np.sqrt(np.mean(filtfilt(b, a, sig) ** 2)))

qrs = band_rms(x, 5, 25, fs_csv)
noise2 = band_rms(x, 40, 48, fs_csv)   # 100Hz CSV Nyquist=50, 噪声带 40-48
print(f"[quality] QRS带(5-25Hz) RMS={qrs*1000:.2f} mV | 噪声带(40-48Hz) RMS={noise2*1000:.2f} mV")
print(f"[snr] SNR(QRS/噪声40-48) = {20*np.log10(qrs/max(noise2,1e-9)):.1f} dB")

# 带通后 R 峰
b, a = butter(2, [5, 25], btype="band", fs=fs_csv)
xf = filtfilt(b, a, x)
ac = np.correlate(xf - xf.mean(), xf - xf.mean(), mode="full")[len(xf)-1:]
lo, hi = int(0.4*fs_csv), int(1.5*fs_csv)
lag = np.argmax(ac[lo:hi]) + lo
print(f"[periodicity] 自相关峰 lag={lag} 样本 (0.4-1.5s 范围) = {lag/fs_csv:.2f}s -> ~{60/(lag/fs_csv):.0f} BPM")
print(f"[periodicity] 自相关峰强度 = {ac[lag]/ac[0]:.3f} (越接近1越有规律)")
# R 峰幅度 vs 基线噪声
from scipy.signal import find_peaks
pk, _ = find_peaks(xf, distance=int(0.35*fs_csv), height=np.median(np.abs(xf))*2)
print(f"[peaks] find_peaks 检出 {len(pk)} 个峰 (height=2x中位) -> {len(pk)/(len(xf)/fs_csv)*60:.0f} BPM 等效")
