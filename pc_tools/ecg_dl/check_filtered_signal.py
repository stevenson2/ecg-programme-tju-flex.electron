"""正确重测: 用 filtered 列 (用户所见) 测真实信号质量 + R 峰清晰度。
CSV 实际速率 = 2715 行/30s ≈ 90.5 Hz (每5帧@~450Hz? 实测按行数/秒)。"""
import numpy as np, re
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks

lines = open(str(Path(__file__).resolve().parent.parent / "serial" / "raw_standard2.csv"), encoding="utf-8").read().splitlines()
cols = [[], [], []]
for l in lines:
    l = l.strip()
    m = re.match(r"^(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),", l)
    if m:
        cols[0].append(float(m.group(1))); cols[1].append(float(m.group(2))); cols[2].append(float(m.group(3)))
clean = np.array(cols[0]); noisy = np.array(cols[1]); filt = np.array(cols[2])
# 实测 CSV 速率: 30s 2715 行 -> 90.5Hz (串口 100Hz 设计, 实际帧率约 452Hz? 按 90.5 算)
fs = 90.5
print(f"[cols] clean pp={clean.max()-clean.min():.3f}V | noisy pp={noisy.max()-noisy.min():.3f}V | filtered pp={filt.max()-filt.min():.3f}V")

def perr(sig, lo, hi):
    b, a = butter(2, [lo/(fs/2), hi/(fs/2)], btype="band")
    xf = filtfilt(b, a, sig)
    return xf, float(np.sqrt(np.mean(xf**2)))

xf, qrs_rms = perr(filt, 5, 20)
_, noise_rms = perr(filt, 25, 44)
print(f"[filtered] QRS带(5-20Hz) RMS={qrs_rms*1000:.2f} mV | 噪声带(25-44Hz) RMS={noise_rms*1000:.2f} mV | SNR={20*np.log10(qrs_rms/max(noise_rms,1e-9)):.1f} dB")

# 自相关周期 (filtered 列)
ac = np.correlate(xf - xf.mean(), xf - xf.mean(), mode="full")[len(xf)-1:]
lo, hi = int(0.4*fs), int(1.5*fs)
lag = np.argmax(ac[lo:hi]) + lo
print(f"[filtered] 自相关峰 lag={lag} 样本 = {lag/fs:.2f}s -> ~{60/(lag/fs):.0f} BPM, 强度={ac[lag]/ac[0]:.3f}")

# R 峰
pk, _ = find_peaks(xf, distance=int(0.3*fs), height=np.percentile(xf, 80))
print(f"[filtered] find_peaks 检出 {len(pk)} 峰 -> {len(pk)/(len(xf)/fs)*60:.0f} BPM 等效")
