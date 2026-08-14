"""量化分析 v2: 自相关心动周期 -> 有效采样率; 窗口化质量扫描; 精细 R 峰.
"""
import sys, struct
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

path = sys.argv[1]
data = open(path, "rb").read()
samples_n = struct.unpack_from("<I", data, 18)[0]
dur = struct.unpack_from("<I", data, 14)[0]
x = np.frombuffer(data, dtype="<i2", count=samples_n, offset=32).astype(np.float64) / 8000.0
nom_fs = samples_n / dur
print("nominal_eff_fs = %.2f Hz (samples/seconds)" % nom_fs)

# 带通 5-25Hz (零相位)
b, a = butter(2, [5.0, 25.0], btype="band", fs=nom_fs)
xf = filtfilt(b, a, x)

# 自相关: 心动周期 (滞后 0.35s~1.6s)
ac = np.correlate(xf - xf.mean(), xf - xf.mean(), mode="full")[len(xf) - 1:]
lo = int(0.35 * nom_fs); hi = int(1.6 * nom_fs)
lag = np.argmax(ac[lo:hi]) + lo
print("autocorr period = %.1f samples = %.3f s (at eff_fs)" % (lag, lag / nom_fs))

# R 峰 (按 eff_fs 重设参数)
pk, _ = find_peaks(xf, distance=int(nom_fs * 0.28), height=np.percentile(xf, 75))
rrs = np.diff(pk).astype(np.float64)
med_rr = float(np.median(rrs))
print("Rpeaks n=%d medianRR=%.1f samples (%.3f s) -> HR=%.1f bpm" % (
    len(pk), med_rr, med_rr / nom_fs, 60.0 / (med_rr / nom_fs)))

# 有效采样率推断: 串口 CSV 同期 HR ~73-81 bpm, 自相关周期应等于 RR
# eff_fs = RR_samples * HR_true / 60
for hr in (73, 76, 80):
    print("if true HR=%d: eff_fs = %.1f Hz" % (hr, med_rr * hr / 60))

# 窗口质量扫描 (10s 窗)
win = int(nom_fs * 10)
rows = []
for i0 in range(0, len(x) - win, win):
    seg = x[i0:i0 + win]
    rows.append((i0 / nom_fs, seg.max() - seg.min(), float((np.abs(seg) > 1.55).sum()) / len(seg)))
good = [r for r in rows if r[2] < 0.01]
print("windows total=%d | good(clip<1%%)=%d -> clean %.1fs" % (len(rows), len(good), len(good) * 10))
for r in rows:
    print("  t=%6.1fs pp=%5.2fV clip=%5.3f%s" % (r[0], r[1], r[2], "" if r[2] < 0.01 else "  <-- artifact"))
