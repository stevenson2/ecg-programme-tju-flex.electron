"""解析 .ecgr 真实采集记录: 头部 + 样本统计 + R峰 + 有效采样率 + 削顶比例 + 波形图.
用法: python3 analyze_real_record.py <ecgr文件> [输出前缀]
"""
import sys, struct, json
import numpy as np

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "real_analysis"

data = open(path, "rb").read()
magic, ver, flags = data[0:4], data[4], data[5]
rate = struct.unpack_from("<I", data, 6)[0]
start_unix = struct.unpack_from("<I", data, 10)[0]
dur = struct.unpack_from("<I", data, 14)[0]
samples_n = struct.unpack_from("<I", data, 18)[0]
abn_sec = struct.unpack_from("<I", data, 22)[0]
print("HEADER magic=%s ver=%d flags=0x%02x rate=%d start=%d dur=%d samples=%d abnSec=%d filesize=%d"
      % (magic, ver, flags, rate, start_unix, dur, samples_n, abn_sec, len(data)))

x = np.frombuffer(data, dtype="<i2", count=samples_n, offset=32).astype(np.float64) / 8000.0  # 8000 = V
print("SIGNAL n=%d mean=%.4f std=%.4f min=%.4f max=%.4f pp=%.4fV"
      % (len(x), x.mean(), x.std(), x.min(), x.max(), x.max() - x.min()))
clip_frac = float(((np.abs(x) > 1.55).sum()) / len(x))
print("CLIP frac(|x|>1.55V)=%.4f" % clip_frac)

# 标称 250Hz 下粗略 R 峰 (带通后峰值检测)
try:
    from scipy.signal import butter, filtfilt, find_peaks
    b, a = butter(2, [5.0, 25.0], btype="band", fs=250.0)
    xf = filtfilt(b, a, x)
    pk, _ = find_peaks(xf, distance=int(250 * 0.3), height=0.05)
    rrs = np.diff(pk).astype(np.float64)
    if len(rrs) > 0:
        med_rr = np.median(rrs)
        hr_250 = 60.0 / (med_rr / 250.0)
        print("RPEAKS n=%d medianRR=%.1f samples (250Hz假设) -> HR=%.1f bpm" % (len(pk), med_rr, hr_250))
        # 有效采样率: 参考 HR 73-80 (同会话串口 CSV), eff_fs = RR_samples * HR / 60
        print("EFFRATE if HR=73: %.1f Hz | if HR=76: %.1f | if HR=80: %.1f"
              % (med_rr * 73 / 60, med_rr * 76 / 60, med_rr * 80 / 60))
except ImportError as e:
    print("NO scipy:", e)

# 保存 10s 波形图 + 全览图
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    t10 = np.arange(0, min(len(x), 2500)) / 250.0
    axes[0].plot(t10, x[:2500], lw=0.6)
    axes[0].set_title("first 10s (nominal 250Hz)")
    axes[1].plot(np.arange(len(x)) / 250.0, x, lw=0.2)
    axes[1].set_title("full %ds" % dur)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=110)
    print("PLOT saved:", out + ".png")
except ImportError as e:
    print("NO matplotlib:", e)
