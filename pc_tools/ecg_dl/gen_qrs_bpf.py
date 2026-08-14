"""QRS 带通 8-25Hz 系数 (butter 2阶 x2, fs=500)"""
from scipy.signal import butter
for tag, b, a in [("LP25", *butter(2, 25/250, btype='low')), ("HP8", *butter(2, 8/250, btype='high'))]:
    print(f"#define QRS_{tag}_A1  {a[1]:.6f}f")
    print(f"#define QRS_{tag}_A2  {a[2]:.6f}f")
    print(f"#define QRS_{tag}_B0  {b[0]:.6f}f")
    print(f"#define QRS_{tag}_B1  {b[1]:.6f}f")
    print(f"#define QRS_{tag}_B2  {b[2]:.6f}f")
# 检查 R(窄 12ms Gaussian) vs T(48ms Gaussian) 在 8-25Hz 带内的能量保留
import numpy as np
for name, sigma_samp in [("R(窄12ms)", 6), ("T(48ms)", 24)]:
    sig = np.exp(-np.arange(-200, 201)**2 / (2*sigma_samp**2))
    # 频谱能量在 8-25Hz 的比例
    N = len(sig)
    f = np.fft.rfftfreq(N, 1/500)
    S = np.abs(np.fft.rfft(sig))**2
    band = ((f >= 8) & (f <= 25)).sum() / ((f >= 0.5) & (f <= 40)).sum()
    tot = S.sum()
    inband = S[(f>=8)&(f<=25)].sum()
    print(f"{name}: 8-25Hz 带内能量占比 = {inband/tot*100:.1f}%")
