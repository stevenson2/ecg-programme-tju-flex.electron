"""显示链 HP 4Hz 系数 (Butterworth 2阶高通, fs=500) + 幅度响应检查"""
from scipy.signal import butter
import numpy as np

b, a = butter(2, 4.0 / 250.0, btype="high")
print("=== HP 4Hz fs=500 ===")
print("#define DISP_HP4_A1  %.16f" % a[1])
print("#define DISP_HP4_A2  %.16f" % a[2])
print("#define DISP_HP4_B0  %.16f" % b[0])
print("#define DISP_HP4_B1  %.16f" % b[1])
print("#define DISP_HP4_B2  %.16f" % b[2])
for f in (0.3, 1.0, 4.0, 10.0, 25.0):
    w = 2 * np.pi * f / 500
    z = np.exp(1j * w)
    H = abs((b[0] + b[1]*z + b[2]*z*z) / (1 + a[1]*z + a[2]*z*z))
    print(f"  |H({f}Hz)| = {H:.4f} ({20*np.log10(H):+.1f} dB)")
