"""计算显示链 LP 系数 (Butterworth 2阶, fs=500): 40Hz (现用) 与 4Hz (试验)"""
from scipy.signal import butter

for fc in (40.0, 4.0):
    b, a = butter(2, fc / 250.0, btype="low")   # Wn = fc/(fs/2) = fc/250
    print(f"=== LP {fc:g}Hz fs=500 ===")
    print("#define DISP_LP_A1  %.16f" % a[1])
    print("#define DISP_LP_A2  %.16f" % a[2])
    print("#define DISP_LP_B0  %.16f" % b[0])
    print("#define DISP_LP_B1  %.16f" % b[1])
    print("#define DISP_LP_B2  %.16f" % b[2])
    # 验证: 40Hz 应与现有 LP_* 宏一致
    if abs(fc - 40.0) < 1e-9:
        print("(对照现有宏: A1 -1.3072850288493234 / A2 0.4918122372225752 / B0 0.046131802093312926)")
    # QRS 主频 10Hz 的幅度响应
    import numpy as np
    w10 = 2 * np.pi * 10 / 500
    z = np.exp(1j * w10)
    H = abs((b[0] + b[1]*z + b[2]*z*z) / (1 + a[1]*z + a[2]*z*z))
    print(f"    |H(10Hz)| = {H:.4f} (QRS 主频衰减 {20*np.log10(H):+.1f} dB)")
