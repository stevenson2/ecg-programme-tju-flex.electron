"""
验证和计算 ECG 滤波器系数 — 精确版
=====================================
功能:
  1. 验证现有 filter.cpp 系数
  2. 计算 v2.0 两级新陷波 + QRS BPF 的精确系数
  3. 生成可直接粘贴到 C 代码的宏定义
"""

import numpy as np
from scipy import signal
import math

FS = 250.0

def analyze(b, a, name, freqs=None):
    w, h = signal.freqz(b, a, fs=FS)
    mag = 20 * np.log10(np.abs(h) + 1e-12)
    if freqs is None:
        freqs = [0.5, 5, 10, 15, 30, 40, 48, 50, 52, 98, 100, 102]
    print(f"\n  {name}")
    for f in freqs:
        idx = np.argmin(np.abs(w - f))
        print(f"    {f:5.1f}Hz: {mag[idx]:+7.2f}dB")
    mask = mag > -3
    if np.any(mask):
        print(f"    -3dB通带: {w[mask][0]:.2f}~{w[mask][-1]:.2f}Hz")

def print_coeff(name, b, a):
    """格式化为C宏定义，5位小数"""
    print(f"#define {name}_B0  {b[0]:.5f}f")
    print(f"#define {name}_B1  {b[1]:.5f}f")
    print(f"#define {name}_B2  {b[2]:.5f}f")
    print(f"#define {name}_A1  {a[1]:.5f}f")
    print(f"#define {name}_A2  {a[2]:.5f}f")

print("="*70)
print(" ECG 滤波器系数精确计算与验证")
print("="*70)

# ============ 1. 生成并验证所有系数 ============
w0 = 2 * math.pi * 50 / FS
cos50 = math.cos(w0)
sin50 = math.sin(w0)

# Notch 50Hz Q20
a_Q20 = sin50 / (2*20)
n20_b = np.r_[1, -2*cos50, 1] / (1+a_Q20)
n20_a = np.r_[1, -2*cos50/(1+a_Q20), (1-a_Q20)/(1+a_Q20)]

# Notch 50Hz Q30
a_Q30 = sin50 / (2*30)
n30_b = np.r_[1, -2*cos50, 1] / (1+a_Q30)
n30_a = np.r_[1, -2*cos50/(1+a_Q30), (1-a_Q30)/(1+a_Q30)]

# Notch 100Hz Q15
w0_100 = 2*math.pi*100/FS
cos100 = math.cos(w0_100)
a_Q15 = math.sin(w0_100) / (2*15)
n100_b = np.r_[1, -2*cos100, 1] / (1+a_Q15)
n100_a = np.r_[1, -2*cos100/(1+a_Q15), (1-a_Q15)/(1+a_Q15)]

# HP 0.5Hz, LP 40Hz (from scipy)
hp_b, hp_a = signal.butter(2, 0.5, 'high', fs=FS, output='ba')
lp_b, lp_a = signal.butter(2, 40, 'low', fs=FS, output='ba')

# QRS BPF: LP 15Hz + HP 5Hz (two 2nd-order sections)
lp15_b, lp15_a = signal.butter(2, 15, 'low', fs=FS, output='ba')
hp5_b, hp5_a = signal.butter(2, 5, 'high', fs=FS, output='ba')

# ============ 2. 验证原系数 (filter.cpp) ============
print("\n\n>>> 原系数验证 <<<")
print(f"  HP 0.5Hz: B={hp_b[0]:.6f},{hp_b[1]:.6f},{hp_b[2]:.6f} A={hp_a[1]:.6f},{hp_a[2]:.6f}")
print(f"  LP 40Hz:  B={lp_b[0]:.6f},{lp_b[1]:.6f},{lp_b[2]:.6f} A={lp_a[1]:.6f},{lp_a[2]:.6f}")

# ============ 3. 验证新陷波系数 ============
print("\n\n>>> v2.0 新陷波系数（与filter.cpp对比）<<<")
print(f"  Notch50 Q20: B={n20_b[0]:.5f},{n20_b[1]:.5f},{n20_b[2]:.5f} A={n20_a[1]:.5f},{n20_a[2]:.5f}")
print(f"  Notch50 Q30: B={n30_b[0]:.5f},{n30_b[1]:.5f},{n30_b[2]:.5f} A={n30_a[1]:.5f},{n30_a[2]:.5f}")
print(f"  Notch100 Q15: B={n100_b[0]:.5f},{n100_b[1]:.5f},{n100_b[2]:.5f} A={n100_a[1]:.5f},{n100_a[2]:.5f}")

# ============ 4. 五级级联总响应 ============
print("\n\n>>> 五级级联总响应 <<<")
b_c = hp_b.copy()
a_c = hp_a.copy()
for b_i, a_i in [(lp_b, lp_a), (n20_b, n20_a), (n30_b, n30_a), (n100_b, n100_a)]:
    b_c = np.convolve(b_c, b_i)
    a_c = np.convolve(a_c, a_i)
analyze(b_c, a_c, "五级级联(HP→LP→N20→N30→N100)")

# ============ 5. QRS BPF 验证 ============
print("\n\n>>> QRS BPF 5~15Hz <<<")
b_bpf = np.convolve(lp15_b, hp5_b)
a_bpf = np.convolve(lp15_a, hp5_a)
analyze(b_bpf, a_bpf, "QRS BPF (LP15+HP5级联)")

# ============ 6. 生成C代码 ============
print("\n\n>>> 精确C代码宏定义 <<<")
print("""
/* =============================================
 *  生成: pc_tools/verify_filter_coeffs.py
 *  采样率: 250Hz, 精度: 5位小数
 * ============================================= */

/* ========== filter.cpp 修正 (v2.0) ========== */""")
print_coeff("NOTCH1", n20_b, n20_a)  # 50Hz Q20 (更新精确值)
print("\n/* 第4级: 50Hz 陷波 Q=30 */")
print_coeff("NOTCH2", n30_b, n30_a)
print("\n/* 第5级: 100Hz 陷波 Q=15 */")
print_coeff("NOTCH3", n100_b, n100_a)

print("""
/* ========== heartrate.cpp 新增 (v2.0) ========== */
/* QRS专用5~15Hz带通 - 第1节: 低通15Hz */""")
print_coeff("QRS_LP15", lp15_b, lp15_a)
print("\n/* QRS专用5~15Hz带通 - 第2节: 高通5Hz */")
print_coeff("QRS_HP5", hp5_b, hp5_a)