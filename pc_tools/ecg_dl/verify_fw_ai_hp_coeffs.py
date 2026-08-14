#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_fw_ai_hp_coeffs.py — P0-2 Task 2: 固件 AI_HP_* 系数与训练侧因果链一致性验证
================================================================================
验证三点 (误差上限 1e-9, double 精度下应 ~1e-15):
  V1  固件 src/filter/filter.cpp 的 AI_HP_A1/A2/B0/B1/B2 宏 == 训练侧
      data/preprocess.py 的 AI_HP_FS250_* 常量 (即 ai_hp_coeffs_fs250.txt FS250 值);
  V2  固件 DF2T 公式 (applyBiquad: w=x-a1*w1-a2*w2; y=b0*w+b1*w1+b2*w2; 零初始状态)
      与 scipy.signal.lfilter (causal_hp_05_fs250 底层) 逐样本一致;
  V3  causal_hp_05_fs250 (训练侧 streaming) == 固件公式逐样本一致 (位级语义)。

用法 (WSL, 与训练环境一致):
  python3 verify_fw_ai_hp_coeffs.py
"""
import re
import sys
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.preprocess import (
    causal_hp_05_fs250,
    AI_HP_FS250_B0, AI_HP_FS250_B1, AI_HP_FS250_B2,
    AI_HP_FS250_A1, AI_HP_FS250_A2,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FW_CPP = REPO_ROOT / "src" / "filter" / "filter.cpp"

TOL = 1e-9


def parse_fw_macros():
    """从 filter.cpp 提取 AI_HP_A1/A2/B0/B1/B2 宏值 (完整 double 精度)."""
    src = FW_CPP.read_text(encoding="utf-8")
    pat = re.compile(r"#define\s+AI_HP_(\w+)\s+([-\d.eE+]+)")
    macros = {}
    for m in pat.finditer(src):
        name, val = m.group(1), float(m.group(2))
        if name in ("A1", "A2", "B0", "B1", "B2"):
            macros[name] = val
    return macros


def firmware_biquad_ref(x, b0, b1, b2, a1, a2):
    """固件 filter.cpp applyBiquad 公式 (double 状态, 零初始, 逐样本)."""
    w1 = 0.0
    w2 = 0.0
    y = np.zeros(len(x), dtype=np.float64)
    for i in range(len(x)):
        w = float(x[i]) - a1 * w1 - a2 * w2
        y[i] = b0 * w + b1 * w1 + b2 * w2
        w2 = w1
        w1 = w
    return y


def main():
    print("=" * 70)
    print("verify_fw_ai_hp_coeffs.py — 固件 AI_HP_* 系数与训练侧因果链一致性")
    print("=" * 70)

    fw = parse_fw_macros()
    assert len(fw) == 5, f"filter.cpp 应解析出 5 个 AI_HP_* 宏, 实得 {fw}"
    py = {
        "B0": AI_HP_FS250_B0, "B1": AI_HP_FS250_B1, "B2": AI_HP_FS250_B2,
        "A1": AI_HP_FS250_A1, "A2": AI_HP_FS250_A2,
    }

    # ---- V1: filter.cpp 宏 == Python 常量 ----
    print("\n[V1] filter.cpp AI_HP_* 宏 vs preprocess.py AI_HP_FS250_* 常量")
    maxdiff = 0.0
    for k in ["B0", "B1", "B2", "A1", "A2"]:
        d = abs(fw[k] - py[k])
        maxdiff = max(maxdiff, d)
        flag = "✓" if d < 1e-15 else "✗"
        print(f"    {k}: fw={fw[k]:.17g}  py={py[k]:.17g}  diff={d:.3e} {flag}")
    assert maxdiff < 1e-15, f"V1 FAIL: 固件宏与训练侧常量不一致 (max {maxdiff:.3e})"
    print(f"    max|Δ| = {maxdiff:.3e} < 1e-15  PASS")

    # ---- V2: 固件 DF2T 公式 vs scipy.lfilter (causal_hp_05_fs250 底层) ----
    print("\n[V2] 固件 DF2T 公式 vs scipy.signal.lfilter (causal_hp_05_fs250 底层)")
    rng = np.random.default_rng(42)
    x = rng.standard_normal(10000) * 0.4 + 0.2  # ECG 类 (非零均值 + 噪声)
    y_py = causal_hp_05_fs250(x)
    y_fw = firmware_biquad_ref(
        x, fw["B0"], fw["B1"], fw["B2"], fw["A1"], fw["A2"],
    )
    d2 = float(np.max(np.abs(y_py - y_fw)))
    print(f"    max|Δ| = {d2:.3e} (TOL {TOL:.0e})")
    assert d2 < TOL, f"V2 FAIL: {d2:.3e} >= {TOL:.0e}"
    print("    PASS (causal_hp_05_fs250 == 固件公式 逐样本一致)")

    # ---- V3: 系数语义 (修正系数 == butter(2,0.5,fs=250)) ----
    print("\n[V3] 系数语义: 固件宏 == butter(2, 0.5, 'high', fs=250)")
    b250, a250 = scipy_signal.butter(2, 0.5, btype='high', fs=250)
    d3 = max(
        abs(fw["B0"] - b250[0]), abs(fw["B1"] - b250[1]), abs(fw["B2"] - b250[2]),
        abs(fw["A1"] - a250[1]), abs(fw["A2"] - a250[2]),
    )
    print(f"    max|Δ| = {d3:.3e}")
    assert d3 < 1e-15, f"V3 FAIL: {d3:.3e}"
    print("    PASS")

    print("\n" + "=" * 70)
    print("ALL FIRMWARE AI_HP COEFF CONSISTENCY CHECKS PASSED (V1-V3, <1e-9)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
