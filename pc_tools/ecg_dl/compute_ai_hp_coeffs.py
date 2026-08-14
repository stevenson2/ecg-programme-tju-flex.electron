#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_ai_hp_coeffs.py — P0-2 Step 1: 计算 AI 输入链 HP 0.5Hz 修正系数
=================================================================================
背景 (P0-2 因果链修正):
  固件 src/filter/filter.cpp 的 AI_HP_* 系数是 butter(2, 0.5, 'high', fs=500) 设计,
  但 AI 链经 2:1 抽取后实际为 250Hz → 有效截止 0.25Hz (而非设计 0.5Hz)。
  修正: 用 butter(2, 0.5, 'high', fs=250) 设计系数, 作为固件 AI_HP_* 的修正规格,
  同时供训练侧复刻"修正后因果链" (causal HP 0.5Hz @250Hz)。

本脚本:
  1. 计算 butter(2, 0.5, 'high', fs=500) 与 butter(2, 0.5, 'high', fs=250) 的 b/a;
  2. 与固件 filter.cpp 现有 AI_HP_* 系数比对 (应精确匹配 fs=500, diff ~1e-16);
  3. 量化现有系数在 250Hz 链上的有效截止 (应 ≈0.25Hz);
  4. 写出修正系数规格文件 ai_hp_coeffs_fs250.txt (供固件侧 + 训练侧共用)。
"""
import sys
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

ROOT = Path(__file__).resolve().parent
SPEC_OUT = ROOT / "ai_hp_coeffs_fs250.txt"

# 固件 filter.cpp 现有 AI_HP_* 系数 (fs=500 设计, 2026-08-10)
FW_AI_HP = {
    "A1": -1.9911142922016536,
    "A2": 0.99115359586893537,
    "B0": 0.99556697201764721,
    "B1": -1.9911339440352944,
    "B2": 0.99556697201764721,
}


def butter_hp_coeffs(cutoff, fs, order=2):
    """butter(order, cutoff, 'high', fs=fs) → (B, A) 完整 double 精度.

    返回 dict: {'A1','A2','B0','B1','B2'} (a0 归一化为 1, 与 filter.cpp 宏同构).
    """
    b, a = scipy_signal.butter(order, cutoff, btype='high', fs=fs)
    # scipy 返回 a[0]==1.0 归一化; b=[b0,b1,b2], a=[1,a1,a2]
    assert abs(a[0] - 1.0) < 1e-15, "a0 应为 1"
    return {
        "B0": float(b[0]), "B1": float(b[1]), "B2": float(b[2]),
        "A1": float(a[1]), "A2": float(a[2]),
    }


def effective_cutoff_hz(b, a, fs):
    """数值求 -3dB 截止频率 (Hz): 在 [0, fs/2] 上找 |H|² = 0.5 的点.

    高通: |H| 从 DC(=0) 单调上升到通带(=1), -3dB 点即上升沿穿过 0.5 处。
    """
    freqs, h = scipy_signal.freqz(b, a, worN=2 ** 20, fs=fs)
    mag2 = np.abs(h) ** 2
    cross = None
    for i in range(len(mag2) - 1):
        # 上升沿: mag2 从 <0.5 变为 >=0.5
        if mag2[i] <= 0.5 <= mag2[i + 1]:
            cross = i
            break
    if cross is None:
        return float(freqs[0])
    f0, f1 = freqs[cross], freqs[cross + 1]
    m0, m1 = mag2[cross], mag2[cross + 1]
    fcut = f0 + (f1 - f0) * (0.5 - m0) / (m1 - m0)
    return float(fcut)


def fmt(c):
    """17 位有效数字 (double 完整精度, 与 filter.cpp 宏同精度)."""
    return f"{c:.17g}"


def main():
    print("=" * 70)
    print("P0-2 Step 1: AI 输入链 HP 0.5Hz 修正系数计算")
    print("=" * 70)

    c500 = butter_hp_coeffs(0.5, 500)
    c250 = butter_hp_coeffs(0.5, 250)

    # --- 1. 验证现有 AI_HP_* = butter(2,0.5,fs=500) ---
    print("\n[1] 现有固件 AI_HP_* 系数 vs butter(2, 0.5, fs=500):")
    max_diff_500 = 0.0
    for k in ["B0", "B1", "B2", "A1", "A2"]:
        d = abs(FW_AI_HP[k] - c500[k])
        max_diff_500 = max(max_diff_500, d)
        flag = "✓" if d < 1e-12 else "✗"
        print(f"    {k}: 固件={FW_AI_HP[k]:.17g}  butter500={c500[k]:.17g}  diff={d:.3e} {flag}")
    print(f"    最大 diff = {max_diff_500:.3e}")

    # --- 2. 修正系数 (fs=250) ---
    print("\n[2] 修正系数 butter(2, 0.5, fs=250):")
    for k in ["B0", "B1", "B2", "A1", "A2"]:
        print(f"    {k} = {fmt(c250[k])}")

    # --- 3. 有效截止频率 ---
    b500 = [c500["B0"], c500["B1"], c500["B2"]]
    a500 = [1.0, c500["A1"], c500["A2"]]
    b250 = [c250["B0"], c250["B1"], c250["B2"]]
    a250 = [1.0, c250["A1"], c250["A2"]]
    fc_500_on500 = effective_cutoff_hz(b500, a500, 500)
    fc_500_on250 = effective_cutoff_hz(b500, a500, 250)   # 现有系数挂在 250Hz 链
    fc_250_on250 = effective_cutoff_hz(b250, a250, 250)   # 修正系数挂在 250Hz 链
    print("\n[3] 有效截止频率 (-3dB):")
    print(f"    现有系数 (fs=500 设计) @500Hz 链 : {fc_500_on500:.4f} Hz  (设计 0.5Hz)")
    print(f"    现有系数 (fs=500 设计) @250Hz 链 : {fc_500_on250:.4f} Hz  (→ 0.25Hz, BUG)")
    print(f"    修正系数 (fs=250 设计) @250Hz 链 : {fc_250_on250:.4f} Hz  (设计 0.5Hz)")

    # --- 4. 写规格文件 ---
    lines = [
        "# AI 输入链 HP 0.5Hz 修正系数 (P0-2 Step 1)",
        "# 背景: 固件 AI_HP_* 原为 butter(2,0.5,'high',fs=500) 设计, 但 AI 链经 2:1 抽取后",
        "#        实际 250Hz → 有效截止 0.25Hz (非 0.5Hz)。本文件给出 fs=250 设计的修正系数,",
        "#        使因果 HP 0.5Hz 在 250Hz 链上真正实现 0.5Hz 截止。",
        "# 结构: 二阶 Butterworth 高通, a0 归一化为 1, 与 filter.cpp AI_HP_* 宏同构。",
        "#        B0/B1/B2 = 分子系数, A1/A2 = 分母系数 (DF2T: w=x-a1*w1-a2*w2; y=b0*w+b1*w1+b2*w2)。",
        "# 生成: compute_ai_hp_coeffs.py (scipy.signal.butter(2, 0.5, 'high', fs=250))",
        "",
        "# ==== 修正系数 (fs=250) ====",
        f"#define AI_HP_FS250_B0  {fmt(c250['B0'])}",
        f"#define AI_HP_FS250_B1  {fmt(c250['B1'])}",
        f"#define AI_HP_FS250_B2  {fmt(c250['B2'])}",
        f"#define AI_HP_FS250_A1  {fmt(c250['A1'])}",
        f"#define AI_HP_FS250_A2  {fmt(c250['A2'])}",
        "",
        "# ==== 现有系数 (fs=500 设计, 供对照) ====",
        f"#define AI_HP_FS500_B0  {fmt(c500['B0'])}",
        f"#define AI_HP_FS500_B1  {fmt(c500['B1'])}",
        f"#define AI_HP_FS500_B2  {fmt(c500['B2'])}",
        f"#define AI_HP_FS500_A1  {fmt(c500['A1'])}",
        f"#define AI_HP_FS500_A2  {fmt(c500['A2'])}",
        "",
        "# ==== 验证记录 ====",
        f"# 现有系数 vs butter(2,0.5,fs=500): 最大 diff = {max_diff_500:.3e}",
        f"# 现有系数 @250Hz 链有效截止: {fc_500_on250:.4f} Hz (应为 0.25Hz)",
        f"# 修正系数 @250Hz 链有效截止: {fc_250_on250:.4f} Hz (应为 0.5Hz)",
        "",
    ]
    SPEC_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[4] 规格文件已写入: {SPEC_OUT}")

    # 供 Python 侧直接 import 的系数 (写入 spec 同时打印)
    print("\n[5] Python 常量 (复制到 data/preprocess.py):")
    print(f'    AI_HP_FS250_B0 = {fmt(c250["B0"])}')
    print(f'    AI_HP_FS250_B1 = {fmt(c250["B1"])}')
    print(f'    AI_HP_FS250_B2 = {fmt(c250["B2"])}')
    print(f'    AI_HP_FS250_A1 = {fmt(c250["A1"])}')
    print(f'    AI_HP_FS250_A2 = {fmt(c250["A2"])}')

    # --- 6. 合理性断言 ---
    print("\n[6] 合理性断言:")
    assert max_diff_500 < 1e-12, "现有 AI_HP_* 应精确匹配 butter(2,0.5,fs=500)"
    assert abs(fc_500_on250 - 0.25) < 0.01, "现有系数 @250Hz 有效截止应 ≈0.25Hz"
    assert abs(fc_250_on250 - 0.5) < 0.01, "修正系数 @250Hz 有效截止应 ≈0.5Hz"
    print("    ✓ 全部通过 (现有=fs500 匹配 / 0.25Hz bug 确认 / 修正=0.5Hz)")
    print("\nDONE")


if __name__ == "__main__":
    main()
