#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_causal_chain_consistency.py — P0-2 Step 2 一致性测试
================================================================================
验证 Python 复刻的"修正后因果链"与固件 filter.cpp 公式逐样本一致 (误差 <1e-9)。

测试项:
  T1  biquad 一致性: causal_hp_05_fs250 (scipy.lfilter) vs 固件 applyBiquad DF2T
      公式 (手写 double 循环, 零初始状态)。修正系数 fs=250。
  T2  全链一致性: corrected_deployment_chain (D3 链 + 因果 HP 0.5Hz) vs 手写参考
      链 (梳状 convolve + HP0.05/LP40 lfilter + 2:1 抽取 + 因果 HP DF2T 公式)。
  T3  系数语义交叉验证: butter(2,0.5,fs=250) == butter(2,1.0,fs=500) (归一化频率等价);
      且现有 fs=500 系数 @250Hz 有效截止 0.25Hz、修正系数 @250Hz 0.5Hz。

所有断言误差上限 1e-9 (double 精度下 scipy.lfilter 与手写 DF2T 应 ~1e-15)。
"""
import sys
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.preprocess import (
    causal_hp_05_fs250,
    apply_biquad_df2t,
    AI_HP_FS250_B0, AI_HP_FS250_B1, AI_HP_FS250_B2,
    AI_HP_FS250_A1, AI_HP_FS250_A2,
)
from eval_deploy_match import (
    corrected_deployment_chain,
    deployment_chain,
    HP_B, HP_A, LP_B, LP_A, N_WARMUP,
)

TOL = 1e-9


def firmware_biquad_ref(x, b0, b1, b2, a1, a2):
    """固件 filter.cpp applyBiquad 公式 (double 状态, 零初始, 逐样本).

        double w = (double)x - a1*w1 - a2*w2;
        double y = b0*w + b1*w1 + b2*w2;
        w2 = w1; w1 = w;
    """
    w1 = 0.0
    w2 = 0.0
    y = np.zeros(len(x), dtype=np.float64)
    for i in range(len(x)):
        w = float(x[i]) - a1 * w1 - a2 * w2
        y[i] = b0 * w + b1 * w1 + b2 * w2
        w2 = w1
        w1 = w
    return y


def _comb_ref(sig):
    """双级 10 抽头滑动平均 (与固件 applyCombFilter 同构)."""
    kernel = np.ones(10, dtype=np.float64) / 10.0
    y1 = np.convolve(sig, kernel, mode="full")[: len(sig)]
    y2 = np.convolve(y1, kernel, mode="full")[: len(y1)]
    return y2


def test_t1_biquad():
    print("\n[T1] biquad 一致性: causal_hp_05_fs250 vs 固件 DF2T 公式")
    rng = np.random.default_rng(42)
    x = rng.standard_normal(10000) * 0.4 + 0.2  # ECG 类信号 (非零均值 + 噪声)
    y_py = causal_hp_05_fs250(x)
    y_ref = firmware_biquad_ref(
        x, AI_HP_FS250_B0, AI_HP_FS250_B1, AI_HP_FS250_B2,
        AI_HP_FS250_A1, AI_HP_FS250_A2,
    )
    maxdiff = float(np.max(np.abs(y_py - y_ref)))
    print(f"    max|Δ| = {maxdiff:.3e} (TOL {TOL:.0e})")
    assert maxdiff < TOL, f"T1 FAIL: {maxdiff:.3e} >= {TOL:.0e}"
    print("    PASS")


def test_t2_full_chain():
    print("\n[T2] 全链一致性: corrected_deployment_chain vs 手写参考链 (360Hz 原生)")
    rng = np.random.default_rng(7)
    n = 20000
    sig = rng.standard_normal(n).astype(np.float64) * 0.3 + 0.15

    y_py = corrected_deployment_chain(sig, 360)

    # 手写参考: native→500 (resample_poly 25/18) → DC → comb → HP0.05+LP40(warmup)
    #            → 2:1 抽取 → 因果 HP 0.5Hz (固件 DF2T 公式)
    s500 = scipy_signal.resample_poly(sig, 25, 18)
    dc = s500 - np.mean(s500)
    combed = _comb_ref(dc)
    padded = np.concatenate([np.full(N_WARMUP, combed[0], dtype=np.float64), combed])
    hped = scipy_signal.lfilter(HP_B, HP_A, padded)
    lped = scipy_signal.lfilter(LP_B, LP_A, hped)
    filtered = lped[N_WARMUP:]
    decimated = filtered[0::2]
    y_ref = firmware_biquad_ref(
        decimated,
        AI_HP_FS250_B0, AI_HP_FS250_B1, AI_HP_FS250_B2,
        AI_HP_FS250_A1, AI_HP_FS250_A2,
    )

    # 长度对齐 (resample_poly 长度可能差 ±1)
    n_use = min(len(y_py), len(y_ref))
    maxdiff = float(np.max(np.abs(y_py[:n_use] - y_ref[:n_use])))
    print(f"    max|Δ| = {maxdiff:.3e} (TOL {TOL:.0e}), 长度 py={len(y_py)} ref={len(y_ref)}")
    assert maxdiff < TOL, f"T2 FAIL: {maxdiff:.3e} >= {TOL:.0e}"

    # 附加断言: corrected 链 == D3 链输出再经因果 HP (组合正确性)
    d3 = deployment_chain(sig, 360)
    recombined = causal_hp_05_fs250(d3)
    assert np.max(np.abs(recombined - y_py)) < TOL, "corrected != D3+causalHP (组合错误)"
    print("    PASS (含组合正确性断言: corrected == D3 输出 + 因果 HP)")


def test_t3_coeff_semantics():
    print("\n[T3] 系数语义交叉验证")
    # butter(2,0.5,fs=250) 应精确等于 butter(2,1.0,fs=500) (归一化频率等价)
    b250, a250 = scipy_signal.butter(2, 0.5, btype='high', fs=250)
    b500, a500 = scipy_signal.butter(2, 1.0, btype='high', fs=500)
    assert np.max(np.abs(b250 - b500)) < 1e-15, "butter(2,0.5,fs=250) != butter(2,1.0,fs=500)"
    assert np.max(np.abs(a250 - a500)) < 1e-15

    # 修正系数应等于 butter(2,0.5,fs=250) 的 b/a
    assert abs(AI_HP_FS250_B0 - b250[0]) < 1e-15
    assert abs(AI_HP_FS250_A1 - a250[1]) < 1e-15

    # 有效截止: 现有 fs=500 系数 @250Hz → 0.25Hz; 修正系数 @250Hz → 0.5Hz
    def cutoff_hz(b, a, fs):
        freqs, h = scipy_signal.freqz(b, a, worN=2 ** 20, fs=fs)
        mag2 = np.abs(h) ** 2
        for i in range(len(mag2) - 1):
            if mag2[i] <= 0.5 <= mag2[i + 1]:
                return float(freqs[i])
        return None

    b_old, a_old = scipy_signal.butter(2, 0.5, btype='high', fs=500)
    fc_old = cutoff_hz(b_old, a_old, 250)
    fc_new = cutoff_hz(b250, a250, 250)
    print(f"    现有系数 @250Hz 有效截止 = {fc_old:.4f} Hz (应 0.25)")
    print(f"    修正系数 @250Hz 有效截止 = {fc_new:.4f} Hz (应 0.5)")
    assert abs(fc_old - 0.25) < 0.01, f"现有系数截止 {fc_old} != 0.25"
    assert abs(fc_new - 0.5) < 0.01, f"修正系数截止 {fc_new} != 0.5"
    print("    PASS")


def main():
    print("=" * 70)
    print("test_causal_chain_consistency.py — P0-2 因果链一致性测试")
    print("=" * 70)
    test_t1_biquad()
    test_t2_full_chain()
    test_t3_coeff_semantics()
    print("\n" + "=" * 70)
    print("ALL CONSISTENCY TESTS PASSED (T1-T3, 误差 <1e-9)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
