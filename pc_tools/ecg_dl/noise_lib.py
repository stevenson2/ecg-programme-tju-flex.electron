#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""noise_lib.py — 多类型噪声合成 + SNR 定标 + 设备链整形 (M2, TH §106)
================================================================================
用途:
  1. 板上回放语料 (make_replay_corpus.py): 载波 + 噪声在 500Hz 原始域混合,
     板上过真实设备链 — 保真度由板子保证。
  2. 训练增强 (augment_noise.py): 噪声先经 PC 设备链复刻 (eval_deploy_match.
     corrected_deployment_chain 的组件) 整形, 再叠加到 z-score 心拍窗 —
     捕获部署链的频谱整形 (LP40 杀 50Hz 以上、HP 抠基线), 避免给模型喂
     板上永远看不到的满带噪声。

噪声类型 (合成, NSTDB 在本地库缺失 → 按简报全合成):
  baseline_wander  基线漂移 (0.15-0.5Hz 正弦族 + 慢随机游走)
  respiratory      呼吸调制 (0.25Hz 基线 + 载波幅度 ±12% 调制)
  mains            工频 50/100/150Hz (谐波 1/0.3/0.1)
  emg              肌电 (白噪 → butter 20-150Hz 带通)
  motion           运动伪迹 (平滑电报过程阶跃 + 1-3Hz 摆动)
  electrode_off    电极脱落瞬态 (饱和/平顶事件, 非SNR标定, 按事件数)
  adc_quant        ADC 量化 (后处理, 非加性, 按位数)

SNR 口径: 10*log10(P_signal / P_noise), 功率在 500Hz 原始域去均值后计算。
泄漏纪律: 本库只做信号处理, 不取数; 取数脚本必须过 data/split_guard.py。
"""
import sys
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

# 设备链复刻系数 (与 eval_deploy_match.py 同源, 与固件 filter.cpp 一致:
# HP 0.05Hz @500Hz + LP 40Hz @500Hz, 均为因果 direct-form)
HP_B = np.array([0.99955581, -1.99911162, 0.99955581], dtype=np.float64)
HP_A = np.array([1.0, -1.99911142, 0.99911182], dtype=np.float64)
LP_B = np.array([0.0461509, 0.0923018, 0.0461509], dtype=np.float64)
LP_A = np.array([1.0, -1.3072850, 0.4916968], dtype=np.float64)

NOISE_TYPES = ("baseline_wander", "respiratory", "mains", "emg", "motion",
               "electrode_off", "adc_quant")
SNR_LADDER = (30, 20, 10, 5, 0)   # dB 梯度


def _unit_power(x: np.ndarray) -> np.ndarray:
    """归一到单位方差 (去均值)。"""
    x = x - np.mean(x)
    p = np.mean(x * x)
    return x / np.sqrt(p + 1e-12)


def synth_baseline_wander(n, fs, rng):
    t = np.arange(n) / fs
    x = np.zeros(n)
    for f, a in [(0.15, 1.0), (0.30, 0.6), (0.50, 0.3)]:
        ph = rng.uniform(0, 2 * np.pi)
        x += a * np.sin(2 * np.pi * f * t + ph)
    walk = np.cumsum(rng.normal(0, 0.02, n))
    walk -= np.linspace(walk[0], walk[-1], n)   # 端点回零防趋势泄漏
    return _unit_power(x + walk)


def synth_respiratory(n, fs, rng):
    """呼吸: 0.25Hz 基线摆动 (载波幅度调制在 mix 阶段做)。"""
    t = np.arange(n) / fs
    ph = rng.uniform(0, 2 * np.pi)
    x = np.sin(2 * np.pi * 0.25 * t + ph) + 0.4 * np.sin(2 * np.pi * 0.40 * t + ph * 1.7)
    return _unit_power(x)


def synth_mains(n, fs, rng):
    t = np.arange(n) / fs
    ph = rng.uniform(0, 2 * np.pi)
    x = (np.sin(2 * np.pi * 50 * t + ph)
         + 0.3 * np.sin(2 * np.pi * 100 * t + ph * 2.1)
         + 0.1 * np.sin(2 * np.pi * 150 * t + ph * 0.7))
    return _unit_power(x)


def synth_emg(n, fs, rng):
    x = rng.normal(0, 1, n)
    sos = scipy_signal.butter(2, [20, 150], btype="band", fs=fs, output="sos")
    return _unit_power(scipy_signal.sosfilt(sos, x))


def synth_motion(n, fs, rng):
    """平滑电报过程: 平均每 2s 换一次随机目标幅度, 经 1Hz 低通, 叠加摆动。"""
    target_change_p = 0.5 / fs          # 平均 0.5 次/秒换目标
    steps = np.zeros(n)
    cur = 0.0
    for i in range(n):
        if rng.random() < target_change_p:
            cur = rng.normal(0, 1)
        steps[i] = cur
    b, a = scipy_signal.butter(2, 1.0, fs=fs)
    base = scipy_signal.lfilter(b, a, steps)
    t = np.arange(n) / fs
    sway = 0.5 * np.sin(2 * np.pi * rng.uniform(1.0, 3.0) * t + rng.uniform(0, 6.28))
    return _unit_power(base + sway)


def synth_noise(noise_type, n, fs, rng):
    fn = {
        "baseline_wander": synth_baseline_wander,
        "respiratory": synth_respiratory,
        "mains": synth_mains,
        "emg": synth_emg,
        "motion": synth_motion,
    }.get(noise_type)
    if fn is None:
        raise ValueError(f"非加性/事件型噪声请用专用函数: {noise_type}")
    return fn(n, fs, rng).astype(np.float32)


def mix_at_snr(signal, noise, snr_db):
    """在原始域按 SNR 混合 (双方都去均值后定标)。"""
    s = np.asarray(signal, dtype=np.float32) - np.mean(signal)
    nz = np.asarray(noise, dtype=np.float32) - np.mean(noise)
    ps = np.mean(s * s) + 1e-12
    pn = np.mean(nz * nz) + 1e-12
    gain = np.sqrt(ps / pn / (10 ** (snr_db / 10.0)))
    return s + gain * nz


def apply_electrode_off(signal, fs, rng, n_events=3, dur_range=(0.4, 1.5),
                        sat_value=1.8):
    """电极脱落瞬态: 随机时刻输出饱和偏置 (等效拔接头拉弧/半脱落)。"""
    y = np.array(signal, dtype=np.float32, copy=True)
    n = len(y)
    dur = n / fs
    for _ in range(n_events):
        t0 = rng.uniform(1.0, dur - 2.0)
        length = rng.uniform(*dur_range)
        i0, i1 = int(t0 * fs), min(int((t0 + length) * fs), n)
        if i1 <= i0:
            continue
        kind = rng.random()
        if kind < 0.5:
            y[i0:i1] = sat_value          # 饱和
        else:
            y[i0:i1] = 0.2                # 平顶 (脱落)
    return y


def apply_adc_quant(signal, bits, full_scale=4.0):
    q = full_scale / (2 ** bits)
    return np.round(np.asarray(signal, dtype=np.float32) / q) * q


def chain_shape(noise500, warmup=240):
    """PC 设备链复刻整形: comb×2 + HP0.5 + LP40 + 2:1 抽取 (500Hz→250Hz)。
    与 eval_deploy_match.deployment_chain 相同组件, 但不做重采样 (输入已是 500Hz)。"""
    x = np.asarray(noise500, dtype=np.float64)
    x = x - np.mean(x)
    kernel = np.ones(10, dtype=np.float64) / 10.0
    y1 = np.convolve(x, kernel, mode="full")[: len(x)]
    y2 = np.convolve(y1, kernel, mode="full")[: len(x)]
    padded = np.concatenate([np.zeros(warmup), y2])
    hped = scipy_signal.lfilter(HP_B, HP_A, padded)
    lped = scipy_signal.lfilter(LP_B, LP_A, hped)
    return lped[warmup:][0::2].astype(np.float32)


def selftest():
    rng = np.random.default_rng(0)
    fs, n = 500, 5000
    for t in ("baseline_wander", "respiratory", "mains", "emg", "motion"):
        x = synth_noise(t, n, fs, rng)
        assert x.shape == (n,) and abs(np.std(x) - 1) < 1e-3, t
    # SNR 混合精度
    sig = synth_noise("emg", n, fs, rng) * 0.5
    nz = synth_noise("mains", n, fs, rng)
    mix = mix_at_snr(sig, nz, 10.0)
    ps = np.var(sig)
    pn = np.var(mix - sig)
    got = 10 * np.log10(ps / pn)
    assert abs(got - 10.0) < 0.01, got
    # 链整形: 50Hz 工频过链后应大幅衰减 (LP40)
    nz50 = synth_noise("mains", 2 * n, fs, rng)
    shaped = chain_shape(nz50)
    assert len(shaped) == n
    att = np.std(shaped) / np.std(nz50)
    assert att < 0.2, f"50Hz 衰减不足: {att}"
    # 电极脱落/量化
    y = apply_electrode_off(np.zeros(n, dtype=np.float32), fs, rng, n_events=3)
    assert np.any(np.abs(y) > 1.0)
    q = apply_adc_quant(np.linspace(-1, 1, 100, dtype=np.float32), 8)
    assert np.allclose(q[0], np.round(q[0]), atol=0.005)
    print("noise_lib selftest OK (含 50Hz 链衰减=%.3f)" % att)


if __name__ == "__main__":
    selftest()
