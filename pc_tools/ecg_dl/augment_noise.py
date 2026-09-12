#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""augment_noise.py — 心拍窗噪声增强 (M2 管线/M3 训练用, TH §106)
================================================================================
输入域: deploy_causal 心拍窗 (250 样本 @250Hz, 逐窗 z-score, std≈1)。
增强: 合成噪声 @500Hz → noise_lib.chain_shape 过设备链复刻 (comb×2 + HP0.05
+ LP40 + 2:1 抽取) → 叠加到心拍窗, 幅度按 SNR 相对单位方差定标
(std = 10^(-snr/20))。链整形是关键: 直接加满带噪声会给模型喂板上永远
看不到的频谱 (LP40 后 50Hz 工频残余仅 ~0.4%, noise_lib.selftest 实测)。

类型×梯度: {baseline_wander, respiratory, mains, emg, motion} ×
{30, 20, 10, 5, 0} dB, 均匀抽样; electrode_off/adc_quant 为非加性类型,
语料侧已覆盖, 训练侧不用于常规变体。

泄漏纪律: 本模块不取数; 调用方必须经 data/split_guard.py 取训练拍。
CLI 自检: 形状/统计断言 + 输出 JSON 报告 (无公共库取数, 不触守卫)。
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from noise_lib import synth_noise, chain_shape, SNR_LADDER

AUG_TYPES = ("baseline_wander", "respiratory", "mains", "emg", "motion")
AUG_SNRS = SNR_LADDER          # (30, 20, 10, 5, 0)
NORM_EPS = 1e-6


def _zscore(x):
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / np.maximum(sd, NORM_EPS)


def augment_beats(x, rng, variants_per_beat=1, types=AUG_TYPES, snrs=AUG_SNRS):
    """对 z-score 心拍窗逐拍独立生成带噪变体 (每拍独立抽 type/SNR/噪声实现)。

    x: (N, 250) float32 (已 z-score 或接近单位方差)
    返回 (x_aug, prov): x_aug (N*variants, 250) —— 每个变体重新 z-score,
    与部署链逐窗归一化语义一致; prov 为类型×SNR 计数 (多样性与配比审计)。
    """
    x = np.asarray(x, dtype=np.float32)
    n, win = x.shape
    outs = np.empty((n * variants_per_beat, win), dtype=np.float32)
    counts = {}
    for i in range(n):
        for v in range(variants_per_beat):
            ntype = str(rng.choice(types))
            snr = int(rng.choice(snrs))
            shaped = chain_shape(synth_noise(ntype, 2 * win, 500, rng)).astype(np.float32)
            assert len(shaped) == win, (len(shaped), win)
            shaped = (shaped - shaped.mean()) / (shaped.std() + NORM_EPS)
            amp = float(10 ** (-snr / 20.0))
            outs[i * variants_per_beat + v] = _zscore(x[i] + amp * shaped)
            key = "%s@%ddB" % (ntype, snr)
            counts[key] = counts.get(key, 0) + 1
    prov = {"per_beat_independent": True,
            "variants_per_beat": variants_per_beat,
            "counts": dict(sorted(counts.items()))}
    return outs, prov


def selftest():
    rng = np.random.default_rng(7)
    win = 250
    # 模拟心拍窗 (z-score 白噪代替, 仅形状/统计验证; 不取公共库数据)
    beats = rng.normal(0, 1, (32, win)).astype(np.float32)
    xaug, prov = augment_beats(beats, rng, variants_per_beat=2)
    assert xaug.shape == (64, win), xaug.shape
    assert prov["per_beat_independent"] and sum(prov["counts"].values()) == 64
    stds = xaug.std(axis=1)
    assert np.all(np.abs(stds - 1) < 1e-3), stds[:5]
    # 验证实际叠加幅度: 30dB 变体偏差应小, 0dB 应大
    x30, _ = augment_beats(beats, rng, variants_per_beat=1, snrs=(30,))
    x0, _ = augment_beats(beats, rng, variants_per_beat=1, snrs=(0,))
    d30 = float(np.mean(np.abs(x30 - beats)))
    d0 = float(np.mean(np.abs(x0 - beats)))
    assert d0 > d30 * 5, (d30, d0)
    out = {"selftest": "ok", "n_in": 32, "n_out": len(xaug),
           "mean_abs_dev_30db": round(d30, 4), "mean_abs_dev_0db": round(d0, 4),
           "types": AUG_TYPES, "snrs": list(AUG_SNRS)}
    print(json.dumps(out, ensure_ascii=False))
    (BASE / "corpus").mkdir(exist_ok=True)
    (BASE / "corpus" / "augment_noise_selftest.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    selftest()
