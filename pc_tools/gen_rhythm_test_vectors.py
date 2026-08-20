#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_rhythm_test_vectors.py - 心律异常合成测试向量生成器

生成 500Hz、单位 V、float32 的 ECG 测试信号，用于：
  1. PC 端 AI 离线验证:  python 07_pc_inference.py --source file --input test_vectors/vf.npy
  2. 固件回放扩展参考: 输出 CSV/npy，可仿照 make_replay_data.py 转成 C 数组

依赖: numpy（可选 scipy 用于重采样，本脚本纯合成不依赖）

输出目录: pc_tools/ecg_dl/test_vectors/
   normal.npy / normal.csv     正常窦性 75 bpm (阴性对照)
   veb.npy    / veb.csv        室性早搏 VEB (宽大 QRS 偶发)
   vf.npy     / vf.csv         室颤 VF (4~9Hz 混沌)
   af.npy     / af.csv         房颤 AF (RR 随机 0.5~1.2s)
   asystole.npy / asystole.csv 停搏 (10s 正常 + 6s 直线 + 10s 正常)
   tachy.npy  / tachy.csv      过速 200 bpm
   brady.npy  / brady.csv      过缓 30 bpm
   flatline.npy / flatline.csv 纯零线 + 极低噪声
"""

import numpy as np
from pathlib import Path

FS = 500  # Hz

# ---------- P-QRS-T 模板参数 (与固件 ecg_simulator 一致) ----------
CYCLE = 400  # 75 bpm @500Hz 默认周期（仅 normal 用；其余段按需要重设 RR）

def gaussian(x, amp, center, sigma):
    t = (x - center) / sigma
    return amp * np.exp(-0.5 * t * t)

def beat(rr_samples: int, fs: int = FS, abnormal: bool = False, seed: int = 0):
    """生成一个心拍，长度 rr_samples。
    abnormal=True 时生成宽大畸形 QRS（模拟 VEB）。"""
    n = int(rr_samples)
    t = np.arange(n) / fs
    # 正常波形归一化到 0~1 周期
    x = t / (n / fs)  # 0..1
    sig = (
        gaussian(x, 0.25, 0.18, 0.030)
        + gaussian(x, -0.10, 0.30, 0.020)
        + gaussian(x, 1.20, 0.33, 0.015)
        + gaussian(x, -0.15, 0.37, 0.025)
        + gaussian(x, 0.30, 0.55, 0.060)
    )
    if abnormal:
        # VEB: QRS 增宽 (sigma 放大 2.5 倍)、幅度增大、T 波反向
        sig = (
            gaussian(x, 0.25, 0.18, 0.030)
            + gaussian(x, -0.30, 0.30, 0.050)
            + gaussian(x, 1.80, 0.33, 0.040)
            + gaussian(x, -0.60, 0.40, 0.060)
            + gaussian(x, -0.40, 0.60, 0.100)
        )
    rng = np.random.default_rng(seed)
    sig += rng.normal(0, 0.005, n)  # 微小系统噪声
    return sig.astype(np.float32)

def concat_beats(rr_list, abnormal_indices=(), seed=0):
    """按 RR 样本数列表拼接心拍。abnormal_indices 指定哪些拍为 VEB。"""
    segs = []
    for i, rr in enumerate(rr_list):
        segs.append(beat(rr, abnormal=(i in abnormal_indices), seed=seed + i))
    return np.concatenate(segs).astype(np.float32)

def save(vec: np.ndarray, name: str, out_dir: Path):
    vec = vec.astype(np.float32)
    np.save(out_dir / f"{name}.npy", vec)
    np.savetxt(out_dir / f"{name}.csv", vec, fmt="%.6f")
    print(f"  {name:10s} {len(vec):6d} samples ({len(vec)/FS:6.1f}s)  amp={np.abs(vec).max():.3f}V")

def main():
    out_dir = Path(__file__).resolve().parent / "ecg_dl" / "test_vectors"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    print("生成正常窦性 75bpm (RR=0.8s, 30s)...")
    save(concat_beats([400] * 38), "normal", out_dir)  # 0.8s*38≈30s

    print("生成室早 VEB (正常 8 拍 + 1 VEB 循环, 30s)...")
    rr = [400] * 9 + [520]  # 8 normal + 1 早搏(代偿间歇)
    rr_list = (rr * 6)[:76]  # ~30s
    save(concat_beats(rr_list, abnormal_indices=[8, 19, 30, 41, 52, 63]), "veb", out_dir)

    print("生成室颤 VF (4~9Hz 调频混沌, 20s)...")
    n = FS * 20
    t = np.arange(n) / FS
    freq = 4.0 + 5.0 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t))  # 4~9Hz
    phase = 2 * np.pi * np.cumsum(freq) / FS
    vf = 0.8 * np.sin(phase) * (1.0 + 0.4 * np.sin(2 * np.pi * 1.3 * t))
    vf += rng.normal(0, 0.03, n)
    save(vf, "vf", out_dir)

    print("生成房颤 AF (RR 随机 0.5~1.2s, 30s)...")
    rr_af = rng.uniform(0.5, 1.2, 36)  # 30s 约 36 拍
    save(concat_beats((rr_af * FS).astype(int), seed=100), "af", out_dir)

    print("生成停搏 (10s 正常 + 6s 直线 + 10s 正常)...")
    asys = np.concatenate([
        concat_beats([400] * 12),          # 10s 正常
        np.zeros(FS * 6, dtype=np.float32), # 6s 停搏
        concat_beats([400] * 12),          # 10s 正常
    ])
    save(asys, "asystole", out_dir)

    print("生成过速 200bpm (RR=0.3s, 40s)...")
    save(concat_beats([150] * 133), "tachy", out_dir)  # 0.3s*133≈40s

    print("生成过缓 30bpm (RR=2.0s, 40s)...")
    save(concat_beats([1000] * 20), "brady", out_dir)  # 2.0s*20=40s

    print("生成 Flatline (零线+极低噪声, 15s)...")
    nf = FS * 15
    flat = rng.normal(0, 0.001, nf).astype(np.float32)
    save(flat, "flatline", out_dir)

    print(f"\n完成。输出目录: {out_dir}")
    print("PC 端 AI 验证示例:")
    print(f"  python pc_tools/ecg_dl/07_pc_inference.py --source file --input {out_dir/'vf.npy'}")

if __name__ == "__main__":
    main()
