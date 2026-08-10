#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECGR 样本记录生成器

根据 ECG 记录文件格式规范生成合成测试用的 .ecgr 文件，
用于离线链测试和云端 mock 服务器端到端验证。

参考: include/storage/ecg_recorder_format.h
文件布局: 32B 头部 + int16 样本流 (LE) + 可选的异常位图 (1 byte/s)
"""

import math
import struct
import sys
import os

# ======================== 常量 ========================
ECGR_MAGIC = b"ECGR"
ECGR_VERSION = 1
ECGR_HEADER_SIZE = 32
ECGR_FLAG_HAS_ABNORMAL_BITMAP = 0x01

# 默认录制参数
DEFAULT_SAMPLE_RATE = 250
DEFAULT_DURATION_SEC = 60
DEFAULT_ABNORMAL_SEC = 5
DEFAULT_START_UNIX = 1700000000


def generate_ecg_samples(duration_sec, sample_rate, seed=42):
    """生成合成心电样本 (int16 LE)。

    模拟类窦性节律信号: 基频约 72 BPM (1.2 Hz) 正弦波 +
    每 ~0.83s 一个 R-peak 尖峰，叠加少量高频噪声。
    确定性生成，种子固定。
    """
    import random
    rng = random.Random(seed)

    total_samples = duration_sec * sample_rate
    # 心率 ~72 BPM => 1.2 Hz; 每拍约 0.833s => 每 (1.2 * duration_sec) 拍
    beat_interval_samples = int(sample_rate / 1.2)  # ~208 点/拍

    samples = []
    for i in range(total_samples):
        t = i / sample_rate  # 秒

        # 基频窦性节律: 1.2 Hz 正弦波，幅值 ~800 ADC 单位
        base = 800.0 * math.sin(2.0 * math.pi * 1.2 * t)

        # R-peak: 每 beat_interval_samples 出现一次尖峰
        phase_in_beat = i % beat_interval_samples
        # QRS 尖峰宽度 ~20 点 (80ms @250Hz)
        if phase_in_beat < 20:
            # 高斯形 R-peak，峰值为 3000
            peak = 3000.0 * math.exp(-0.5 * ((phase_in_beat - 10) / 5.0) ** 2)
        else:
            peak = 0.0

        # T 波: 在 R-peak 后约 60-100 点处，幅值 ~400
        t_wave = 0.0
        tw_start = int(beat_interval_samples * 0.3)
        tw_end = int(beat_interval_samples * 0.55)
        if tw_start <= phase_in_beat < tw_end:
            progress = (phase_in_beat - tw_start) / (tw_end - tw_start)
            t_wave = 400.0 * math.sin(math.pi * progress)

        # 高频噪声 (少量)
        noise = rng.gauss(0, 30.0)

        value = base + peak + t_wave + noise
        # 限制在 int16 范围
        value = max(-32768, min(32767, round(value)))
        samples.append(value)

    return samples


def generate_abnormal_bitmap(duration_sec, abnormal_seconds, start_sec=20):
    """生成异常位图: 在指定的 seconds (start_sec .. start_sec+abnormal_seconds-1)
    处置 1，其余置 0。
    """
    bitmap = bytearray(duration_sec)
    for sec in range(start_sec, min(start_sec + abnormal_seconds, duration_sec)):
        bitmap[sec] = 1
    return bytes(bitmap)


def write_ecgr(filepath, sample_rate, start_unix, duration_sec, total_samples,
               abnormal_seconds, samples, bitmap):
    """将完整 ECGR 记录写入文件。"""
    flags = ECGR_FLAG_HAS_ABNORMAL_BITMAP  # bit0=1: 包含异常位图
    reserved = bytes(6)  # offset 26-31 保留零

    # 构建 32B 头部 (小端)
    header = struct.pack(
        "<4sBBIII I I6s",
        ECGR_MAGIC,              # offset 0-3: magic "ECGR"
        ECGR_VERSION,            # offset 4: version
        flags,                   # offset 5: flags
        sample_rate,             # offset 6-9: sampleRate u32 LE
        start_unix,              # offset 10-13: startUnix u32 LE
        duration_sec,            # offset 14-17: durationSec u32 LE
        total_samples,           # offset 18-21: totalSamples u32 LE
        abnormal_seconds,        # offset 22-25: abnormalSeconds u32 LE
        reserved                 # offset 26-31: reserved (6 bytes)
    )

    # 样本流: int16 LE
    samples_bytes = struct.pack(f"<{len(samples)}h", *samples)

    with open(filepath, "wb") as f:
        f.write(header)
        f.write(samples_bytes)
        f.write(bitmap)

    return os.path.getsize(filepath)


def main():
    """主入口: 生成 sample_60s.ecgr。"""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "sample_60s.ecgr")

    sample_rate = DEFAULT_SAMPLE_RATE
    duration_sec = DEFAULT_DURATION_SEC
    total_samples = duration_sec * sample_rate  # 60 * 250 = 15000
    abnormal_seconds = DEFAULT_ABNORMAL_SEC

    print(f"生成合成 ECG 样本: {total_samples} 点 @ {sample_rate}Hz ({duration_sec}s)")
    samples = generate_ecg_samples(duration_sec, sample_rate)

    print(f"生成异常位图: {abnormal_seconds}s 异常 (秒 {20}-{24})")
    bitmap = generate_abnormal_bitmap(duration_sec, abnormal_seconds, start_sec=20)

    print(f"写入文件: {output_path}")
    file_size = write_ecgr(output_path, sample_rate, DEFAULT_START_UNIX,
                           duration_sec, total_samples, abnormal_seconds,
                           samples, bitmap)

    # 验证文件大小
    expected_size = (
        ECGR_HEADER_SIZE          # 32
        + total_samples * 2        # 30000
        + duration_sec             # 60 (bitmap)
    )
    assert expected_size == 30092, f"expected_size 计算错误: {expected_size}"
    assert file_size == expected_size, (
        f"文件大小不匹配: 实际 {file_size} != 预期 {expected_size}"
    )

    print(f"文件大小: {file_size} bytes (预期 {expected_size})")
    print("验证通过: 32 + 30000 + 60 = 30092")


if __name__ == "__main__":
    main()
