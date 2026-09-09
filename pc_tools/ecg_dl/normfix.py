#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""normfix.py — §105 逐拍归一化修复 (normfix) 共享参考实现
================================================================================
单一事实源: 训练预处理 / 数组重建 / 固件数值一致性验证都从本模块取公式与
常数。常数全部来自 models/deploy_match/normfix_spec_v1.json (先于任何训练
脚本定稿), 本文件不硬编码任何数值。

规格 (Option A, 详见 spec JSON):
  输入流 = 修正后因果链 250Hz 输出 (D3 + 因果 HP 0.5Hz@250Hz, 基线/漂移
  已由该级因果去除; 固件为 aiApplyFilter 后的逐样本流)。
  新增一级因果慢包络 (一阶 IIR 均方), 逐样本:
      e2[n] = e2[n-1] + ALPHA * (x[n]^2 - e2[n-1])      # 含当前样本
      y[n]  = x[n] / max(sqrt(e2[n]), SMIN)
  顺序固定: 先用"含当前样本"的包络再相除。无窗口均值减法 (均值已由上游
  因果 HP 去除)。包络初值 e2_init = S_INIT^2 (保守常数尺度, 收敛期 ≈3τ)。

与固件的对应 (Phase B): 固件在 ai_inference_push 内对每个抽取样本以同一
公式、同一常数 (float32) 归一化后写入环形缓冲, preprocess_samples 的窗口
z-score 移除。公式漂移 = 全链作废, 一致性验证容差 ≤1e-6 相对误差。
"""
import json
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

SPEC_PATH = Path(__file__).resolve().parent / "models" / "deploy_match" / "normfix_spec_v1.json"


def load_spec(path=SPEC_PATH):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    option = spec.get("option", "A")
    required = {
        "A": ("tau_s", "fs_hz", "alpha", "s_init", "smin"),
        "B": ("scale",),
    }.get(option)
    if required is None:
        raise KeyError(f"normfix spec 未知 option {option!r}: {path}")
    for key in required:
        if key not in spec["constants"]:
            raise KeyError(f"normfix spec 缺少常数 {key}: {path}")
    return spec


def normfix_stream_const(x250, spec):
    """Option B: 固定常数尺度归一化 y[n] = x[n] / SCALE (无状态, 单调仿射)。

    全局单调仿射变换, AUC 不变; 用于隔离 '窗口 z-score 抹幅度' 这一变量。
    """
    scale = float(spec["constants"]["scale"])
    return np.asarray(x250, dtype=np.float64) / scale


def normalize_stream(x250, spec):
    """按 spec.option 分派: A=因果慢包络, B=固定常数尺度。"""
    if spec.get("option") == "B":
        return normfix_stream_const(x250, spec)
    return normfix_stream(x250, spec)


def normfix_stream(x250, spec):
    """修正后因果链 250Hz 流 → normfix 归一化流 (逐样本, 因果, 单遍)。

    x250: float64 数组 (因果 HP 0.5Hz 之后的 250Hz 样本)。
    返回与输入等长的归一化流。每段流独立初始化包络 (训练侧 = 每记录;
    固件 = 每次上电/推理会话), 初值 e2 = s_init^2。
    """
    c = spec["constants"]
    alpha = float(c["alpha"])
    e2_init = float(c["s_init"]) ** 2
    smin = float(c["smin"])

    x = np.asarray(x250, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    keep = 1.0 - alpha
    # e2[n] = keep*e2[n-1] + alpha*x[n]^2, e2[-1] = e2_init。
    # lfilter 的 zi 约定 (zf = a1*y0 + ...) 使首输出即"含当前样本"的包络,
    # 与逐样本循环同一递推; float64 运算次序不同导致最大相对偏差 ~3e-15
    # (normfix_spec_selfcheck.log), 远小于固件一致性验证容差 1e-6。
    e2 = lfilter([alpha], [1.0, -keep], x * x, zi=[keep * e2_init])[0]
    scale = np.sqrt(e2)
    scale = np.where(scale > smin, scale, smin)
    return x / scale


def window_beats_normfix(stream250, r_idx, domain, beat_len):
    """在归一化流上做窗口提取; 窗口/边缘规则与 extract_beats_deploy 完全
    一致 (MIT: pad/skip-50%; INCART/PTB: strict skip), 唯一差异: 不做窗口
    z-score (归一化已在流级逐样本完成)。"""
    half = beat_len // 2
    n_sig = len(stream250)
    beats = []
    for ri in r_idx:
        if domain == "mit":
            start = max(0, ri - half)
            end = min(n_sig, ri + half)
            if end - start < beat_len * 0.5:
                continue
            beat = stream250[start:end].copy()
            if len(beat) < beat_len:
                pad_before = (beat_len - len(beat)) // 2
                pad_after = beat_len - len(beat) - pad_before
                beat = np.pad(beat, (pad_before, pad_after), mode="constant")
            elif len(beat) > beat_len:
                center = len(beat) // 2
                beat = beat[center - half:center + half]
        else:
            lo = ri - half
            hi = lo + beat_len
            if lo < 0 or hi > n_sig:
                continue
            beat = stream250[lo:hi].copy()
        beats.append(beat)
    if beats:
        return np.stack(beats, axis=0).astype(np.float32)
    return np.empty((0, beat_len), dtype=np.float32)
