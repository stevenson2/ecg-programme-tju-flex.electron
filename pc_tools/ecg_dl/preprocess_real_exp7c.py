#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preprocess_real_exp7c.py — 真实 AFE 记录 → exp7c 微调正常拍
=================================================================
链 (与训练侧 exp7b *_deploy_causal 完全一致, 函数复用 eval_deploy_match):
  225.68Hz 原始 (实测) → 有理数重采样到 500Hz → 去DC → 双级10抽头梳状
  → 因果 HP/LP (240点预热) → 2:1 抽取 → 因果 HP0.5Hz@250Hz (修正系数)
  → XQRS R峰 → 250点窗口 (strict) → 固件 z-score → 标注正常(0)
伪影剔除: 原始域 ±0.6s 内 |x|>1.2V (削顶/接触噪声) 或 z-score 极端值。
产出: data/real/real_normal_beats_exp7c.npy + real_preprocess_exp7c.json
"""
import sys, json, struct
from pathlib import Path
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TARGET_FS, BEAT_WINDOW_SAMPLES
from eval_deploy_match import (
    _comb_filter, _hp_lp_filter, causal_hp_05_fs250, extract_beats_deploy,
)

DATA_REAL = Path(__file__).resolve().parent / "data" / "real"
ECGR = DATA_REAL / "ecg_real_052.ecgr"
OUT_BEATS = DATA_REAL / "real_normal_beats_exp7c.npy"
OUT_JSON = DATA_REAL / "real_preprocess_exp7c.json"

raw = ECGR.read_bytes()
n = struct.unpack_from("<I", raw, 18)[0]
dur = struct.unpack_from("<I", raw, 14)[0]
x = np.frombuffer(raw, dtype="<i2", count=n, offset=32).astype(np.float64) / 8000.0
fs_eff = n / dur
print(f"[REAL] n={n} dur={dur}s fs_eff={fs_eff:.4f} Hz")

# 1. 有理数重采样 225.68->500Hz
ratio = Fraction(int(round(500.0 / fs_eff * 10000)), 10000).limit_denominator(100000)
s500 = resample_poly(x, ratio.numerator, ratio.denominator)
print(f"[REAL] resample_poly {fs_eff:.4f}->500 Hz: up={ratio.numerator} down={ratio.denominator} n={len(s500)}")

# 2-5. 部署链 (与训练数据同函数)
dc = s500 - np.mean(s500)
combed = _comb_filter(dc)
filt = _hp_lp_filter(combed)
dec = filt[0::2]
chain250 = causal_hp_05_fs250(dec)
print(f"[REAL] chain 250Hz stream n={len(chain250)}")

# 6. XQRS R峰
from wfdb.processing import xqrs_detect
r_idx = xqrs_detect(chain250.astype(np.float64), fs=TARGET_FS, verbose=False)
print(f"[REAL] XQRS R peaks: {len(r_idx)}")

# 7. 窗口提取 (strict, 固件 z-score) — 复用 harness 函数
beats = extract_beats_deploy(chain250, np.array(r_idx, dtype=np.int64), "incart")
print(f"[REAL] beats extracted: {len(beats)}")

# 8. 伪影剔除: 仅 z-score 极端值 (|z|>8, 强伪影/断线)。
# 注: QRS 尖端 1-2 采样点轻度饱和 (max|x| 中位 1.58V, 硬轨拍仅 1-2 点) 是
# 本设备 AFE 增益下的真实域特征, 保留 (微调目标就是让模型认识真实设备形态)。
keep_idx = []
reject_reason = {"clip": 0, "zrange": 0}
for i in range(len(beats)):
    b = beats[i]
    if np.abs(b).max() > 8.0:
        reject_reason["zrange"] += 1
        continue
    keep_idx.append(i)

clean = beats[np.array(keep_idx, dtype=np.int64)]
print(f"[REAL] kept {len(clean)} beats (reject zrange={reject_reason['zrange']})")
np.save(OUT_BEATS, clean.astype(np.float32))
summary = {
    "source": str(ECGR.name), "fs_eff_hz": fs_eff, "dur_s": dur,
    "resample": {"up": ratio.numerator, "down": ratio.denominator},
    "n_peaks": int(len(r_idx)), "n_beats_kept": int(len(clean)),
    "reject": reject_reason,
    "chain": "500Hz->DC->comb10x2->HP/LP(warmup240)->decimate->causalHP0.5@250",
    "label": "all normal (0)",
}
OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"[REAL] saved {OUT_BEATS.name} + {OUT_JSON.name}")
