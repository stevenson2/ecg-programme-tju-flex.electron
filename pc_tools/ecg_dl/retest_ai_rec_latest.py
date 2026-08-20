#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retest_ai_rec_latest.py — esp_timer 后 AI 输入链离线重测 (rec_latest.ecgr)
========================================================================
复刻固件 AI 部署链:
  248Hz 录音原始 → 有理数重采样 500Hz → DC 去均值 → 双级 10 抽头梳状
  → HP(0.05Hz? 按 preprocess_real_exp7c 的 _hp_lp_filter 即固件 AI 链 HP0.05+LP40)
  → 2:1 抽取 → 因果 HP0.5Hz@250Hz → 固件滑动窗口 (AI_STRIDE=250, OFFSET=6)
  → 窗内 Z-score → INT8 量化 → exp7c INT8 TFLite → 直接取异常类概率 (固件语义)

输出: models/deploy_match/ai_rec_latest_int8.json
"""
import json
import struct
import sys
from pathlib import Path
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_deploy_match import _comb_filter, _hp_lp_filter
from data.preprocess import causal_hp_05_fs250

ROOT = Path(__file__).resolve().parents[2]
ECGR = ROOT / 'rec_latest.ecgr'
TFL = Path(__file__).resolve().parent / 'models' / 'ecg_model_exp7c_int8.tflite'
OUT = Path(__file__).resolve().parent / 'models' / 'deploy_match' / 'ai_rec_latest_int8.json'
THRESH = 0.60
CONFIRM = 5
WINDOW = 250
STRIDE = 250
OFFSET = 6


def load_chain():
    raw = ECGR.read_bytes()
    n = struct.unpack_from('<I', raw, 18)[0]
    dur = struct.unpack_from('<I', raw, 14)[0]
    x = np.frombuffer(raw, dtype='<i2', count=n, offset=32).astype(np.float64) / 8000.0
    fs_eff = n / dur
    ratio = Fraction(int(round(500.0 / fs_eff * 10000)), 10000).limit_denominator(100000)
    s500 = resample_poly(x, ratio.numerator, ratio.denominator)
    dc = s500 - np.mean(s500)
    combed = _comb_filter(dc)
    filt = _hp_lp_filter(combed)
    dec = filt[0::2]
    chain250 = causal_hp_05_fs250(dec)
    return x, fs_eff, dur, chain250


def run_tflite_windows(chain250):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(TFL))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp['quantization']
    out_scale, out_zp = out['quantization']

    starts = []
    probs = []
    for n in range(WINDOW, len(chain250)):
        if n % STRIDE == OFFSET:
            w = chain250[n - WINDOW + 1:n + 1]
            if len(w) != WINDOW:
                break
            # 固件 Z-score (std<1e-6 时置 1)
            mean = w.mean()
            std = w.std()
            if std < 1e-6:
                std = 1.0
            xf = (w - mean) / std
            # 固件 INT8 量化
            xq = np.clip(np.round(xf.astype(np.float32) / in_scale) + in_zp,
                         -128, 127).astype(np.int8)
            interp.set_tensor(inp['index'], xq[None, :, None])
            interp.invoke()
            q = interp.get_tensor(out['index'])[0]
            # 固件 parse_output_confidence: 反量化后直接取异常类
            p = float((q[1].astype(np.float32) - out_zp) * out_scale)
            probs.append(max(0.0, min(1.0, p)))
            starts.append(n)
    return np.array(starts), np.array(probs)


def main():
    x, fs_eff, dur, chain250 = load_chain()
    starts, probs = run_tflite_windows(chain250)
    abnormal = probs > THRESH
    # 连续异常最长段 (多拍确认)
    longest = 0
    cur = 0
    alarm_blocks = 0
    for a in abnormal:
        if a:
            cur += 1
            longest = max(longest, cur)
            if cur == CONFIRM:
                alarm_blocks += 1
        else:
            cur = 0
    summary = {
        "source": str(ECGR.name),
        "chain": "248Hz→500Hz resample→DC→comb10x2→HP/LP→2:1→causalHP0.5@250→"
                 f"slide W={WINDOW} S={STRIDE} O={OFFSET}",
        "fs_eff_hz": round(fs_eff, 4),
        "dur_s": int(dur),
        "raw_n": int(len(x)),
        "chain250_n": int(len(chain250)),
        "windows": int(len(probs)),
        "threshold": THRESH,
        "confirm": CONFIRM,
        "conf": {
            "mean": round(float(probs.mean()), 5),
            "median": round(float(np.median(probs)), 5),
            "p90": round(float(np.percentile(probs, 90)), 5),
            "p99": round(float(np.percentile(probs, 99)), 5),
            "max": round(float(probs.max()), 5),
        },
        "alarm": {
            "frac_gt_threshold": round(float(np.mean(abnormal)), 5),
            "longest_consecutive_abnormal": int(longest),
            "alarm_blocks_after_confirm": int(alarm_blocks),
        },
        "note": "无人工金标准; 用户确认该段为静息真实 AFE 验证, 归档时视为正常段。"
                "直接概率为固件语义 (反量化后取异常类, 无二次 softmax)。",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('saved:', OUT)


if __name__ == '__main__':
    main()
