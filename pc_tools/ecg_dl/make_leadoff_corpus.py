#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_leadoff_corpus.py — 真实电极脱落等效语料生成 (Round-H, TH §114)
================================================================================
背景: 2026-09-14 用户真机拔线实测不报警。真实拔线后 AFE 输出非数学平线
(电极弹出瞬态/基线漂移/50Hz 工频拾取), QRS 检测器在伪迹上出"假拍"致 4s
无拍窗口 (ALARM_SRC_FLAT) 永远凑不齐 —— 合成平线段 (seg2 replay_flat) 覆盖
不了"有能量的坏信号"。本脚本生成 LEADOFF 检测器 (ALARM_SRC_LEADOFF 0x10)
的正/负回归语料。

生成纪律 (沿 M2 make_replay_corpus.py):
  - 载波: 解析 ecg_replay_data.h 的 MIT-100 (与既有语料同源, 零下载依赖);
  - 既有 31 case 语料 (ecg_replay_corpus.h) 逐字节不动, 本脚本写独立头
    ecg_replay_corpus2.h, 运行时由 ecg_replay.cpp 按段号路由;
  - int16 微伏量化, mV 数值域 ×1000;
  - RNG seed 固定全程可复现; PC 设备链复刻 (noise_lib.chain_shape) 预标定
    链后 RMS/crest, 作为固件阈值 (ALARM_LO_*) 的语料依据。

case 矩阵 (7 个, seg 编号 = 3 + 31 + 序号 = 34..40, 各 60s):
  c0 lo_mains_strong (正) 10s 正常 -> 50s 平线+50.3Hz 工频拾取 (raw rms 0.35mV)
  c1 lo_mains_weak   (正) 同上但 raw rms 0.10mV (链后 < RMS 地板, 走静默路)
  c2 lo_drift_float  (正) 10s 正常 -> 50s 平线+低频漂浮(0.15-0.5Hz, 0.30mV)
                          + 每 ~6s 运动突发 (0.5mV, 0.8s)
  c3 lo_pop_noise    (正) 10s 正常 -> 1.5s 电极弹出瞬态 (1.8mV 饱和+指数衰减)
                          -> 48.5s 噪声平线 (宽带 0.10mV + 弱工频 0.05mV
                          + 泊松尖峰 3/s, 诱发 QRS 假拍 —— 用户实测场景等效)
  c4 lo_neg_walk     (负) 60s MIT-100 + motion @ SNR 0dB (走动等效, 拍应存活)
  c5 lo_neg_vf       (负) 10s 正常 -> 50s 合成 VF (seg31 同配方, rms~0.5mV;
                          VF 不得误判脱落 — 红线 §2.2.1)
  c6 lo_mixed_cycle  (混合) 10 正常 / 15 工频拾取 / 10 正常 / 20 弹出+噪声 /
                          5 正常 —— 擎住/解除/再触发时序验证 (§3.4)

工频拾取含 +0.3Hz 频偏: 固件梳状滤波器 (10 抽头 MA @500Hz) 在 50.000Hz
有数学零点, 纯 50Hz 会被完全抑制; 真实电网频偏 ±0.1-0.5Hz 使抑制有限,
50.3Hz + 幅度调制 + 谐波 (100.6/150.9Hz) 是保守的真实等效。
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from noise_lib import synth_noise, chain_shape
from make_replay_corpus import parse_replay_header

REPO = BASE.parent.parent
HDR_DATA = (REPO / "experiments" / "esp_idf_ecg_migration" / "components" /
            "ecg_core" / "include" / "signal_generator" / "ecg_replay_data.h")
HDR_CORPUS1 = HDR_DATA.parent / "ecg_replay_corpus.h"
HDR_OUT = HDR_DATA.parent / "ecg_replay_corpus2.h"
CORPUS_DIR = BASE / "corpus"

FS = 500
DUR_S = 60
N = DUR_S * FS
N_CASES = 7
SEED = 20260914
SEG_BASE = 34          # 3 + 31 (既有语料) + index
LEADIN_S = 10


def md5_file(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def tile_carrier(carrier, n):
    """载波循环铺到 n 样本 (ecg_replay_normal 仅 45s, 60s case 需拼接)。"""
    reps = int(np.ceil(n / len(carrier)))
    return np.tile(carrier, reps)[:n].astype(np.float32)


def synth_mains_pickup(n, fs, rng, rms_target):
    """50.3Hz 工频拾取族: 基波(+0.3Hz 频偏) + 100.6/150.9Hz 谐波 + 慢 AM。
    输出定标到 raw 域 rms_target (mV)。"""
    t = np.arange(n) / fs
    ph = rng.uniform(0, 2 * np.pi)
    x = (np.sin(2 * np.pi * 50.3 * t + ph)
         + 0.3 * np.sin(2 * np.pi * 100.6 * t + ph * 2.1)
         + 0.1 * np.sin(2 * np.pi * 150.9 * t + ph * 0.7))
    am = 1.0 + 0.35 * np.sin(2 * np.pi * 0.23 * t + rng.uniform(0, 6.28))
    x = x * am
    x = x - np.mean(x)
    return (x / np.sqrt(np.mean(x * x)) * rms_target).astype(np.float32)


def synth_wander(n, fs, rng, rms_target):
    """低频漂浮: 0.15/0.30/0.50Hz 正弦族 + 慢游走, 定标 rms_target。"""
    t = np.arange(n) / fs
    x = np.zeros(n)
    for f, a in [(0.15, 1.0), (0.30, 0.6), (0.50, 0.3)]:
        x += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    walk = np.cumsum(rng.normal(0, 0.02, n))
    walk -= np.linspace(walk[0], walk[-1], n)
    x = x + walk
    x = x - np.mean(x)
    return (x / np.sqrt(np.mean(x * x)) * rms_target).astype(np.float32)


def synth_motion_bursts(n, fs, rng, period_s, dur_s, amp):
    """周期性运动突发: 平滑阶跃簇 (走动摆动等效), 非全局定标。"""
    y = np.zeros(n, dtype=np.float32)
    t0 = period_s / 2
    while t0 < n / fs - dur_s:
        i0, i1 = int(t0 * fs), int((t0 + dur_s) * fs)
        m = i1 - i0
        steps = np.zeros(m)
        cur = 0.0
        for i in range(m):
            if rng.random() < 0.5 / fs * 4:      # 突发内更高换目标率
                cur = rng.normal(0, 1)
            steps[i] = cur
        b, a = butter(2, 1.0, fs=fs)
        base = lfilter(b, a, steps)
        base = base / (np.max(np.abs(base)) + 1e-9)
        y[i0:i1] = (amp * base).astype(np.float32)
        t0 += period_s + rng.uniform(-1.0, 1.0)
    return y


def synth_pop_transient(n, fs, sat=1.8, tau=0.5, dur=1.5):
    """电极弹出瞬态: 饱和阶跃 + 指数衰减回基线。"""
    y = np.zeros(n, dtype=np.float32)
    i1 = min(int(dur * fs), n)
    t = np.arange(i1) / fs
    y[:i1] = sat * np.exp(-t / tau)
    return y


def synth_spiky_flatline(n, fs, rng, emg_rms, mains_rms, spike_rate,
                         spike_amp=(0.3, 0.8)):
    """噪声平线: 宽带 EMG + 弱工频 + 泊松尖峰 (诱发 QRS 假拍)。"""
    emg = synth_noise("emg", n, fs, rng)
    emg = emg / np.sqrt(np.mean(emg * emg)) * emg_rms
    mains = synth_mains_pickup(n, fs, rng, mains_rms)
    y = emg + mains
    n_spikes = int(spike_rate * n / fs)
    for _ in range(n_spikes):
        i0 = int(rng.integers(0, n - 30))
        w = int(rng.uniform(0.005, 0.015) * fs)
        a = rng.uniform(*spike_amp) * rng.choice([-1.0, 1.0])
        w = min(w, n - i0)
        t = np.arange(w) / fs
        y[i0:i0 + w] += a * np.sin(np.pi * t / (w / fs))   # 半正弦尖峰
    return y.astype(np.float32)


def synth_vf(n, fs, rng):
    """合成 VF (seg31 synthetic_vf_probe 同配方): 4.5-6.5Hz 频率游走正弦
    × 慢幅度调制, rms ~0.5mV。"""
    t = np.arange(n) / fs
    wander = np.sin(2 * np.pi * 5.5 * t + 6.0 * np.sin(2 * np.pi * 0.4 * t))
    amp = 0.5 * (1 + 0.4 * np.sin(2 * np.pi * 0.8 * t + 1.0))
    return (wander * amp).astype(np.float32)


def phase_stats(sig, lo_s, hi_s, fs=FS):
    """某相位窗经 PC 设备链复刻后的统计 (链后域 = 固件 LEADOFF 判据域)。"""
    seg = sig[int(lo_s * fs):int(hi_s * fs)]
    shaped = chain_shape(seg)
    x = shaped - np.mean(shaped)
    rms = float(np.sqrt(np.mean(x * x)))
    crest = float(np.max(np.abs(x)) / (rms + 1e-12))
    return {"rms_mV": round(rms, 4), "crest": round(crest, 2),
            "p2p_mV": round(float(np.ptp(shaped)), 3)}


def main():
    rng = np.random.default_rng(SEED)
    normal, _ = parse_replay_header(HDR_DATA)
    leadin = tile_carrier(normal, LEADIN_S * FS)
    flat_base = 0.2

    md5_c1_before = md5_file(HDR_CORPUS1)

    def leadin_plus(tail):
        return np.concatenate([leadin, np.asarray(tail, dtype=np.float32)])

    mains_s = synth_mains_pickup((DUR_S - LEADIN_S) * FS, FS, rng, 0.35)
    mains_w = synth_mains_pickup((DUR_S - LEADIN_S) * FS, FS, rng, 0.10)

    drift_tail = (synth_wander((DUR_S - LEADIN_S) * FS, FS, rng, 0.20)
                  + synth_motion_bursts((DUR_S - LEADIN_S) * FS, FS, rng,
                                        period_s=6.0, dur_s=0.8, amp=0.35))

    pop_tail = np.concatenate([
        synth_pop_transient(int(1.5 * FS), FS),
        synth_spiky_flatline((DUR_S - LEADIN_S) * FS - int(1.5 * FS), FS,
                             rng, emg_rms=0.25, mains_rms=0.05,
                             spike_rate=3.0),
    ])

    walk = tile_carrier(normal, N)
    walk = walk + synth_noise("motion", N, FS, rng)[:N] * np.sqrt(
        np.mean((walk - np.mean(walk)) ** 2))   # SNR 0dB: 噪声功率=信号功率

    vf_tail = synth_vf((DUR_S - LEADIN_S) * FS, FS, rng)

    mixed_tail = np.concatenate([
        synth_mains_pickup(15 * FS, FS, rng, 0.35),
        tile_carrier(normal, 10 * FS),
        np.concatenate([
            synth_pop_transient(int(1.5 * FS), FS),
            synth_spiky_flatline(20 * FS - int(1.5 * FS), FS, rng,
                                 emg_rms=0.10, mains_rms=0.05, spike_rate=3.0),
        ]),
        tile_carrier(normal, 5 * FS),
    ])

    cases = [
        ("lo_mains_strong", {"kind": "positive", "phases": "10s normal + 50s flat+mains50.3Hz raw-rms 0.35mV"},
         leadin_plus(flat_base + mains_s)),
        ("lo_mains_weak", {"kind": "positive", "phases": "10s normal + 50s flat+mains50.3Hz raw-rms 0.10mV"},
         leadin_plus(flat_base + mains_w)),
        ("lo_drift_float", {"kind": "positive", "phases": "10s normal + 50s flat+wander0.20mV + motion bursts 0.35mV/6s"},
         leadin_plus(flat_base + drift_tail)),
        ("lo_pop_noise", {"kind": "positive", "phases": "10s normal + 1.5s pop(1.8mV,tau0.5) + 48.5s spiky flatline(emg0.25+mains0.05+spikes3/s)"},
         leadin_plus(flat_base + pop_tail)),
        ("lo_neg_walk", {"kind": "negative", "phases": "60s mit100 + motion SNR 0dB (walk equivalent)"},
         walk),
        ("lo_neg_vf", {"kind": "negative", "phases": "10s normal + 50s synthetic VF (seg31 recipe)"},
         leadin_plus(vf_tail)),
        ("lo_mixed_cycle", {"kind": "mixed", "phases": "10 normal/15 mains/10 normal/20 pop+noise/5 normal"},
         np.concatenate([leadin, mixed_tail])[:N]),
    ]

    assert len(cases) == N_CASES
    for name, _, sig in cases:
        assert sig.shape == (N,), (name, sig.shape)
        assert np.all(np.isfinite(sig)), name

    arr = np.stack([c[2] for c in cases]).astype(np.float32)
    q = np.clip(np.round(arr * 1000.0), -32768, 32767).astype(np.int16)
    HDR_OUT.write_text(gen_header(cases, q), encoding="utf-8")

    np.save(CORPUS_DIR / "leadoff_corpus_500hz.npy", arr)

    # ---- PC 设备链预标定: 各 case 关键相位链后 rms/crest ----
    stats = {}
    for name, params, sig in cases:
        phases = {"full": phase_stats(sig, 0, DUR_S)}
        if params["kind"] in ("positive", "negative", "mixed"):
            phases["leadin_0_10s"] = phase_stats(sig, 0, 10)
            phases["tail_10_60s"] = phase_stats(sig, 10, DUR_S)
        stats[name] = phases

    prov = {
        "generator": "make_leadoff_corpus.py",
        "seed": SEED,
        "fs": FS,
        "dur_s": DUR_S,
        "n_cases": N_CASES,
        "segment_base": SEG_BASE,
        "units": "header int16 microvolt (x0.001 -> mV numeric at runtime)",
        "purpose": "LEADOFF (ALARM_SRC_LEADOFF 0x10) regression corpus, TH §114",
        "mains_offset_note": ("工频含 +0.3Hz 频偏: 固件梳状滤波器在 50.000Hz "
                              "有数学零点, 纯 50Hz 会被完全抑制, 真实电网频偏 "
                              "使抑制有限 (保守等效)"),
        "corpus1_md5_before": md5_c1_before,
        "corpus1_md5_after": md5_file(HDR_CORPUS1),
        "chain_precalibration": stats,
        "cases": [],
    }
    for i, (name, params, sig) in enumerate(cases):
        prov["cases"].append({
            "index": i, "segment": SEG_BASE + i, "name": name,
            "params": params,
            "md5_int16uv": hashlib.md5(q[i].tobytes()).hexdigest(),
            "p2p_mV": round(float(np.ptp(sig)), 4),
        })
    (CORPUS_DIR / "leadoff_corpus_provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")

    assert prov["corpus1_md5_before"] == prov["corpus1_md5_after"], \
        "corpus1 header modified! 既有语料必须逐字节不变"

    print(f"[leadoff-corpus] {N_CASES} cases x {DUR_S}s -> {HDR_OUT.name}")
    print(f"[leadoff-corpus] header {HDR_OUT.stat().st_size/1024:.0f}KB, "
          f"flash payload {q.nbytes/1024:.0f}KB")
    print(f"[leadoff-corpus] corpus1 md5 unchanged: {md5_c1_before[:12]}...")
    for name, ph in stats.items():
        tail = ph.get("tail_10_60s") or ph["full"]
        print(f"  {name:18s} tail: rms={tail['rms_mV']:.4f} mV "
              f"crest={tail['crest']:.2f} p2p={tail['p2p_mV']:.2f}")


def gen_header(cases, q):
    names = [c[0] for c in cases]
    lines = []
    lines.append("/* Auto-generated by pc_tools/ecg_dl/make_leadoff_corpus.py (Round-H, TH §114)")
    lines.append(" * 真实电极脱落等效语料: %d case x %d 样本 (60s @500Hz), int16 微伏。" % (len(cases), q.shape[1]))
    lines.append(" * 正例(mains strong/weak, drift, pop+noise) 负例(walk, vf) 混合(mixed_cycle)。")
    lines.append(" * 逐 case 来源/参数/MD5/链后预标定: pc_tools/ecg_dl/corpus/leadoff_corpus_provenance.json")
    lines.append(" * 仅 N16R8 (16MB flash) 编入; 段号 = 3 + 31 + index (ecg_replay.cpp 路由)。 */")
    lines.append("#ifndef ECG_REPLAY_CORPUS2_H")
    lines.append("#define ECG_REPLAY_CORPUS2_H")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("#define ECG_REPLAY_CORPUS2_N %d" % len(cases))
    lines.append("#define ECG_REPLAY_CORPUS2_LEN %d" % q.shape[1])
    lines.append("")
    lines.append("static const char *const ecg_corpus2_names[ECG_REPLAY_CORPUS2_N] = {")
    for n in names:
        lines.append('    "%s",' % n)
    lines.append("};")
    lines.append("")
    lines.append("/* int16 微伏语料数组 [case][样本] */")
    lines.append("static const int16_t ecg_corpus2[ECG_REPLAY_CORPUS2_N][ECG_REPLAY_CORPUS2_LEN] = {")
    for i in range(len(cases)):
        row = q[i]
        lines.append("    { /* %d (seg%d): %s */" % (i, SEG_BASE + i, names[i]))
        for j in range(0, len(row), 16):
            lines.append("        " + ", ".join(str(int(v)) for v in row[j:j + 16]) + ",")
        lines.append("    },")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* ECG_REPLAY_CORPUS2_H */")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
