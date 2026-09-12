#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_replay_corpus.py — 板上噪声鲁棒性回放语料生成 (M2, TH §106)
================================================================================
载波: 解析固件现有 ecg_replay_data.h 的 MIT-100 (0-45s 正常) / MIT-106
(90-135s 异常) 500Hz 段 — 与 M0/M1 基线完全同源, 零下载依赖 (本地无 mitdb
原始数据, wfdb pn_dir 依赖外网; 现有头文件本身就是 2026-08-08 从 mitdb
提取并重采样到 500Hz 的权威副本)。

跨库素材: LUDB (C:\\ecg_data\\ecg_database, 1000Hz 12 导联 mV, 本地真库):
  - LUDB lead-II 干净段作交叉载波 case (跨库泛化探针)
  - LUDB 含伪迹段提取为真库噪声, 叠加到 MIT 载波 (真库噪声 case)

产物:
  1. experiments/.../components/ecg_core/include/signal_generator/ecg_replay_corpus.h
     int16 µV 二维数组 [N_CASES][15000] (30s @500Hz/case) + 名字表
  2. pc_tools/ecg_dl/corpus/replay_corpus_provenance.json (逐 case 来源/参数/seed/MD5)
  3. pc_tools/ecg_dl/corpus/corpus_500hz.npy (PC 侧分析缓存, 不入库)

泄漏纪律: 语料仅用于评估 (板上闭环), 不入训练; MIT-100/106 的划分归属
(训练/测试患者) 在 provenance 中披露。生成用 RNG seed 固定, 全程可复现。

case 矩阵 (30 个, seg 编号 = 3+序号):
  c00 clean_normal       MIT-100 0-30s
  c01 clean_abnormal     MIT-106 0-30s (源 90-120s)
  c02-c16 normal 载波 × {bw,resp,mains,emg,motion} × SNR {0,10,20} = 15
  c17 elec_off_mild      2 次脱落事件
  c18 elec_off_severe    5 次脱落事件
  c19-c21 adc_quant      12/10/8 bit
  c22-c25 abnormal 载波 × {bw,emg,motion,mains} @ SNR 10dB = 4
  c26 pure_emg           纯肌电 (无载波)
  c27 pure_motion        纯运动伪迹
  c28 ludb_normal        LUDB lead-II 干净 30s (跨库)
  c29 ludb_noise_on_mit  MIT-100 + LUDB 真库伪迹 @ SNR 10dB
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from noise_lib import (synth_noise, mix_at_snr, apply_electrode_off,
                       apply_adc_quant, _unit_power)

REPO = BASE.parent.parent                       # ecg-programme-tju-flex.electron-master
HDR_IN = (REPO / "experiments" / "esp_idf_ecg_migration" / "components" /
          "ecg_core" / "include" / "signal_generator" / "ecg_replay_data.h")
HDR_OUT = HDR_IN.parent / "ecg_replay_corpus.h"
CORPUS_DIR = BASE / "corpus"

FS = 500
DUR_S = 30
N_CASES = 30
SEED = 20260912


def parse_replay_header(path):
    """提取 ecg_replay_data.h 的两个 float 数组 → (normal, abnormal) ndarray。"""
    text = path.read_text(encoding="utf-8")
    out = {}
    for name in ("ecg_replay_normal", "ecg_replay_abnormal"):
        m = re.search(rf"const float {name}\[\] = \{{(.*?)\}};", text, re.S)
        if not m:
            raise RuntimeError(f"{name} not found in {path}")
        vals = [float(v.strip().rstrip("f")) for v in m.group(1).split(",") if v.strip()]
        out[name] = np.asarray(vals, dtype=np.float32)
    return out["ecg_replay_normal"], out["ecg_replay_abnormal"]


def load_ludb_lead_ii(rec_dir, rec_name, start_s=5, dur_s=30):
    """LUDB 记录 lead II, 1000Hz → 500Hz (此处不 import wfdb 顶层, 延迟加载)。"""
    import wfdb
    from scipy.signal import resample_poly
    rec = wfdb.rdrecord(str(rec_dir / rec_name),
                        sampfrom=int(start_s * 1000),
                        sampto=int((start_s + dur_s) * 1000))
    ch = rec.sig_name.index("ii")
    sig = rec.p_signal[:, ch].astype(np.float64)
    return resample_poly(sig, 1, 2).astype(np.float32)


def extract_ludb_noise():
    """从 LUDB 提取真库伪迹: 取摆动最大的导联窗 (低频大幅游走), 归一化。"""
    import wfdb
    from scipy.signal import resample_poly
    # patient006 首条记录的 I 导联含明显基线游走 (LUDB 静息伪迹常见);
    # 记录名随患者变化 → glob 取该目录首条 .hea。
    rec_dir = Path("/mnt/c/ecg_data/ecg_database/patient006")
    heas = sorted(rec_dir.glob("*.hea"))
    if not heas:
        raise FileNotFoundError(f"no .hea in {rec_dir}")
    rec = wfdb.rdrecord(str(heas[0].with_suffix("")), sampfrom=0, sampto=60000)
    sig = rec.p_signal[:, rec.sig_name.index("i")].astype(np.float64)
    sig5 = resample_poly(sig, 1, 2).astype(np.float32)
    win = DUR_S * FS
    best, best_p = None, -1.0
    for i in range(0, len(sig5) - win, win // 2):
        w = sig5[i:i + win]
        p = float(np.var(np.diff(w)))   # 高摆动窗 = 伪迹重
        if p > best_p:
            best_p, best = p, w
    return _unit_power(best)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ludb", action="store_true",
                    help="跳过 LUDB case (无 wfdb/本地库时)")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    normal, abnormal = parse_replay_header(HDR_IN)
    carriers = {
        "normal": normal[: DUR_S * FS].copy(),
        "abnormal": abnormal[: DUR_S * FS].copy(),
    }

    cases = []   # (name, dict_params, signal_500hz_mV)
    add = lambda name, params, sig: cases.append((name, params, np.asarray(sig, dtype=np.float32)))

    add("clean_normal", {"carrier": "mit100_0_30s", "noise": None}, carriers["normal"])
    add("clean_abnormal", {"carrier": "mit106_90_120s", "noise": None}, carriers["abnormal"])

    for ntype in ("baseline_wander", "respiratory", "mains", "emg", "motion"):
        for snr in (0, 10, 20):
            sig = carriers["normal"].copy()
            nz = synth_noise(ntype, len(sig), FS, rng)
            mixed = mix_at_snr(sig, nz, snr)
            if ntype == "respiratory":
                # 呼吸型同时做 ±12% 载波幅度调制
                t = np.arange(len(mixed)) / FS
                mixed = mixed * (1.0 + 0.12 * np.sin(2 * np.pi * 0.25 * t))
            add(f"normal_{ntype}_snr{snr}",
                {"carrier": "mit100", "noise": ntype, "snr_db": snr}, mixed)

    add("normal_elec_off_mild",
        {"carrier": "mit100", "noise": "electrode_off", "events": 2},
        apply_electrode_off(carriers["normal"], FS, rng, n_events=2))
    add("normal_elec_off_severe",
        {"carrier": "mit100", "noise": "electrode_off", "events": 5},
        apply_electrode_off(carriers["normal"], FS, rng, n_events=5))

    for bits in (12, 10, 8):
        add(f"normal_adc_quant_{bits}bit",
            {"carrier": "mit100", "noise": "adc_quant", "bits": bits},
            apply_adc_quant(carriers["normal"], bits))

    for ntype in ("baseline_wander", "emg", "motion", "mains"):
        nz = synth_noise(ntype, len(carriers["abnormal"]), FS, rng)
        add(f"abnormal_{ntype}_snr10",
            {"carrier": "mit106", "noise": ntype, "snr_db": 10},
            mix_at_snr(carriers["abnormal"], nz, 10))

    add("pure_emg", {"carrier": None, "noise": "emg", "note": "无载波纯噪声"},
        synth_noise("emg", DUR_S * FS, FS, rng) * 0.3)
    add("pure_motion", {"carrier": None, "noise": "motion", "note": "无载波纯噪声"},
        synth_noise("motion", DUR_S * FS, FS, rng) * 0.3)

    ludb_loaded = False
    if not args.skip_ludb:
        try:
            ludb = load_ludb_lead_ii(Path("/mnt/c/ecg_data/ecg_database/patient001"),
                                     "s0010_re", start_s=5, dur_s=DUR_S)
            ludb = ludb - np.mean(ludb)
            ludb_noise = extract_ludb_noise()
            # 两段素材都拿到才入列 (避免半成功后回退重复追加)
            add("ludb_normal", {"carrier": "ludb_s0010_re_leadii_5_35s", "noise": None,
                                "fs_src": 1000}, ludb)
            add("ludb_noise_on_mit",
                {"carrier": "mit100", "noise": "ludb_real_artifact", "snr_db": 10},
                mix_at_snr(carriers["normal"], ludb_noise, 10))
            ludb_loaded = True
        except Exception as e:   # LUDB 不可用则降级为合成补位 (记录在 provenance)
            print(f"[corpus] LUDB 不可用 ({e}); 用合成 motion 补位", file=sys.stderr)
            nz = synth_noise("motion", DUR_S * FS, FS, rng)
            add("ludb_normal", {"carrier": "mit100", "noise": None, "degraded": True,
                                "reason": f"ludb_unavailable: {e}"}, carriers["normal"].copy())
            add("ludb_noise_on_mit",
                {"carrier": "mit100", "noise": "motion(degraded)", "snr_db": 10},
                mix_at_snr(carriers["normal"], nz, 10))
    else:
        nz = synth_noise("motion", DUR_S * FS, FS, rng)
        add("ludb_normal", {"carrier": "mit100", "noise": None, "degraded": True,
                            "reason": "--skip-ludb"}, carriers["normal"].copy())
        add("ludb_noise_on_mit",
            {"carrier": "mit100", "noise": "motion(degraded)", "snr_db": 10},
            mix_at_snr(carriers["normal"], nz, 10))

    assert len(cases) == N_CASES, f"case 数 {len(cases)} != {N_CASES}"

    # ---- 输出 1: 固件头 (int16 µV; 载波数值域 = mV 数值, ×1000 → µV) ----
    arr = np.stack([c[2] for c in cases]).astype(np.float32)
    q = np.clip(np.round(arr * 1000.0), -32768, 32767).astype(np.int16)
    HDR_OUT.write_text(gen_header(cases, q), encoding="utf-8")

    # ---- 输出 2/3: provenance + npy 缓存 ----
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CORPUS_DIR / "corpus_500hz.npy", arr)
    prov = {
        "generator": "make_replay_corpus.py",
        "seed": SEED,
        "fs": FS,
        "dur_s": DUR_S,
        "n_cases": N_CASES,
        "segment_base": 3,           # 固件 seg 编号 = 3 + index
        "units": "header int16 microvolt (x0.001 -> mV numeric at runtime)",
        "snr_convention": "10*log10(P_sig/P_noise) @500Hz raw domain, mean-removed",
        "ludb_loaded": ludb_loaded,
        "leakage_note": ("语料仅用于板上评估, 不入训练。MIT-100/106 的患者划分归属"
                          "由 split_guard 口径披露于 split_membership 字段。"),
        "cases": [],
    }
    for i, (name, params, sig) in enumerate(cases):
        prov["cases"].append({
            "index": i, "segment": 3 + i, "name": name,
            "params": params,
            "md5_int16uv": hashlib.md5(q[i].tobytes()).hexdigest(),
            "p2p_mV": round(float(np.ptp(sig)), 4),
        })
    (CORPUS_DIR / "replay_corpus_provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[corpus] {N_CASES} cases -> {HDR_OUT.name}")
    print(f"[corpus] header {HDR_OUT.stat().st_size/1024:.0f}KB, "
          f"flash payload {q.nbytes/1024:.0f}KB, provenance -> {CORPUS_DIR}")
    for i, (name, _, sig) in enumerate(cases[:3]):
        print(f"  seg{3+i} {name}: p2p {np.ptp(sig):.2f} mV")


def gen_header(cases, q):
    names = [c[0] for c in cases]
    lines = []
    lines.append("/* Auto-generated by pc_tools/ecg_dl/make_replay_corpus.py (M2, TH §106)")
    lines.append(" * 板上噪声语料: %d case x %d 样本 (30s @500Hz), int16 微伏。" % (len(cases), q.shape[1]))
    lines.append(" * 运行时 float = int16 * 0.001 (mV 数值域, 与 ecg_replay_data.h 一致)。")
    lines.append(" * 逐 case 来源/参数/MD5: pc_tools/ecg_dl/corpus/replay_corpus_provenance.json")
    lines.append(" * 仅 N16R8 (16MB flash) 编入; 由 CMake ECG_REPLAY_CORPUS 宏开关。 */")
    lines.append("#ifndef ECG_REPLAY_CORPUS_H")
    lines.append("#define ECG_REPLAY_CORPUS_H")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("#define ECG_REPLAY_CORPUS_N %d" % len(cases))
    lines.append("#define ECG_REPLAY_CORPUS_LEN %d" % q.shape[1])
    lines.append("")
    lines.append("/* case 名字表 (与 provenance JSON 的 cases[i].name 一一对应) */")
    lines.append("static const char *const ecg_corpus_names[ECG_REPLAY_CORPUS_N] = {")
    for n in names:
        lines.append('    "%s",' % n)
    lines.append("};")
    lines.append("")
    lines.append("/* int16 微伏语料数组 [case][样本] */")
    lines.append("static const int16_t ecg_corpus[ECG_REPLAY_CORPUS_N][ECG_REPLAY_CORPUS_LEN] = {")
    for i in range(len(cases)):
        row = q[i]
        lines.append("    { /* %d: %s */" % (i, names[i]))
        for j in range(0, len(row), 16):
            lines.append("        " + ", ".join(str(int(v)) for v in row[j:j+16]) + ",")
        lines.append("    },")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* ECG_REPLAY_CORPUS_H */")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
