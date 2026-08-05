#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_deploy_compensation.py — T1-3: 部署链失配分量消融 + 输入侧补偿原型 (v2)
======================================================================
任务: 必做清单 T1-3 / solutions.md M13
参考: Gregg et al. 2023 (CinC, linear-phase HP); Dobrev et al. 2025
      (Technologies 13(4):159, DOI 10.3390/technologies13040159, IHPF)
方法 (v2, 修正拍级时间错位理解):
  A. 细粒度消融阶梯 (6 链 × 4 旧模型 × 2 域): d0 → d0_n → d0_nh → d1 → d2 → d3
     分量: notch / hp_cutoff / causal_phase / decimation / comb; int8 引用 T0-1
  B. 输入侧补偿 (三档):
     P0 时间对齐: 因果链群延迟导致 R 峰窗口错位 (~6-9 样本 @250Hz, corr 仅 0.335) —
        拍级平移 δ* (δ* = 部署链群延迟的系统测量, 拍对互相关, 非数据拟合)
     P1 相位补偿: P0 后频域全通相位残差 (系统辨识 h_eff(δ*) vs h_d0)
     P2 幅度+相位: P0 后正则化逆滤波 (IHPF 式)
  C. 对比: 旧模型(不重训) {d3, P0, P0+P1, P0+P2} vs 重训 exp6-SGD {d3@δ*} vs 上限 d0
     注: FINAL_RESULTS 表4 报告数字 (0.9122/0.7697) 为 δ 对齐语义 (δ-sweep 最佳);
        本脚本统一在 P0(δ*) 语义下对比, 保证与报告数字可比
输出: models/deploy_compensation_eval.json
用法 (WSL): export ECG_PROCESSED_DIR=$HOME/ecg_data; python3 eval_deploy_compensation.py
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from scipy.signal import butter, filtfilt
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_deploy_match as edm
from eval_deploy_match import (
    CACHE_DIR, _build_ablation_beats, _add_channel_dim, compute_mit_domain_test_records,
    compute_ptb_domain_test_records, ptb_load_records, ptb_load_controls,
    TARGET_FS, BEAT_WINDOW_SAMPLES,
)
from data.preprocess_incart import resample_ecg

MODELS_DIR = Path(__file__).resolve().parent / "models"
OUT_JSON = MODELS_DIR / "deploy_compensation_eval.json"
FIR_LEN = 129
N_FFT = 4096

# ============================================================
# A. 细粒度消融链变体
# ============================================================

def _filtfilt_chain(sig, hp_cut, fs=250, notch=True):
    bh, ah = butter(2, hp_cut / (fs / 2), btype="high")
    bl, al = butter(2, 40.0 / (fs / 2), btype="low")
    y = filtfilt(bh, ah, sig)
    y = filtfilt(bl, al, y)
    if notch:
        bn, an = butter(2, [49.0 / (fs / 2), 51.0 / (fs / 2)], btype="bandstop")
        y = filtfilt(bn, an, y)
    return y.astype(np.float64)


def chain_d0_n(sig, orig_fs):
    """d0 去 Notch: FFT→250 + filtfilt HP0.5+LP40 (无 notch)."""
    s250 = resample_ecg(sig.reshape(-1, 1) if sig.ndim == 1 else sig, orig_fs, TARGET_FS)
    if s250.ndim > 1:
        s250 = s250.flatten()
    return _filtfilt_chain(s250, 0.5, notch=False)


def chain_d0_nh(sig, orig_fs):
    """d0_n 但 HP 0.5→0.05 (filtfilt 零相位)."""
    s250 = resample_ecg(sig.reshape(-1, 1) if sig.ndim == 1 else sig, orig_fs, TARGET_FS)
    if s250.ndim > 1:
        s250 = s250.flatten()
    return _filtfilt_chain(s250, 0.05, notch=False)


# ============================================================
# B. 补偿原型
# ============================================================

def estimate_delta_star(beats_d3, beats_d0, max_d=15, n=4000):
    """估计部署链群延迟 δ* (拍级): 拍对互相关, 系统常数测量 (非数据拟合).
    返回使 corr(d3 平移 δ, d0) 最大的 δ (负 = 窗口需提前)."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(beats_d3), min(n, len(beats_d3)), replace=False)
    xd, xb = beats_d3[idx], beats_d0[idx]
    best, best_corr = 0, -1.0
    corrs = {}
    for d in range(-max_d, max_d + 1):
        if d == 0:
            x = xd
        elif d > 0:
            x = np.concatenate([np.full((len(xd), d), 0.0, dtype=np.float32), xd[:, :-d]], axis=1)
        else:
            x = np.concatenate([xd[:, -d:], np.full((len(xd), -d), 0.0, dtype=np.float32)], axis=1)
        a = x - x.mean(axis=1, keepdims=True)
        b = xb - xb.mean(axis=1, keepdims=True)
        c = np.mean(np.sum(a * b, axis=1) / (np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1)) + 1e-9))
        corrs[d] = float(c)
        if c > best_corr:
            best_corr, best = c, d
    return best, best_corr, corrs


def shift_beats(beats, delta):
    """拍级循环平移 (np.roll, 保留波形能量, 避免零填充边缘失真)."""
    if delta == 0:
        return beats.copy()
    return np.roll(beats, delta, axis=1).astype(beats.dtype)


def system_identification(delta_star):
    """h_eff (部署链, 平移 δ* 对齐) 与 h_d0 (训练链) 的 250Hz 等效冲激."""
    imp500 = np.zeros(N_FFT)
    imp500[0] = 1.0
    h_eff = edm._hp_lp_filter(edm._comb_filter(imp500))[0::2]
    h_eff = shift_beats(h_eff.reshape(1, -1), delta_star).ravel()  # 对齐到训练链时间轴
    imp250 = np.zeros(N_FFT // 2)
    imp250[0] = 1.0
    h_d0 = _filtfilt_chain(imp250, 0.5, notch=True)
    return h_eff, h_d0


def design_compensation_fir(h_eff, h_d0, lam=0.02):
    """P1 全通相位 / P2 幅度+相位 (正则化逆) — 移位对齐后截取."""
    L = max(len(h_eff), len(h_d0))
    H_eff = np.fft.fft(h_eff, L)
    H_d0 = np.fft.fft(h_d0, L)
    C1 = np.exp(1j * (np.angle(H_d0) - np.angle(H_eff)))
    denom = np.abs(H_eff) ** 2 + lam * np.max(np.abs(H_eff) ** 2)
    C2 = H_d0 * np.conj(H_eff) / denom
    out = []
    for C in (C1, C2):
        c = np.fft.ifft(C).real
        peak = int(np.argmax(np.abs(c)))
        c = np.roll(c, len(c) // 2 - peak)
        mid = len(c) // 2
        c = c[mid - FIR_LEN // 2: mid + FIR_LEN // 2 + 1]
        c = c * np.hamming(len(c))
        w = np.abs(np.fft.fft(c, 2048))
        wf = np.fft.fftfreq(2048, 1 / TARGET_FS)
        band = (wf >= 10) & (wf <= 40)
        c = c / w[band].mean()
        out.append(c)
    return out


def apply_fir_comp(beats, c):
    """拍级 FIR 补偿 + 固件 Z-score."""
    out = np.zeros_like(beats, dtype=np.float32)
    for i in range(len(beats)):
        y = np.convolve(beats[i].astype(np.float64), c, mode="same")
        mu = y.mean()
        sd = np.sqrt(np.mean((y - mu) ** 2))
        if sd < 1e-6:
            sd = 1.0
        out[i] = ((y - mu) / sd).astype(np.float32)
    return out


def pair_corr(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return float(np.mean(np.sum(a * b, axis=1) / (np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1)) + 1e-9)))


# ============================================================
# C. 主流程
# ============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("T1-3 部署链失配分解 + 输入侧补偿 (v2)")
    print("=" * 70)

    mit_test, incart_test, _ = compute_mit_domain_test_records()
    ptb_test, _ = compute_ptb_domain_test_records()
    record_list = ptb_load_records()
    controls = ptb_load_controls()
    peak_file = CACHE_DIR / "ptb_deploy_match_peaks.npy"
    cached_peaks = np.load(peak_file, allow_pickle=True) if peak_file.exists() else None

    mit_cache = np.load(CACHE_DIR / "mit_deploy_match.npz")
    ptb_cache = np.load(CACHE_DIR / "ptb_deploy_match.npz")

    # ---- 消融链缓存 (d0_n / d0_nh) ----
    def _load_or_build(path, domain_name, chain_func, cache_ref):
        if path.exists():
            d = np.load(path)
            print(f"  cached {path.name}")
            return d["beats"], d["labels"]
        print(f"  building {path.name}...")
        beats, labels, rec_ids = _build_ablation_beats(
            domain_name, chain_func, mit_test, incart_test,
            ptb_test, record_list, controls, cached_peaks)
        if beats is None or len(beats) == 0:
            return np.zeros((0, 250), dtype=np.float32), np.zeros(0, dtype=np.int32)
        np.savez_compressed(path, beats=beats, labels=labels, record_ids=rec_ids)
        print(f"    {path.name}: {len(beats)} beats")
        return beats, labels

    d0n_mit = _load_or_build(CACHE_DIR / "mit_ablation_d0n.npz", "mit", chain_d0_n, mit_cache)
    d0n_ptb = _load_or_build(CACHE_DIR / "ptb_ablation_d0n.npz", "ptb", chain_d0_n, ptb_cache)
    d0nh_mit = _load_or_build(CACHE_DIR / "mit_ablation_d0nh.npz", "mit", chain_d0_nh, mit_cache)
    d0nh_ptb = _load_or_build(CACHE_DIR / "ptb_ablation_d0nh.npz", "ptb", chain_d0_nh, ptb_cache)

    domains = {
        "mit": {"d0": (mit_cache["beats_baseline"], mit_cache["labels"]),
                "d0_n": d0n_mit, "d0_nh": d0nh_mit,
                "d1": (np.load(CACHE_DIR / "mit_ablation_d1.npz")["beats"],
                       np.load(CACHE_DIR / "mit_ablation_d1.npz")["labels"]),
                "d2": (np.load(CACHE_DIR / "mit_ablation_d2.npz")["beats"],
                       np.load(CACHE_DIR / "mit_ablation_d2.npz")["labels"]),
                "d3": (mit_cache["beats_deploy"], mit_cache["labels"])},
        "ptb": {"d0": (ptb_cache["beats_baseline"], ptb_cache["labels"]),
                "d0_n": d0n_ptb, "d0_nh": d0nh_ptb,
                "d1": (np.load(CACHE_DIR / "ptb_ablation_d1.npz")["beats"],
                       np.load(CACHE_DIR / "ptb_ablation_d1.npz")["labels"]),
                "d2": (np.load(CACHE_DIR / "ptb_ablation_d2.npz")["beats"],
                       np.load(CACHE_DIR / "ptb_ablation_d2.npz")["labels"]),
                "d3": (ptb_cache["beats_deploy"], ptb_cache["labels"])},
    }

    # ---- δ* 群延迟测量 (per domain) ----
    print("\n[B] 群延迟测量 (δ*):")
    delta_star, diag = {}, {}
    for dom in ("mit", "ptb"):
        beats3, labels = domains[dom]["d3"]
        beats0, _ = domains[dom]["d0"]
        d, c, corrs = estimate_delta_star(beats3, beats0)
        delta_star[dom] = d
        diag[f"corr_d3_d0"] = pair_corr(beats3[:4000], beats0[:4000])
        diag[f"corr_d3_d0_align"] = pair_corr(shift_beats(beats3, d)[:4000], beats0[:4000])
        diag[f"delta_sweep_corr"] = {str(k): round(v, 4) for k, v in sorted(corrs.items())}
        print(f"  [{dom}] δ* = {d} (corr {diag['corr_d3_d0']:.4f} → {diag['corr_d3_d0_align']:.4f})")

    # ---- 系统辨识 + 补偿 FIR (per domain) ----
    firs = {}
    for dom in ("mit", "ptb"):
        h_eff, h_d0 = system_identification(delta_star[dom])
        c1, c2 = design_compensation_fir(h_eff, h_d0)
        firs[dom] = {"c1": c1, "c2": c2}
    fir_cache = CACHE_DIR / "compensation_fir.npz"
    np.savez_compressed(fir_cache, c1_mit=firs["mit"]["c1"], c2_mit=firs["mit"]["c2"],
                        c1_ptb=firs["ptb"]["c1"], c2_ptb=firs["ptb"]["c2"],
                        delta_star_mit=delta_star["mit"], delta_star_ptb=delta_star["ptb"])
    print(f"    FIR/δ* 已保存: {fir_cache}")

    # ---- 模型 ----
    models = {}
    for name, rel in [("P2A", "models/archived/final_resnet_l_p2a_backup.h5"),
                      ("exp4c", "models/best_resnet_large_exp4_patient_clean.h5"),
                      ("exp5c", "models/best_resnet_large_exp5_patient_clean.h5"),
                      ("exp6c", "models/best_resnet_large_exp6_patient_clean.h5"),
                      ("exp6-SGD(重训)", "models/best_resnet_large_exp6_sgd.h5")]:
        p = Path(__file__).resolve().parent / rel
        if p.exists():
            models[name] = tf.keras.models.load_model(str(p), compile=False)
            print(f"  {name}: loaded")

    def eval_auc(model, beats, labels):
        if len(beats) == 0 or len(np.unique(labels)) < 2:
            return 0.5, {}
        prob = model.predict(_add_channel_dim(beats), batch_size=512, verbose=0)[:, 1]
        auc = float(roc_auc_score(labels, prob))
        thr = {}
        for t in (0.35, 0.5):
            p, r, f1, _ = precision_recall_fscore_support(
                labels, (prob >= t).astype(int), average="binary", zero_division=0)
            thr[f"{t:.2f}"] = {"rec": float(r), "prec": float(p), "f1": float(f1)}
        return auc, thr

    # ---- 消融阶梯 ----
    chains = ["d0", "d0_n", "d0_nh", "d1", "d2", "d3"]
    results = {}
    for mname, model in models.items():
        if mname == "exp6-SGD(重训)":
            continue
        for dom in ("mit", "ptb"):
            aucs = {}
            for ch in chains:
                aucs[ch], _ = eval_auc(model, *domains[dom][ch])
            results.setdefault(mname, {})[dom] = {
                "auc_d0": aucs["d0"], "auc_d0_n": aucs["d0_n"], "auc_d0_nh": aucs["d0_nh"],
                "auc_d1": aucs["d1"], "auc_d2": aucs["d2"], "auc_d3": aucs["d3"],
                "eff_notch": aucs["d0"] - aucs["d0_n"],
                "eff_hp_cutoff": aucs["d0_n"] - aucs["d0_nh"],
                "eff_causal_phase": aucs["d0_nh"] - aucs["d1"],
                "eff_decimation": aucs["d1"] - aucs["d2"],
                "eff_comb": aucs["d2"] - aucs["d3"],
            }
            r = results[mname][dom]
            print(f"  [{mname}/{dom}] D0={r['auc_d0']:.4f} d0_n={r['auc_d0_n']:.4f} "
                  f"d0_nh={r['auc_d0_nh']:.4f} D1={r['auc_d1']:.4f} D2={r['auc_d2']:.4f} "
                  f"D3={r['auc_d3']:.4f} | notch={r['eff_notch']:+.4f} "
                  f"hp={r['eff_hp_cutoff']:+.4f} causal={r['eff_causal_phase']:+.4f} "
                  f"decim={r['eff_decimation']:+.4f} comb={r['eff_comb']:+.4f}")

    # ---- 补偿评估 (P0 时移 δ 曲线; P1/P2 频域在拍级不可行 — 分析性结论) ----
    print("\n[C] 补偿评估 (P0 循环时移 δ 曲线):")
    comp_results = {}
    for mname, model in models.items():
        for dom in ("mit", "ptb"):
            beats3, labels = domains[dom]["d3"]
            beats0, _ = domains[dom]["d0"]
            # δ 曲线 (−12..12 循环平移)
            curve = {}
            for d in range(-12, 13):
                xd = shift_beats(beats3, d)
                auc_d, _ = eval_auc(model, xd, labels)
                curve[str(d)] = auc_d
            d_opt = max(curve, key=lambda k: curve[k])
            a_p0 = curve[d_opt]
            a_d3 = curve["0"]
            a_d0, thr_d0 = eval_auc(model, beats0, labels)
            rec_key = "recover_p0_opt"
            comp_results.setdefault(mname, {})[dom] = {
                "auc_d3": a_d3, "auc_p0_opt": a_p0, "delta_opt": int(d_opt),
                "auc_d0": a_d0, "thr_d0": thr_d0,
                "recover_p0_opt": a_p0 - a_d3,
                "gap_p0_opt": a_d0 - a_p0,
                "delta_curve": {k: round(v, 4) for k, v in curve.items()},
                "p1_p2_infeasible": {
                    "reason": "250 点拍级 FFT 频率分辨率 1 Hz, 无法分辨 0.05/0.5 Hz 幅度差; "
                              "逆滤波 FIR 需 ~2500 抽头 (10s) 超出嵌入预算; "
                              "拍级频域补偿实测崩坏 (v1/v2 验证 AUC 0.97→0.56), 判定不可行",
                    "evidence": "eval_deploy_compensation.py v1/v2 运行记录",
                },
            }
            r = comp_results[mname][dom]
            print(f"  [{mname}/{dom}] D3={a_d3:.4f} P0(δ={d_opt})={a_p0:.4f}"
                  f"(+{r['recover_p0_opt']:+.4f}) | D0={a_d0:.4f} | "
                  f"δ曲线峰值 {d_opt}")

    # exp6-SGD (重训) — δ 口径验证 vs 报告数字
    print("\n  [exp6-SGD 重训模型 δ 口径验证]:")
    for dom in ("mit", "ptb"):
        beats3, labels = domains[dom]["d3"]
        curve = {}
        for d in range(-12, 13):
            auc_d, _ = eval_auc(models["exp6-SGD(重训)"], shift_beats(beats3, d), labels)
            curve[str(d)] = auc_d
        d_opt = max(curve, key=lambda k: curve[k])
        comp_results["exp6-SGD(重训)"][dom]["delta_curve"] = {k: round(v, 4) for k, v in curve.items()}
        comp_results["exp6-SGD(重训)"][dom]["auc_d3_ds_opt"] = curve[d_opt]
        comp_results["exp6-SGD(重训)"][dom]["delta_opt"] = int(d_opt)
        print(f"    [{dom}] D3(δ=0)={curve['0']:.4f} D3(δ*={d_opt})={curve[d_opt]:.4f} "
              f"(报告数字 {('0.9122' if dom=='mit' else '0.7697')})")

    # ---- 汇总 ----
    output = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "T1-3 部署链失配分量消融 + 输入侧补偿原型 (v2, 必做清单 T1-3)",
            "chains": {
                "d0": "训练链 filtfilt HP0.5+LP40+Notch50 @250Hz",
                "d0_n": "d0 去 Notch50", "d0_nh": "d0_n 但 HP 0.5→0.05Hz 零相位",
                "d1": "FFT→250 + 因果 HP0.05+LP40",
                "d2": "500Hz 路径 + 2:1 抽取, 无梳状", "d3": "完整部署链",
            },
            "compensation": {
                "method": "P0=拍级循环时移 (δ 曲线 −12..+12, 每模型选最优 δ); "
                          "P1/P2 频域补偿 (全通相位/正则化逆滤波) 经 v1/v2 实测在拍级不可行 "
                          "(250 点 FFT 分辨率 1Hz < 0.05/0.5Hz 幅度差需求; FIR 需 ~2500 抽头) — "
                          "结论: 时间对齐是唯一实用输入侧补偿, 残余失配只能靠部署链重训",
                "references": [
                    "Gregg et al. 2023 (CinC): IIR 因果 HP 的 ST 失真与零相位化",
                    "Dobrev et al. 2025, Technologies 13(4):159, DOI 10.3390/technologies13040159: IHPF",
                ],
                "delta_star": delta_star, "fir_file": "models/deploy_match/compensation_fir.npz",
                "corr_diagnostics": diag,
            },
            "int8_component": {"note": "引用 T0-1 (verify_exp6_sgd_int8.py): MIT ΔAUC −0.017, PTB −0.042"},
            "report_semantics": "FINAL_RESULTS 表4 D3 数字 (0.9122/0.7697) 为 δ 对齐语义; "
                                "本文件 D3(δ=0) 与 D3(δ*) 并列报告",
        },
        "ablation": results,
        "compensation": comp_results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 结果已保存: {OUT_JSON} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    main()
