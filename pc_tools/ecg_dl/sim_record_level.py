#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sim_record_level.py — 记录级决策层可行性研究 (zero-training, §8.9.9)

在现有单拍模型之上评估"记录级决策层": 把拍级概率按记录内拍序聚合为 K 拍/段
(默认 K=30 ≈ 30s @60bpm, 对齐 Apple Watch 30s 筛查窗口), 回答:
  (1) 段级 AUC 是否优于拍级 AUC?
  (2) 段级误报率是否相对拍级下降 (MIT 21.6% / PTB 0.4%)?

数据/模型口径与 eval_binary_all.py 完全一致:
  - _deploy npz, 患者级划分 (seed 42), P2A=archived/final_resnet_l_p2a_backup.h5
    (MIT 专家), KD=kd_a070_t1.h5 (PTB 专家), CPU 推理。
  - 拍级基线先复现: P2A/MIT θ=0.5 → R 0.9353/P 0.3889/误报 21.6%;
    KD/PTB θ=0.5 → R 0.3230/P 0.9967/误报 0.4%。

用法:
  python3 sim_record_level.py            # 完整评估 (长时, 生成 JSON + 4 图)
  python3 sim_record_level.py --smoke    # 快速自检 (随机概率, 不加载模型)
"""
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── 常量 ─────────────────────────────────────────────────────────────────────
MODELS = Path(__file__).resolve().parent / "models"
FIG_DIR = MODELS / "figures" / "patient"
TH_BEAT = 0.5                       # 拍级阈值 (abn_frac / nofm 用)
K_LIST = [15, 30, 60, 120]          # 段长扫
TAU_SEG_LIST = [0.0, 0.05, 0.10]    # 段标签异常占比阈值
TAU_GRID = sorted(set([round(v, 2) for v in np.arange(0.05, 1.0, 0.05)] + [0.35, 0.5, 0.65]))
PREVALENCE = [0.015, 0.034, 0.05, 0.10, 0.128, 0.25, 0.781]  # 筛查 1.5%–3.4%
STRATEGIES = ["max", "mean", "p95", "abn_frac", "nofm_3_5"]

# 拍级基线期望 (binary_class_eval_all.json, θ=0.5)
BASELINE_EXPECTED = {
    ("P2A", "MIT"): {"R": 0.9353, "P": 0.3889, "FP": 0.2160, "AUC": 0.9233},
    ("KD", "PTB"):  {"R": 0.3230, "P": 0.9967, "FP": 0.0039, "AUC": 0.8360},
}

# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def _floatify(d):
    """numpy 标量转 Python 标量 (JSON 序列化)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _floatify(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, np.bool_):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


def _beat_stats(y, prob, th):
    """拍级 P/R/F1/误报/报警 (eval_binary_all.stats 同款)."""
    from sklearn.metrics import precision_recall_fscore_support
    pred = (prob >= th).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    fp_normal = int(((pred == 1) & (y == 0)).sum()) / max(1, int((y == 0).sum()))
    alarm = float((pred == 1).mean())
    return {"P": float(prec), "R": float(rec), "F1": float(f1),
            "误报": float(fp_normal), "报警": alarm}


def _seg_metrics(y, score, tau):
    """段级 P/R/F1/误报/报警 (口径与拍级 stats 一致)."""
    from sklearn.metrics import precision_recall_fscore_support
    pred = (score >= tau).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    fp_normal = int(((pred == 1) & (y == 0)).sum()) / max(1, int((y == 0).sum()))
    alarm = float((pred == 1).mean())
    return {"P": float(prec), "R": float(rec), "F1": float(f1),
            "误报": float(fp_normal), "报警": alarm}


def _segment_scores(pp, theta_beat=TH_BEAT):
    """单段五种聚合分数. pp: 段内拍概率 (1D)."""
    s = {}
    s["max"] = float(pp.max())
    s["mean"] = float(pp.mean())
    s["p95"] = float(np.percentile(pp, 95))
    trig = (pp >= theta_beat).astype(np.int32)
    s["abn_frac"] = float(trig.mean())
    M, N = 5, 3
    if len(pp) >= M:
        from numpy.lib.stride_tricks import sliding_window_view
        wins = sliding_window_view(trig, M)
        s["nofm_3_5"] = float((wins.sum(axis=1) >= N).mean())
    else:
        s["nofm_3_5"] = float(trig.mean())
    return s


def build_segments(probs, labels, record_ids, K, tau_seg):
    """按记录内拍序切成 K 拍/段 (非重叠, 尾部不足 K 保留).

    返回: (scores_dict, y_seg, seg_rids, seg_abn_fracs)
      - scores_dict: strategy -> np.ndarray (n_seg,)
      - y_seg:      段标签 = 段内异常占比 >= tau_seg
      - seg_rids:   段所属记录 id
      - seg_abn_fracs: 段内异常拍占比 (记录级标签用)
    """
    from numpy.lib.stride_tricks import sliding_window_view  # noqa: F401
    unique_rids = np.unique(record_ids)
    scores = {s: [] for s in STRATEGIES}
    y_seg, seg_rids, seg_abn = [], [], []
    for rid in unique_rids:
        mask = record_ids == rid
        p = probs[mask]
        y = labels[mask]
        n = len(p)
        for start in range(0, n, K):
            end = min(start + K, n)
            pp, yy = p[start:end], y[start:end]
            abn_frac = float((yy == 1).mean())
            seg_abn.append(abn_frac)
            y_seg.append(1 if abn_frac >= tau_seg else 0)
            seg_rids.append(int(rid))
            ss = _segment_scores(pp)
            for strat in STRATEGIES:
                scores[strat].append(ss[strat])
    out = {k: np.asarray(v) for k, v in scores.items()}
    return out, np.asarray(y_seg), np.asarray(seg_rids), np.asarray(seg_abn)


def record_level_scores(scores, seg_abn, seg_rids):
    """记录级: label = 记录含任一异常拍; score1 = max 段分数 (max 策略);
    score2 = 记录内异常占比>=5% 的段比例."""
    unique = np.unique(seg_rids)
    rec_y, rec_s1, rec_s2 = [], [], []
    for rid in unique:
        m = seg_rids == rid
        rec_y.append(1 if (seg_abn[m] > 0).any() else 0)
        rec_s1.append(float(scores["max"][m].max()))
        rec_s2.append(float((seg_abn[m] >= 0.05).mean()))
    return (np.asarray(rec_y), np.asarray(rec_s1), np.asarray(rec_s2),
            np.asarray(unique, dtype=np.int64))


def prevalence_precision(R, FP, pi):
    """P = R·π / (R·π + FP·(1−π))"""
    R, FP = float(R), float(FP)
    num = R * pi
    den = num + FP * (1.0 - pi)
    return num / max(1e-12, den)


# ─── 完整评估 ─────────────────────────────────────────────────────────────────

def full_main() -> int:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score, roc_curve

    from data.dataset import (
        set_npz_suffix, load_mit_incart_merged, add_channel_dim, load_ptb_data,
    )
    from data.patient_split import (
        build_mit_patient_map, build_incart_patient_map,
        build_ptb_patient_map, patient_level_split,
    )

    print("=" * 96)
    print("记录级决策层可行性研究 (zero-training, §8.9.9)")
    print("=" * 96)

    # ---- 1. 数据 (与 eval_binary_all.py 完全一致) ------------------------------
    set_npz_suffix("_deploy")
    print("\n[1/6] 加载数据...")
    mi = load_mit_incart_merged()
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({r + 100000: "inc_" + p
                 for r, p in build_incart_patient_map().items()})
    tr, va, te, stats_mit = patient_level_split(mi["record_ids"], pmap)
    x_mit = mi["beats"][te]; y_mit = mi["labels"][te]; rid_mit = mi["record_ids"][te]

    ptb = load_ptb_data()
    pmap_ptb = build_ptb_patient_map()
    tr2, va2, te2, stats_ptb = patient_level_split(ptb["record_ids"], pmap_ptb)
    x_ptb = ptb["beats"][te2]; y_ptb = ptb["labels"][te2]; rid_ptb = ptb["record_ids"][te2]
    print(f"  MIT test: {len(y_mit)} beats, {int(y_mit.sum())} abnormal, "
          f"{len(np.unique(rid_mit))} records")
    print(f"  PTB test: {len(y_ptb)} beats, {int(y_ptb.sum())} abnormal, "
          f"{len(np.unique(rid_ptb))} records")

    # 记录连续性检查 (沿用 sim_temporal_agg)
    for label, rids in [("MIT+INCART", rid_mit), ("PTB", rid_ptb)]:
        n_sw = int((np.diff(rids) != 0).sum()); n_rec = len(np.unique(rids))
        ok = "OK" if n_sw == n_rec - 1 else "NON-CONTIGUOUS"
        print(f"  记录连续性 {label}: {n_rec} records, {n_sw} switches → {ok}")

    # ---- 2. 模型推理 (CPU) ----------------------------------------------------
    print("\n[2/6] 加载模型并推理 (CPU)...")
    p2a = tf.keras.models.load_model(str(MODELS / "archived" / "final_resnet_l_p2a_backup.h5"),
                                     compile=False)
    kd = tf.keras.models.load_model(str(MODELS / "kd_a070_t1.h5"), compile=False)
    p_p2a_mit = p2a.predict(add_channel_dim(x_mit), verbose=0, batch_size=1024)[:, 1]
    p_kd_ptb = kd.predict(add_channel_dim(x_ptb), verbose=0, batch_size=1024)[:, 1]
    del p2a, kd
    print("  推理完成.")

    configs = [
        ("P2A", "MIT", p_p2a_mit, y_mit, rid_mit),
        ("KD",  "PTB", p_kd_ptb,  y_ptb, rid_ptb),
    ]

    # ---- 3. 拍级基线复现 ------------------------------------------------------
    print("\n[3/6] 拍级基线复现 (θ=0.5)...")
    results = {"meta": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": "identical to eval_binary_all.py",
        "models": {"MIT": "archived/final_resnet_l_p2a_backup.h5",
                   "PTB": "kd_a070_t1.h5"},
        "K_list": K_LIST, "tau_seg_list": TAU_SEG_LIST,
        "theta_beat": TH_BEAT, "tau_grid": TAU_GRID,
        "prevalence_points": PREVALENCE,
    }, "baseline_reproduction": {}, "segment_level": {}, "record_level": {},
       "research_answer": {}}

    baselines = {}
    for model, dom, p, y, rids in configs:
        bl = _beat_stats(y, p, 0.5)
        auc = float(roc_auc_score(y, p))
        exp = BASELINE_EXPECTED[(model, dom)]
        checks = {
            "R": abs(bl["R"] - exp["R"]) <= 0.01,
            "P": abs(bl["P"] - exp["P"]) <= 0.01,
            "FP": abs(bl["误报"] - exp["FP"]) <= 0.01,
            "AUC": abs(auc - exp["AUC"]) <= 0.01,
        }
        passed = all(checks.values())
        results["baseline_reproduction"][f"{model}_{dom}"] = {
            "R": bl["R"], "P": bl["P"], "误报": bl["误报"], "AUC": auc,
            "expected": exp, "checks": checks, "PASS": passed}
        baselines[(model, dom)] = {"R": bl["R"], "P": bl["P"], "FP": bl["误报"], "AUC": auc}
        print(f"  {model}/{dom}: R={bl['R']:.4f} P={bl['P']:.4f} "
              f"误报={bl['误报']*100:.1f}% AUC={auc:.4f}  → "
              f"{'PASS' if passed else 'FAIL ' + str(checks)}")

    # ---- 4. 段级评估 ----------------------------------------------------------
    print("\n[4/6] 段级评估...")
    # 4a. 默认配置 (K=30, τ_seg=0.05) 全策略 AUC + P/R/F1 网格
    default = {}
    for model, dom, p, y, rids in configs:
        K, tau_seg = 30, 0.05
        scores, y_seg, seg_rids, seg_abn = build_segments(p, y, rids, K, tau_seg)
        entry = {"n_seg": int(len(y_seg)),
                 "pos_seg": int(y_seg.sum()),
                 "seg_abn_share": float(y_seg.mean()),
                 "strategies": {}}
        for strat in STRATEGIES:
            entry["strategies"][strat] = {"AUC": float(roc_auc_score(y_seg, scores[strat]))}
        # P/R/F1 网格 (默认策略 = AUC 最高者)
        best_strat = max(STRATEGIES, key=lambda s: entry["strategies"][s]["AUC"])
        grid = {}
        for tau in TAU_GRID:
            grid[str(tau)] = _seg_metrics(y_seg, scores[best_strat], tau)
        best_f1_tau = max(TAU_GRID, key=lambda t: grid[str(t)]["F1"])
        entry["best_strategy"] = best_strat
        entry["P_R_F1_grid"] = grid
        entry["at_0.5"] = grid["0.5"]
        entry["at_best_F1"] = grid[str(best_f1_tau)]
        entry["at_best_F1_tau"] = float(best_f1_tau)
        default[f"{model}_{dom}"] = entry
        print(f"  {model}/{dom} K={K} τ_seg={tau_seg}: n_seg={len(y_seg)} "
              f"异常段占比={y_seg.mean()*100:.1f}%")
        print(f"    AUC/策略: " + "  ".join(
            f"{s}={entry['strategies'][s]['AUC']:.4f}" for s in STRATEGIES))
        print(f"    best={best_strat} @τ=0.5: P={grid['0.5']['P']:.3f} "
              f"R={grid['0.5']['R']:.3f} F1={grid['0.5']['F1']:.3f} "
              f"误报={grid['0.5']['误报']*100:.1f}% | "
              f"best-F1: P={entry['at_best_F1']['P']:.3f} "
              f"R={entry['at_best_F1']['R']:.3f} F1={entry['at_best_F1']['F1']:.3f} "
              f"误报={entry['at_best_F1']['误报']*100:.1f}%")
    results["segment_level"]["default_K30_ts005"] = default

    # 4b. 段长扫描 AUC (默认策略, τ_seg=0.05)
    seglen = {}
    for model, dom, p, y, rids in configs:
        row = {}
        for K in K_LIST:
            scores, y_seg, _, _ = build_segments(p, y, rids, K, 0.05)
            best = max(STRATEGIES, key=lambda s: roc_auc_score(y_seg, scores[s]))
            row[str(K)] = {"AUC": float(roc_auc_score(y_seg, scores[best])),
                           "strategy": best, "n_seg": int(len(y_seg))}
        seglen[f"{model}_{dom}"] = row
        print(f"  [段长] {model}/{dom}: " + "  ".join(
            f"K={K} AUC={row[str(K)]['AUC']:.4f}" for K in K_LIST))
    results["segment_level"]["auc_vs_K"] = seglen

    # 4c. τ_seg 影响 (K=30, 默认策略)
    tseg = {}
    for model, dom, p, y, rids in configs:
        row = {}
        for tau_seg in TAU_SEG_LIST:
            scores, y_seg, _, _ = build_segments(p, y, rids, 30, tau_seg)
            best = max(STRATEGIES, key=lambda s: roc_auc_score(y_seg, scores[s]))
            row[str(tau_seg)] = {"AUC": float(roc_auc_score(y_seg, scores[best])),
                                 "pos_frac": float(y_seg.mean())}
        tseg[f"{model}_{dom}"] = row
    results["segment_level"]["auc_vs_tauseg"] = tseg

    # ---- 5. 记录级评估 --------------------------------------------------------
    print("\n[5/6] 记录级评估...")
    for model, dom, p, y, rids in configs:
        scores, y_seg, seg_rids, seg_abn = build_segments(p, y, rids, 30, 0.05)
        rec_y, rec_s1, rec_s2, rec_ids = record_level_scores(scores, seg_abn, seg_rids)
        row = {
            "n_records": int(len(rec_y)),
            "pos_records": int(rec_y.sum()),
            "auc_score_max": float(roc_auc_score(rec_y, rec_s1)),
            "auc_score_fracpos": float(roc_auc_score(rec_y, rec_s2)),
            "grid": {str(tau): _seg_metrics(rec_y, rec_s1, tau) for tau in TAU_GRID},
        }
        row["at_0.5"] = row["grid"]["0.5"]
        results["record_level"][f"{model}_{dom}"] = row
        print(f"  {model}/{dom}: n_rec={len(rec_y)} pos={int(rec_y.sum())} "
              f"AUC(max)={row['auc_score_max']:.4f} AUC(fracpos)={row['auc_score_fracpos']:.4f} "
              f"@τ=0.5: P={row['at_0.5']['P']:.3f} R={row['at_0.5']['R']:.3f} "
              f"误报={row['at_0.5']['误报']*100:.1f}%")

    # ---- 6. 研究结论 + 图表 ---------------------------------------------------
    print("\n[6/6] 研究结论 + 图表...")
    fig_dir = FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    roc_curves = {}
    answer = {}

    for model, dom, p, y, rids in configs:
        key = f"{model}_{dom}"
        # 拍级 ROC
        fpr_b, tpr_b, _ = roc_curve(y, p)
        auc_b = baselines[(model, dom)]["AUC"]
        # 段级 ROC (默认配置 best strategy)
        d = default[key]
        K, tau_seg = 30, 0.05
        scores, y_seg, _, _ = build_segments(p, y, rids, K, tau_seg)
        best = d["best_strategy"]
        fpr_s, tpr_s, _ = roc_curve(y_seg, scores[best])
        auc_s = d["strategies"][best]["AUC"]
        roc_curves[key] = {"beat_fpr": fpr_b.tolist(), "beat_tpr": tpr_b.tolist(),
                           "beat_auc": float(auc_b),
                           "seg_fpr": fpr_s.tolist(), "seg_tpr": tpr_s.tolist(),
                           "seg_auc": float(auc_s), "seg_strategy": best}

        # 研究回答 (固定 τ=0.5 对比 + 匹配召回率对比)
        delta_auc = auc_s - auc_b
        bl = baselines[(model, dom)]
        bl_fp = bl["FP"]; bl_r = bl["R"]; bl_p = bl["P"]
        seg_fp_at05 = d["at_0.5"]["误报"]
        delta_fp = bl_fp - seg_fp_at05
        # 匹配召回率: 段级 τ_grid 上选 R 最接近拍级 R@0.5 的阈值, 比较该点 FP/P
        grid = d["P_R_F1_grid"]
        match_tau = min(TAU_GRID, key=lambda t: abs(grid[str(t)]["R"] - bl_r))
        seg_m = grid[str(match_tau)]
        # 更高召回率工作点: R >= 拍级R 且 P 最大的阈值 (best-F1 对比)
        bf1 = d["at_best_F1"]
        matched = {
            "tau": float(match_tau),
            "seg_R": seg_m["R"], "seg_P": seg_m["P"], "seg_FP": seg_m["误报"],
            "delta_FP_pp_matched": float((bl_fp - seg_m["误报"]) * 100.0),
            "delta_P_matched": float(seg_m["P"] - bl_p),
            "best_F1_tau": d["at_best_F1_tau"],
            "best_F1_R": bf1["R"], "best_F1_P": bf1["P"], "best_F1_FP": bf1["误报"],
        }
        ans = (
            f"{model}/{dom}: 段级AUC={auc_s:.4f} vs 拍级={auc_b:.4f} "
            f"(Δ{delta_auc:+.4f}, {'提升' if delta_auc > 0 else '未提升'}); "
            f"匹配召回率工作点: 段级R={seg_m['R']:.3f}/P={seg_m['P']:.3f}/误报={seg_m['误报']*100:.1f}% "
            f"vs 拍级R={bl_r:.3f}/P={bl_p:.3f}/误报={bl_fp*100:.1f}% "
            f"(Δ误报{matched['delta_FP_pp_matched']:+.1f}pp, "
            f"{'下降' if matched['delta_FP_pp_matched'] > 0 else '未下降'}); "
            f"best-F1: R={bf1['R']:.3f}/P={bf1['P']:.3f}"
        )
        answer[key] = {"segment_auc": float(auc_s), "beat_auc": float(auc_b),
                       "delta_auc": float(delta_auc),
                       "seg_fp_at_0.5": float(seg_fp_at05),
                       "beat_fp": float(bl_fp),
                       "delta_fp_pp_at_0.5": float(delta_fp * 100.0),
                       "matched_recall": matched,
                       "verdict": ans}
        print(f"  → {ans}")
    results["research_answer"] = answer
    results["roc_curves"] = roc_curves

    # ---- 图 1: 段级 vs 拍级 ROC -----------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    colors = {"P2A_MIT": "#c0392b", "KD_PTB": "#2471a3"}
    for model, dom, *_ in configs:
        key = f"{model}_{dom}"
        rc = roc_curves[key]
        ax.plot(rc["beat_fpr"], rc["beat_tpr"], "--", color=colors[key],
                lw=1.6, label=f"{key} beat (AUC={rc['beat_auc']:.3f})")
        ax.plot(rc["seg_fpr"], rc["seg_tpr"], "-", color=colors[key], lw=2.2,
                label=f"{key} segment {rc['seg_strategy']} (AUC={rc['seg_auc']:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Record-Level Decision Layer: Segment vs Beat ROC (K=30, tau_seg=0.05)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "record_level_roc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] record_level_roc.png")

    # ---- 图 2: AUC vs 段长 -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for model, dom, *_ in configs:
        key = f"{model}_{dom}"
        row = seglen[key]
        xs = K_LIST
        ys = [row[str(k)]["AUC"] for k in xs]
        ax.plot(xs, ys, marker="o", lw=2, color=colors[key],
                label=f"{key} (strategy={row['30']['strategy']})")
        ax.axhline(baselines[(model, dom)]["AUC"], ls="--", lw=1.2,
                   color=colors[key], alpha=0.6,
                   label=f"{key} beat AUC={baselines[(model, dom)]['AUC']:.3f}")
    ax.axvspan(28, 32, color="orange", alpha=0.12)
    ax.annotate("K=30 (30s screening)", xy=(30, ax.get_ylim()[0]),
                xytext=(30, ax.get_ylim()[0]), fontsize=9, color="darkorange")
    ax.set_xlabel("Segment Length K (beats)"); ax.set_ylabel("AUC")
    ax.set_title("Segment-Level AUC vs Segment Length (tau_seg=0.05)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "record_level_seglen.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] record_level_seglen.png")

    # ---- 图 3: 聚合策略 AUC 对比 ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(len(STRATEGIES)); w = 0.36
    for i, (model, dom, *_rest) in enumerate(configs):
        key = f"{model}_{dom}"
        aucs = [default[key]["strategies"][s]["AUC"] for s in STRATEGIES]
        ax.bar(x + i * w - w / 2, aucs, w, label=key, color=colors[key], alpha=0.85)
        for xi, v in zip(x + i * w - w / 2, aucs):
            ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(STRATEGIES)
    ax.set_ylabel("Segment AUC"); ax.set_ylim(0.5, 1.0)
    ax.set_title("Segment-Level AUC by Aggregation Strategy (K=30, tau_seg=0.05)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "record_level_aggstrat.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] record_level_aggstrat.png")

    # ---- 图 4: 段级 P vs 异常患病率 (筛查区间标注) -----------------------------
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for model, dom, *_ in configs:
        key = f"{model}_{dom}"
        d = default[key]
        # 拍级工作点 (θ=0.5)
        Rb, FPb = baselines[(model, dom)]["R"], baselines[(model, dom)]["FP"]
        # 段级工作点: τ_grid 上选 R 最接近拍级 R 的阈值
        grid = d["P_R_F1_grid"]
        best_tau = min(TAU_GRID, key=lambda t: abs(grid[str(t)]["R"] - Rb))
        Rs, FPs = grid[str(best_tau)]["R"], grid[str(best_tau)]["误报"]
        pi_arr = np.linspace(0.005, 0.05, 300)
        pb = [prevalence_precision(Rb, FPb, pi) for pi in pi_arr]
        ps = [prevalence_precision(Rs, FPs, pi) for pi in pi_arr]
        ax.plot(pi_arr * 100, pb, "--", color=colors[key], lw=1.6,
                label=f"{key} beat (R={Rb:.3f}, FP={FPb*100:.1f}%)")
        ax.plot(pi_arr * 100, ps, "-", color=colors[key], lw=2.2,
                label=f"{key} segment τ={best_tau} (R={Rs:.3f}, FP={FPs*100:.1f}%)")
    ax.axvspan(1.5, 3.4, color="red", alpha=0.10)
    ax.annotate("Screening prevalence 1.5-3.4%:\nprecision collapses",
                xy=(2.4, 0.08), fontsize=9, color="darkred",
                ha="center", bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_xlabel("Abnormal Prevalence π (%)")
    ax.set_ylabel("Decision Precision P")
    ax.set_title("Precision vs Abnormal Prevalence: Beat vs Segment Decision")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "record_level_prior_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] record_level_prior_curve.png")

    # ---- 保存 JSON ------------------------------------------------------------
    out_path = MODELS / "record_level_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_floatify(results), f, indent=2, ensure_ascii=False)
    print(f"\n[Save] {out_path}")

    # ---- 汇总 ----------------------------------------------------------------
    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(f"{'域':<10}{'拍级AUC':<10}{'段级AUC':<10}{'ΔAUC':<9}"
          f"{'拍级误报':<10}{'段级误报':<10}{'Δ误报(pp)':<10}")
    for key, v in answer.items():
        print(f"{key:<10}{v['beat_auc']:<10.4f}{v['segment_auc']:<10.4f}"
              f"{v['delta_auc']:<+9.4f}{v['beat_fp']*100:<10.1f}"
              f"{v['seg_fp_at_0.5']*100:<10.1f}{v['delta_fp_pp_at_0.5']:<+10.1f}")
    print("\n匹配召回率工作点对比 (段级 τ 选 R≈拍级R@0.5):")
    for key, v in answer.items():
        m = v["matched_recall"]
        print(f"  {key}: 段级 R={m['seg_R']:.3f}/P={m['seg_P']:.3f}/误报={m['seg_FP']*100:.1f}% "
              f"(Δ误报{m['delta_FP_pp_matched']:+.1f}pp, ΔP{m['delta_P_matched']:+.3f})")
    print("\nbest-F1 工作点对比:")
    for key, v in answer.items():
        m = v["matched_recall"]
        print(f"  {key}: 段级 R={m['best_F1_R']:.3f}/P={m['best_F1_P']:.3f}/误报={m['best_F1_FP']*100:.1f}% "
              f"@τ={m['best_F1_tau']:.2f}")
    print("\n结论: 记录级决策层 (K 拍聚合) 是否提升 AUC / 降低误报 → 见各域 verdict 与 JSON。")
    print("Done.")
    return 0


# ─── 冒烟模式 ─────────────────────────────────────────────────────────────────

def smoke_main() -> int:
    rng = np.random.default_rng(0)
    print("[smoke] 段构造 + 聚合 + 指标自检 (随机概率, 不加载模型)")
    for label, n_rec, rec_len in [("MIT-like", 2, 150), ("PTB-like", 2, 90)]:
        rec_ids = np.concatenate([np.full(rec_len, 1000 + i) for i in range(n_rec)])
        n = len(rec_ids)
        probs = rng.random(n)
        labels = (rng.random(n) < 0.15).astype(int)
        for K in [15, 30]:
            for tau_seg in [0.0, 0.05]:
                scores, y_seg, seg_rids, seg_abn = build_segments(
                    probs, labels, rec_ids, K, tau_seg)
                assert len(scores["max"]) == len(y_seg) == len(seg_rids)
                assert set(scores.keys()) == set(STRATEGIES)
                assert all(np.isfinite(s).all() for s in scores.values())
                rec_y, rec_s1, rec_s2, rec_ids2 = record_level_scores(
                    scores, seg_abn, seg_rids)
                assert len(rec_y) == n_rec
                print(f"  {label} K={K} τ_seg={tau_seg}: "
                      f"n_seg={len(y_seg)} (pos={int(y_seg.sum())}), "
                      f"n_rec={len(rec_y)} (pos={int(rec_y.sum())}), "
                      f"score_range=[{scores['max'].min():.2f},{scores['max'].max():.2f}]")
    # 指标函数
    from sklearn.metrics import roc_auc_score
    y = np.array([0, 1, 0, 1, 1]); s = np.array([0.1, 0.6, 0.2, 0.7, 0.9])
    assert 0.5 <= roc_auc_score(y, s) <= 1.0
    m = _seg_metrics(y, s, 0.5)
    print(f"  _seg_metrics @0.5: {m}")
    print("[smoke] ALL PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="记录级决策层可行性研究")
    ap.add_argument("--smoke", action="store_true", help="快速自检 (不加载模型)")
    args = ap.parse_args()
    sys.exit(smoke_main() if args.smoke else full_main())
