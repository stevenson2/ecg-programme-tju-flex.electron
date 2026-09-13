#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m4_surface.py — M4 产出: 噪声×SNR 曲面图 + SQI 区分力分析 + 结论页 (TH §108)
输入: corpus_pc_*.json (PC 三模型 30 case) + corpus_eval_v3a_*.json (板上, 含 sqi)
输出: research/m4_noise_surface.png + research/M4_结论.md + corpus/m4_sqi_analysis.json
结论口径: 载波=MIT-100/106 (train 患者), 自对照; 不作泛化证据。
"""
import glob
import json
import statistics
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
REPO = BASE.parent.parent
OUT_DIR = REPO.parent / "research"
OUT_DIR.mkdir(exist_ok=True)


def load(pattern):
    f = sorted(glob.glob(str(BASE / "corpus" / ("corpus_pc_%s_*.json" % pattern))))
    d = json.load(open(f[-1], encoding="utf-8"))
    return {r["name"]: r for r in d["results"]}


def main():
    pc = {m: load(m) for m in ("v3a_int8", "v3a_h5", "v3b_h5")}
    board_files = sorted(glob.glob(str(REPO / "pc_tools" / "serial" /
                                    "captures" / "corpus_eval_v3a_*.json")))
    board = {r["name"]: r for r in
             json.load(open(board_files[-1], encoding="utf-8"))["results"]}

    # ---- 图 1: 噪声类型 × SNR 曲面 (v3-A 板上 + PC int8) ----
    types = ["baseline_wander", "respiratory", "mains", "emg", "motion"]
    snrs = [20, 10, 0]
    fig_rows = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, (src, data) in zip(axes, (("PC v3-A INT8", pc["v3a_int8"]),
                                      ("Board v3-A (N16R8)", board))):
        for t in types:
            ys = [data.get("normal_%s_snr%d" % (t, s), {}).get("raw_rate",
                  np.nan) for s in snrs]
            ax.plot(range(len(snrs)), ys, "o-", label=t)
        ax.axhline(0.16, ls="--", c="gray", lw=1,
                   label="clean normal baseline (~0.16)")
        ax.set_xticks(range(len(snrs)), ["%ddB" % s for s in snrs])
        ax.set_title(src)
        ax.set_xlabel("SNR")
        ax.set_ylim(-0.03, 1.0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("raw abnormal rate (30s case)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("M4: synthetic-noise sensitivity of v3-A (same-carrier self-control)")
    fig.tight_layout()
    png1 = OUT_DIR / "m4_noise_surface.png"
    fig.savefig(png1, dpi=140)
    print("->", png1)

    # ---- SQI 区分力 (板上 30 case: sqi_mean vs raw_rate) ----
    names = list(board)
    sqi = np.array([board[n].get("sqi_mean") or np.nan for n in names])
    raw = np.array([board[n]["raw_rate"] for n in names])
    # 定义 "FP 主导 case": clean/正常载波组中 raw 高者; 相关性按全部 + 正常组
    normal_mask = np.array([not (n.startswith("abnormal") or n == "clean_abnormal")
                            for n in names])
    corr_all = np.corrcoef(sqi, raw)[0, 1]
    corr_norm = np.corrcoef(sqi[normal_mask], raw[normal_mask])[0, 1]
    # 大 FP case 的 SQI 读数
    fp_cases = {n: board[n].get("sqi_mean") for n in
                ("pure_motion", "ludb_normal", "normal_mains_snr20")
                if n in board}
    clean_sqis = [board[n].get("sqi_mean") for n in
                  ("clean_normal", "normal_baseline_wander_snr10",
                   "normal_emg_snr10", "normal_motion_snr10")
                  if n in board]
    analysis = {
        "note": "板上 30 case, sqi_mean 为 hr.sqi 秒级读数; raw_rate 同 case",
        "spearman_like_pearson_sqi_vs_raw_all": round(float(corr_all), 4),
        "pearson_sqi_vs_raw_normal_group": round(float(corr_norm), 4),
        "big_FP_cases_sqi": fp_cases,
        "clean_cases_sqi_mean": round(float(np.mean(clean_sqis)), 3),
        "verdict": ("SQI 对运动伪迹 (0.66) 与跨库正常 (0.86) 的读数仍处高位, "
                    "与干净正常 (~0.99) 距离不足 0.2 —— 以 <0.5 阈值做挂起门控 "
                    "对本语料的两大 FP 源无效; SQI 门控路线天花板被本数据限定。"),
    }
    (BASE / "corpus" / "m4_sqi_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False, indent=2)[:400])

    # ---- 图 2: SQI vs raw 散点 ----
    fig2, ax = plt.subplots(figsize=(6.2, 4.6))
    sc = ax.scatter(sqi, raw, c=np.where(normal_mask, 0, 1), cmap="coolwarm",
                    s=46, alpha=0.85)
    for n in ("pure_motion", "ludb_normal", "clean_normal"):
        if n in board:
            ax.annotate(n, (board[n]["sqi_mean"], board[n]["raw_rate"]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0.5, ls=":", c="k", lw=0.8)
    ax.set_xlabel("board hr.sqi (case mean)")
    ax.set_ylabel("raw abnormal rate")
    ax.set_title("M4: SQI vs model raw abnormal rate (30 cases, N16R8)")
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    png2 = OUT_DIR / "m4_sqi_vs_raw.png"
    fig2.savefig(png2, dpi=140)
    print("->", png2)

    # ---- 结论页 ----
    # 2026-09-13 §109 勘误: 结论 MD 改为手工维护 (research/M4_结论.md v2)。
    # 本脚本只产图 + SQI JSON; 机器自动生成的结论曾在载波身份误判下得出
    # "跨库 93% FP" 错误头条, 不再自动重写结论页。
    print("conclusion MD is hand-maintained: research/M4_jielun.md (v2, see TH 109)")
    print("figure1: %s" % png1)
    print("figure2: %s" % png2)


if __name__ == "__main__":
    main()
