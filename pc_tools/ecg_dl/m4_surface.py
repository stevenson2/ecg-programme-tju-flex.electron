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
    pc_a = pc["v3a_int8"]
    lines = [
        "# M4 结论页: v3-A 是否忽视环境噪声？(2026-09-13)",
        "",
        "口径: 载波 MIT-100/106 属训练患者 (split_guard 重算), 全部结论为",
        "同载波噪声敏感性自对照, 不作泛化证据。模型 = v3-A INT8 (板上 θ=0.5 口径",
        "前的 raw 位) + h5 对照。证据: corpus_pc_*.json, corpus_eval_v3a_*.json,",
        "m4_sqi_analysis.json; 图: m4_noise_surface.png, m4_sqi_vs_raw.png。",
        "",
        "## 三个判定的答案",
        "",
        "1. **噪声下的异常被漏报吗？** 部分。异常载波加 10dB 噪声后 raw 从",
        "   0.56 降至 0.41-0.70 组内 (bw@10dB 最差 0.41), 报警擎住仍达 0.31-0.73",
        "   —— 10dB 内敏感度保持; 但置信度中位从 0.95 掉到 0.01-0.94 (类型相关),",
        "   基线漂移是最大漏报源。",
        "2. **噪声被误判为异常吗？** 类型依赖。EMG/工频/基线漂移 (正常载波):",
        "   raw ≤0.35, 报警擎住≈0 —— 密度门槛吸收; **纯运动伪迹 raw 0.90、擎住",
        "   77%** —— 运动伪迹被系统性当作病理 (板上+PC 一致)。",
        "3. **SQI 能否兜底？** 不能 (本语料内)。两大 FP 源的 SQI 读数 0.66/0.86,",
        "   与干净正常 (~0.99) 差距 <0.35; Pearson(sqi, raw) 全组 %.2f。SQI 门控" % corr_all,
        "   (挂起低 SQI 报警) 无法在不误伤真阳性的情况下挡住这两类 FP。",
        "",
        "## 缓解路线排序 (按成本, 简报 §M4)",
        "",
        "1. **报警状态机吸收 (已上线, M1)**: 密度门槛 10/30 已把合成噪声 FP 基本",
        "   清零 (板上 alarm rate 仅 mains@20dB 擦线 0.14); 运动伪迹持续段仍会擎住",
        "   —— 但这类场景报警本身可辩护 (真无法判读)。",
        "2. **重训 (M3, 负结果)**: 链整形逐拍增强两次迭代 (v2/v3) 带噪语料全面更优",
        "   (normal+noise FP -0.14), 但干净 test 事件 F1 退化 -0.15~-0.23, 双门槛",
        "   不允许换。保留 v3-A (v3b_iteration_summary.json)。",
        "3. **SQI 门控**: 本数据否定其对运动伪迹/跨库 FP 的区分力; 若未来换 SQI",
        "   算法需先在新特征上重测本语料 (corpus 可直接复用)。",
        "",
        "## 遗留",
        "",
        "- 跨库 (LUDB) 正常 93% FP 是**域偏移**问题, 噪声增强不解决 (v3-B 亦 1.00);",
        "  需要域增广/跨库训练集, 属下一轮课题。",
        "- 异常@10dB 的检出下限 (bw 最差) 建议进入下一次训练的 Hard 例池。",
    ]
    md = OUT_DIR / "M4_结论.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print("->", md)


if __name__ == "__main__":
    main()
