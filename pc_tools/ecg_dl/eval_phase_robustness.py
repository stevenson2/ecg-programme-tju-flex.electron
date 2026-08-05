#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_phase_robustness.py — T2-5: 测试时偏移敏感性曲线 + 相位鲁棒性指标
======================================================================
任务: 必做清单 T2-5 评估侧
方法: 测试拍 R 峰偏移 δ ∈ {0, ±2, ±5, ±10, ±20} (拍级循环平移, T1-3 P0 语义)
      → AUC(δ) 敏感性曲线 → 相位鲁棒性指标
指标定义 (与通用时序 S-Cons. 区分, 本指标专指 R 峰窗口偏移/部署链群延迟语义):
  PR-ΔAUC(δ)  = AUC(δ) − AUC(0)                     (逐点损失)
  PR-AUC20    = mean(AUC(δ)) for δ ∈ {−20..20}      (平均鲁棒性)
  PR-drop20   = AUC(0) − min(AUC(δ)), |δ|≤20        (最坏点损失)
对比: exp6-SGD (无相位增强基线) vs exp6-phase (±10 训练)
输出: models/phase_robustness_eval.json + figures/patient/phase_robustness.png
用法 (WSL): export ECG_PROCESSED_DIR=$HOME/ecg_data; python3 eval_phase_robustness.py
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_deploy_match import CACHE_DIR, _add_channel_dim

MODELS_DIR = Path(__file__).resolve().parent / "models"
OUT_JSON = MODELS_DIR / "phase_robustness_eval.json"
OFFSETS = [0, 2, 5, 10, 20]  # 正偏移 = 循环右移 (R 峰窗口错位模拟)


def shift_beats(beats, delta):
    if delta == 0:
        return beats
    return np.roll(beats, delta, axis=1)


def main():
    t0 = time.time()
    print("=" * 70)
    print("T2-5 相位鲁棒性评估")
    print("=" * 70)

    models = {}
    for name, rel in [("exp6-SGD(基线,无相位增强)", "best_resnet_large_exp6_sgd.h5"),
                      ("exp6-phase(±10相位增强)", "best_resnet_large_exp6_phase.h5")]:
        p = MODELS_DIR / rel
        if p.exists():
            models[name] = tf.keras.models.load_model(str(p), compile=False)
            print(f"  {name}: loaded")
        else:
            print(f"  {name}: 缺失 ({rel}) — 训练中或未完成")

    if not models:
        print("无可用模型, 退出")
        return 1

    results = {}
    for mname, model in models.items():
        for dom in ("mit", "ptb"):
            d = np.load(CACHE_DIR / f"{dom}_deploy_match.npz")
            beats, labels = d["beats_deploy"], d["labels"]
            curve = {}
            for delta in OFFSETS:
                for sgn in (1, -1):
                    if delta == 0 and sgn == -1:
                        continue
                    xd = shift_beats(beats, sgn * delta)
                    prob = model.predict(_add_channel_dim(xd), batch_size=512, verbose=0)[:, 1]
                    curve[str(sgn * delta)] = float(roc_auc_score(labels, prob))
            auc0 = curve["0"]
            # 相位鲁棒性指标
            vals = np.array([curve[str(d)] for d in OFFSETS] + [curve[str(-d)] for d in OFFSETS[1:]])
            pr_auc20 = float(vals.mean())
            pr_drop = float(auc0 - vals.min())
            results.setdefault(mname, {})[dom] = {
                "auc_delta": {k: round(v, 4) for k, v in curve.items()},
                "pr_auc20": round(pr_auc20, 4),
                "pr_drop20": round(pr_drop, 4),
                "auc_d0": round(auc0, 4),
            }
            print(f"  [{mname}/{dom}] AUC(0)={auc0:.4f} PR-AUC20={pr_auc20:.4f} "
                  f"PR-drop20={pr_drop:.4f}")
            for k, v in curve.items():
                print(f"      δ={k:>3}: {v:.4f} ({v - auc0:+.4f})")

    # 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for i, dom in enumerate(("mit", "ptb")):
            ax = axes[i]
            for mname in results:
                curve = results[mname][dom]["auc_delta"]
                ds = sorted(map(int, curve.keys()))
                ax.plot(ds, [curve[str(d)] for d in ds], marker="o",
                        label=mname.split("(")[0])
            ax.set_title(f"{dom.upper()} 域: 偏移量 vs AUC")
            ax.set_xlabel("R 峰偏移 δ (样本 @250Hz)")
            ax.set_ylabel("AUC")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        out_fig = MODELS_DIR / "figures" / "patient" / "phase_robustness.png"
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fig, dpi=150)
        print(f"  图已保存: {out_fig}")
    except Exception as e:
        print(f"  绘图失败: {e}")

    output = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "T2-5 相位鲁棒性 (必做清单 T2-5)",
            "method": "测试拍 R 峰偏移 δ∈{0,±2,±5,±10,±20} (循环平移, T1-3 P0 语义)",
            "indicators": {
                "PR-ΔAUC(δ)": "AUC(δ)−AUC(0) 逐点损失",
                "PR-AUC20": "δ∈[−20,20] 平均 AUC (整体鲁棒性)",
                "PR-drop20": "AUC(0)−min(AUC), |δ|≤20 (最坏点损失)",
                "note": "专指 R 峰窗口偏移/部署链群延迟语义 (T1-3 δ*≈6), 与通用时序 S-Cons. 区分",
            },
            "data": "deploy_match 缓存测试拍 (beats_deploy, δ=0)",
        },
        "results": results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 结果已保存: {OUT_JSON} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    main()
