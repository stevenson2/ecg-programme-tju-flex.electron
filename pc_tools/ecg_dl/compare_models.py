#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型对比可视化工具

比较多个已训练模型的各项指标，生成：
  1. 分组柱状图 — Acc / AUC / Precision / Recall / F1
  2. 雷达图 — 综合能力覆盖
  3. 参数-性能散点图 — 容量 vs 效果

用法:
  python3 compare_models.py --models best_model.h5 best_resnet_lite.h5
  python3 compare_models.py --models *.h5
  python3 compare_models.py --auto  # 自动扫描 models/ 下所有 .h5
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import tensorflow as tf

from config import MODELS_DIR, CLASS_NAMES
from data.dataset import load_mit_incart_merged, load_3beat_merged, train_val_test_split, add_channel_dim

ABNORMAL_IDX = 1
ALL_METRICS = ["Acc", "AUC", "Precision", "Recall", "F1"]
ALL_METRICS_EN = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score"]


def model_name_from_path(path: Path):
    """Return (display_name, family, params_est)."""
    stem = path.stem
    mapping = {
        "best_model":            ("CNN-v2",      "CNN",  15),
        "final_cnn_v2":          ("CNN-v2",      "CNN",  15),
        "final_cnn_v1":          ("CNN-v1",      "CNN",   5),
        "final_cnn_v3":          ("CNN-v3",      "CNN",  30),
        "final_cnn_tiny":        ("CNN-Tiny",    "CNN",   3),
        "best_cnn_m_small":      ("CNN-M-S",     "CNN-M", 114),
        "final_cnn_m_small":     ("CNN-M-S",     "CNN-M", 114),
        "best_cnn_m":            ("CNN-M",       "CNN-M", 140),
        "final_cnn_m":           ("CNN-M",       "CNN-M", 140),
        "best_cnn_m_large":      ("CNN-M-L",     "CNN-M", 453),
        "final_cnn_m_large":     ("CNN-M-L",     "CNN-M", 453),
        "best_small":            ("CNN-v1",      "CNN",   5),
        "best_resnet_lite":      ("ResNet-S",    "ResNet", 25),
        "final_resnet_s":        ("ResNet-S",    "ResNet", 25),
        "best_resnet_medium":    ("ResNet-M",    "ResNet", 55),
        "final_resnet_m":        ("ResNet-M",    "ResNet", 55),
        "best_resnet_large":     ("ResNet-L",    "ResNet", 80),
        "final_resnet_l":        ("ResNet-L",    "ResNet", 80),
        "best_resnet_multitask": ("ResNet-MT",   "MultiTask", 68),
        "final_resnet_multitask":("ResNet-MT",   "MultiTask", 68),
        "final_resnet_multitask_fg":       ("ResNet-MT-FG", "MultiTask", 68),
        "final_resnet_multitask_freeze25":  ("ResNet-MT-f25","MultiTask", 68),
        "final_resnet_multitask_nofreeze":  ("ResNet-MT-f0", "MultiTask", 68),
    }
    for k, v in dict(sorted(mapping.items(), key=lambda x: -len(x[0]))).items():
        if stem == k or stem.startswith(k):
            return v
    return (stem.replace("_", " ").title(), "Unknown", 0)


# Paper-ready: model-family color scheme
FAMILY_COLORS = {"CNN": "#2E86AB", "CNN-M": "#A23B72", "ResNet": "#C73E1D", "MultiTask": "#E6AB02", "Baseline": "#666666", "Unknown": "#888888"}


def eval_one_model(model_path: Path, x_test, y_test, threshold=0.50):
    """Evaluate a single H5 model and return dict of metrics."""
    model = tf.keras.models.load_model(str(model_path), compile=False)

    x_in = add_channel_dim(x_test)
    raw = model.predict(x_in, verbose=0)

    # Handle multi-output models (ResNet-MT: [cls, bpm, sqi])
    if isinstance(raw, (list, tuple)):
        y_prob = raw[0]
    else:
        y_prob = raw

    # Handle single-class or 2-class output
    if y_prob.shape[-1] >= 2:
        prob_ab = y_prob[:, ABNORMAL_IDX]
        y_pred_hard = np.argmax(y_prob, axis=1)
    else:
        prob_ab = y_prob[:, 0]
        y_pred_hard = (prob_ab >= 0.5).astype(int)

    # Metrics at threshold 0.5
    tp_05 = int(((y_pred_hard == 1) & (y_test == 1)).sum())
    fp_05 = int(((y_pred_hard == 1) & (y_test == 0)).sum())
    fn_05 = int(((y_pred_hard == 0) & (y_test == 1)).sum())
    tn_05 = int(((y_pred_hard == 0) & (y_test == 0)).sum())

    acc = (tp_05 + tn_05) / len(y_test)
    prec = tp_05 / max(tp_05 + fp_05, 1)
    recall = tp_05 / max(tp_05 + fn_05, 1)
    f1 = 2 * prec * recall / max(prec + recall, 1e-8)

    # AUC (threshold-invariant)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_test, y_prob[:, ABNORMAL_IDX]))
    except ImportError:
        auc = 0.0

    # Metrics at tuned threshold
    y_pred_t = (prob_ab >= threshold).astype(int)
    tp_t = int(((y_pred_t == 1) & (y_test == 1)).sum())
    fp_t = int(((y_pred_t == 1) & (y_test == 0)).sum())
    fn_t = int(((y_pred_t == 0) & (y_test == 1)).sum())
    tn_t = int(((y_pred_t == 0) & (y_test == 0)).sum())

    tuned_acc = (tp_t + tn_t) / len(y_test)
    tuned_prec = tp_t / max(tp_t + fp_t, 1)
    tuned_recall = tp_t / max(tp_t + fn_t, 1)
    tuned_f1_val = 2 * tuned_prec * tuned_recall / max(tuned_prec + tuned_recall, 1e-8)

    n_params = int(model.count_params())
    name, family, _est = model_name_from_path(model_path)
    tf.keras.backend.clear_session()

    return {
        "name": name, "family": family,
        "file": str(model_path.name),
        "params": n_params,
        "n_samples": int(len(y_test)),
        "acc": acc, "prec": prec, "recall": recall,
        "auc": auc, "f1": f1,
        "tuned_acc": tuned_acc,
        "tuned_prec": tuned_prec,
        "tuned_recall": tuned_recall,
        "tuned_f1": tuned_f1_val,
    }


def plot_bar_chart(results, save_path):
    """Paper-ready grouped bar chart with CNN/ResNet color families."""
    # Sort: CNN first, then ResNet; within each, by params
    results = sorted(results, key=lambda r: (
        0 if r.get("family") == "CNN" else 1,
        r["params"]
    ))
    fig, ax = plt.subplots(figsize=(14, 6))

    names = [r["name"] for r in results]
    families = [r.get("family", "Unknown") for r in results]
    metrics = ["acc", "auc", "prec", "recall", "f1"]
    metric_labels = ["Accuracy", "AUC", "Precision", "Recall", "F1"]
    n_models = len(names)
    n_metrics = len(metrics)

    x = np.arange(n_models)
    width = 0.15
    bar_colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    for i, (met, label, bcolor) in enumerate(zip(metrics, metric_labels, bar_colors)):
        vals = [r[met] for r in results]
        bars = ax.bar(x + i * width - width * (n_metrics - 1) / 2,
                      vals, width, label=label, color=bcolor,
                      edgecolor="white", linewidth=0.8, alpha=0.9)

        for bar, val in zip(bars, vals):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{val:.1%}" if val > 0.1 else f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7,
                        fontweight="bold", color=bcolor)

    ax.set_xticks(x)
    # Color xtick labels by family
    xtick_colors = [FAMILY_COLORS.get(f, "#666") for f in families]
    for tick, color in zip(ax.get_xticklabels(), xtick_colors):
        tick.set_color(color)
        tick.set_fontweight("bold")
    ax.set_xticklabels(names, fontsize=10, rotation=0)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Model Performance Comparison (CNN vs ResNet)",
                 fontsize=15, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=9, ncol=n_metrics,
              framealpha=0.9, edgecolor="#ccc")
    ax.grid(axis="y", alpha=0.15, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Add family separator line between CNN and ResNet groups
    cnn_count = sum(1 for f in families if f == "CNN")
    if 0 < cnn_count < n_models:
        ax.axvline(x=cnn_count - 0.5, color="#ccc", linewidth=1.5,
                   linestyle="--", alpha=0.6)

    # Family legend patches
    from matplotlib.patches import Patch
    legend_patches = []
    for fam in ["CNN", "ResNet"]:
        if fam in families:
            legend_patches.append(Patch(color=FAMILY_COLORS[fam], alpha=0.3,
                                        label=fam))
    leg2 = ax.legend(handles=legend_patches, loc="upper left",
                     fontsize=10, framealpha=0.9, edgecolor="#ccc")
    ax.add_artist(leg2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[对比] 柱状图已保存: {save_path}")


def plot_radar_chart(results, save_path):
    """Paper-ready radar chart with CNN vs ResNet overlay."""
    metrics = ["acc", "auc", "prec", "recall", "f1"]
    metric_labels = ["Accuracy", "AUC", "Precision", "Recall", "F1"]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})

    for j, r in enumerate(results):
        fam = r.get("family", "Unknown")
        color = FAMILY_COLORS.get(fam, "#888")
        vals = [r[m] for m in metrics]
        vals += vals[:1]
        ax.fill(angles, vals, alpha=0.06, color=color)
        ax.plot(angles, vals, "o-", linewidth=2.5,
                label=f"{r['name']} ({fam})", color=color,
                markersize=6, markerfacecolor="white",
                markeredgewidth=2, markeredgecolor=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0.40, 1.02)
    ax.set_yticks([0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0])
    ax.set_yticklabels(["40%", "50%", "60%", "70%", "80%", "90%", "100%"],
                       fontsize=8, color="gray")
    ax.set_title("Holistic Model Comparison", fontsize=14,
                 fontweight="bold", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12),
              fontsize=9, framealpha=0.9, edgecolor="#ccc")
    ax.grid(True, alpha=0.2, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[对比] 雷达图已保存: {save_path}")


def plot_paper_figure(results, save_path):
    """Combined 3-panel publication figure."""
    results = sorted(results, key=lambda r: (
        0 if r.get("family") == "CNN" else 1, r["params"]
    ))
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 0.8], wspace=0.35)

    names = [r["name"] for r in results]
    families = [r.get("family", "Unknown") for r in results]

    # Panel A: Bar chart
    ax_bar = fig.add_subplot(gs[0, 0])
    metrics = ["acc", "auc", "prec", "recall", "f1"]
    metric_labels = ["Accuracy", "AUC", "Precision", "Recall", "F1"]
    bar_colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
    n_models = len(names)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.15

    for i, (met, label, bcolor) in enumerate(zip(metrics, metric_labels, bar_colors)):
        vals = [r[met] for r in results]
        ax_bar.bar(x + i * width - width * (n_metrics - 1) / 2,
                   vals, width, label=label, color=bcolor,
                   edgecolor="white", linewidth=0.5, alpha=0.9)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, fontsize=9, fontweight="bold")
    for tick, fam in zip(ax_bar.get_xticklabels(), families):
        tick.set_color(FAMILY_COLORS.get(fam, "#666"))
    ax_bar.set_ylim(0, 1.08)
    ax_bar.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax_bar.set_title("(a) Performance Metrics", fontsize=12, fontweight="bold")
    ax_bar.legend(loc="lower right", fontsize=7, ncol=3)
    ax_bar.grid(axis="y", alpha=0.15, linestyle="--")
    ax_bar.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    if 0 < sum(1 for f in families if f == "CNN") < n_models:
        cnn_count = sum(1 for f in families if f == "CNN")
        ax_bar.axvline(x=cnn_count - 0.5, color="#999", linewidth=1, linestyle="--")

    # Panel B: Radar chart
    ax_rad = fig.add_subplot(gs[0, 1], projection="polar")
    radar_metrics = ["acc", "auc", "prec", "recall", "f1"]
    radar_labels = ["Acc", "AUC", "Prec", "Recall", "F1"]
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    for r in results:
        fam = r.get("family", "Unknown")
        color = FAMILY_COLORS.get(fam, "#888")
        vals = [r[m] for m in radar_metrics] + [r[radar_metrics[0]]]
        ax_rad.plot(angles, vals, "o-", linewidth=2, label=r["name"],
                    color=color, markersize=4, markerfacecolor="white",
                    markeredgewidth=1.5, markeredgecolor=color)
        ax_rad.fill(angles, vals, alpha=0.05, color=color)
    ax_rad.set_xticks(angles[:-1])
    ax_rad.set_xticklabels(radar_labels, fontsize=9, fontweight="bold")
    ax_rad.set_ylim(0.65, 1.02)
    ax_rad.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    ax_rad.set_yticklabels(["70%", "75%", "80%", "85%", "90%", "95%", "100%"], fontsize=6, color="gray")
    ax_rad.set_title("(b) Radar Overview", fontsize=12, fontweight="bold", pad=20)
    ax_rad.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
    ax_rad.grid(True, alpha=0.2, linestyle="--")

    # Panel C: Params vs Recall scatter
    ax_sc = fig.add_subplot(gs[0, 2])
    for r in results:
        fam = r.get("family", "Unknown")
        color = FAMILY_COLORS.get(fam, "#888")
        px = r["params"] / 1000
        py = r["recall"]
        ax_sc.scatter(px, py, s=180, c=color, edgecolors="white",
                      linewidth=1.5, zorder=5, alpha=0.85)
        ax_sc.annotate(r["name"], (px, py), textcoords="offset points",
                       xytext=(6, 6), fontsize=7, fontweight="bold",
                       color=color)
    ax_sc.set_xlabel("Parameters (K)", fontsize=11, fontweight="bold")
    ax_sc.set_ylabel("Abnormal Recall", fontsize=11, fontweight="bold")
    ax_sc.set_title("(c) Params vs Recall", fontsize=12, fontweight="bold")
    ax_sc.grid(alpha=0.15, linestyle="--")
    ax_sc.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.suptitle("ECG Arrhythmia Detection — CNN vs ResNet Model Family Comparison",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[对比] 论文综合图已保存: {save_path}")


def print_summary_table(results):
    """Print a formatted summary table to console."""
    header = (f"{'Model':<22s} {'Params':>8s}  {'Acc':>7s}  "
              f"{'AUC':>7s}  {'Prec':>7s}  {'Recall':>7s}  {'F1':>7s}  "
              f"{'T-Recall':>9s}  {'T-Prec':>7s}  {'T-F1':>7s}")
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for r in sorted(results, key=lambda x: x["auc"], reverse=True):
        ps = f"{r['params']/1000:.0f}K" if r["params"] < 1e6 else f"{r['params']/1e6:.1f}M"
        print(f"{r['name']:<22s} {ps:>8s}  "
              f"{r['acc']:>6.2%}  {r['auc']:>7.4f}  "
              f"{r['prec']:>6.2%}  {r['recall']:>6.2%}  {r['f1']:>6.2%}  "
              f"{r['tuned_recall']:>8.2%}  {r['tuned_prec']:>6.2%}  {r['tuned_f1']:>6.2%}")

    print(sep)


def main():
    parser = argparse.ArgumentParser(description="ECG 模型对比可视化")
    parser.add_argument("--models", nargs="+", default=None,
                        help="模型文件列表 (相对于 models/ 目录)")
    parser.add_argument("--auto", action="store_true",
                        help="自动扫描 models/*.h5 下所有模型")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="调优阈值 (默认 0.35)")
    parser.add_argument("--json", type=str, default=None,
                        help="加载 JSON 结果文件 (跳过所有推理)")
    parser.add_argument("--baseline", type=str, default=None,
                        help="注入基线 JSON (与 --models 结果合并)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认 models/figures)")
    parser.add_argument("--incart", action="store_true", default=True,
                        help="使用 MIT-BIH + INCART 测试集")
    parser.add_argument("--3beat", dest="use_3beat", action="store_true",
                        help="使用 3-beat 测试数据 (配合 CNN-M 等 750 点模型)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Load results ----
    results = []
    # Only --json (full replace, no model eval)
    if hasattr(args, 'json') and args.json:
        results = json.loads(Path(args.json).read_text())
        print(f"[对比] 从 JSON 加载 {len(results)} 个模型结果")
    else:
        if args.auto or args.models is None:
            model_files = sorted(MODELS_DIR.glob("final_*.h5"))
            if not model_files:
                model_files = (sorted(MODELS_DIR.glob("best_*.h5"))
                             + sorted(MODELS_DIR.glob("*.h5")))
            # Filter out pretrained weights
            model_files = [f for f in model_files if "pretrained" not in f.name]
            print(f"[对比] 自动发现 {len(model_files)} 个模型: {[f.name for f in model_files]}")
        else:
            model_files = [MODELS_DIR / m for m in args.models
                           if (MODELS_DIR / m).exists()]
            missing = [m for m in args.models if not (MODELS_DIR / m).exists()]
            if missing:
                print(f"[对比] ⚠️ 跳过不存在的文件: {missing}")

        if model_files:
            if args.use_3beat:
                print("[对比] 加载测试集 (MIT-BIH + INCART, 3-beat)...")
                data = load_3beat_merged()
            else:
                print("[对比] 加载测试集 (MIT-BIH + INCART)...")
                data = load_mit_incart_merged()
            splits = train_val_test_split(
                data["beats"], data["labels"],
                record_ids=data.get("record_ids")
            )
            x_test, y_test = splits["test"]
            nN = int((y_test == 0).sum())
            nA = int((y_test == 1).sum())
            print(f"[对比] 测试集: {len(y_test)} 样本 (N={nN}, A={nA}, {nA/len(y_test)*100:.1f}%)")

            for mf in model_files:
                print(f"[对比] 评估 {mf.name}...")
                try:
                    r = eval_one_model(mf, x_test, y_test, threshold=args.threshold)
                    results.append(r)
                except (ValueError, tf.errors.InvalidArgumentError) as e:
                    msg = str(e)
                    if "shape" in msg.lower() or "incompatible" in msg.lower():
                        print(f"  [跳过] 输入形状不兼容: {mf.name}")
                    else:
                        print(f"  [跳过] 错误: {mf.name} — {msg[:80]}")
                except Exception as e:
                    print(f"  [跳过] 错误: {mf.name} — {str(e)[:80]}")

            json_path = out_dir / f"model_comparison_{ts}.json"
            json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            print(f"[对比] 结果已保存: {json_path}")

        # Inject baseline JSON alongside evaluated models
        if args.baseline:
            base = json.loads(Path(args.baseline).read_text())
            for entry in base:
                entry.setdefault("family", "Baseline")
                entry.setdefault("n_samples", results[0]["n_samples"] if results else 48914)
            results = base + results
            print(f"[对比] 注入 {len(base)} 个基线数据")

    # ---- Generate charts ----
    if len(results) < 1:
        print("[对比] ❌ 没有结果可展示")
        return

    plot_bar_chart(results, out_dir / f"compare_bars_{ts}.png")
    if len(results) >= 3:
        plot_radar_chart(results, out_dir / f"compare_radar_{ts}.png")
    plot_paper_figure(results, out_dir / f"compare_paper_{ts}.png")

    # ---- Console summary ----
    n_samples = results[0].get("n_samples", "?")
    print(f"\n{'='*70}")
    print(f"  调优阈值: {args.threshold:.2f}  |  测试集: {n_samples} 样本")
    print_summary_table(results)

    # Highlight best per metric
    best = {}
    for met in ["acc", "auc", "recall", "f1", "tuned_recall", "tuned_f1"]:
        best_model = max(results, key=lambda r: r[met])
        best[met] = best_model

    print(f"\n{'★'*30}")
    print(f"  Best Accuracy:      {best['acc']['name']} ({best['acc']['acc']:.2%})")
    print(f"  Best AUC:           {best['auc']['name']} ({best['auc']['auc']:.4f})")
    print(f"  Best Recall @0.5:   {best['recall']['name']} ({best['recall']['recall']:.2%})")
    print(f"  Best Tuned Recall:  {best['tuned_recall']['name']} ({best['tuned_recall']['tuned_recall']:.2%} @{args.threshold:.2f})")
    print(f"  Best Tuned F1:      {best['tuned_f1']['name']} ({best['tuned_f1']['tuned_f1']:.2%})")
    print(f"{'★'*30}")


if __name__ == "__main__":
    main()
