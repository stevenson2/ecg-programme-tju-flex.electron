#!/usr/bin/env python3
"""
ECG Model Figure Generator — Publication-Quality Comparison Plots

Model Registry: models/model_registry.json
Add new models by editing the JSON. Re-run to regenerateall figures.

Usage:
  python3 figures.py --all                    # Generate all figures
  python3 figures.py --bars                   # Bar chart only
  python3 figures.py --roc                    # ROC curves only
  python3 figures.py --history                # Training history only
  python3 figures.py --paper                  # Consolidated paper figure
  python3 figures.py --eval-only              # Re-evaluate models, update cache

Output: models/figures/
"""

import sys, os, json, argparse, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from config import MODELS_DIR, CLASS_NAMES
from data.dataset import (load_mit_incart_merged, load_3beat_merged,
                          train_val_test_split, add_channel_dim)

REGISTRY_PATH = MODELS_DIR / "model_registry.json"
CACHE_PATH = MODELS_DIR / "eval_cache.json"
FIGURES_DIR = MODELS_DIR / "figures"
ABNORMAL_IDX = 1


# ============================================================================
# Registry & Evaluation
# ============================================================================

def load_registry():
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def compute_metrics(y_true, prob_ab, threshold=0.50):
    y_pred = (prob_ab >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    acc = (tp + tn) / len(y_true)
    prec = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * prec * recall / max(prec + recall, 1e-8)
    try:
        auc = float(roc_auc_score(y_true, prob_ab))
    except Exception:
        auc = 0.0
    return {"acc": acc, "auc": auc, "prec": prec, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _predict_cls(model, x):
    """Extract classification probabilities, handling multi-output models."""
    raw = model.predict(x, verbose=0)
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    if raw.shape[-1] >= 2:
        return raw[:, ABNORMAL_IDX]
    return raw[:, 0]


def evaluate_model(entry, x_test, y_test):
    h5 = entry.get("h5")
    ensemble_h5s = entry.get("h5", None)
    if isinstance(ensemble_h5s, list):
        ensemble_h5s = ensemble_h5s

    if isinstance(h5, list):
        probs = []
        for path in h5:
            m = tf.keras.models.load_model(str(MODELS_DIR / path), compile=False)
            p = _predict_cls(m, add_channel_dim(x_test))
            probs.append(p)
            tf.keras.backend.clear_session()
        prob_ab = np.mean(probs, axis=0)
    elif h5:
        m = tf.keras.models.load_model(str(MODELS_DIR / h5), compile=False)
        prob_ab = _predict_cls(m, add_channel_dim(x_test))
        tf.keras.backend.clear_session()
    else:
        return None

    metrics = compute_metrics(y_test, prob_ab)
    tuned = compute_metrics(y_test, prob_ab, 0.35)
    metrics["tuned_recall"] = tuned["recall"]
    metrics["tuned_prec"] = tuned["prec"]
    metrics["tuned_f1"] = tuned["f1"]
    metrics["n_samples"] = len(y_test)
    return {"metrics": metrics, "prob_ab": prob_ab, "y_test": y_test}


def evaluate_all(registry, force=False):
    if CACHE_PATH.exists() and not force:
        cache = json.loads(CACHE_PATH.read_text())
        print(f"[Eval] Loaded cache ({len(cache)} models)")
        results = {}
        for mid, r in cache.items():
            results[mid] = r
            results[mid]["prob_ab"] = None
            results[mid]["y_test"] = None
        return results

    print("[Eval] Loading test data (MIT+INCART)...")
    data = load_mit_incart_merged()
    splits = train_val_test_split(data["beats"], data["labels"],
                                  record_ids=data.get("record_ids"))
    x_test_1beat, y_test = splits["test"]
    nN, nA = int((y_test == 0).sum()), int((y_test == 1).sum())
    print(f"[Eval] Test: {len(y_test)} samples (N={nN}, A={nA})")

    # Also load 3-beat data for CNN-M
    x_test_3beat, y_test_3beat = None, None
    try:
        data3 = load_3beat_merged()
        splits3 = train_val_test_split(data3["beats"], data3["labels"],
                                        record_ids=data3.get("record_ids"))
        x_test_3beat, y_test_3beat = splits3["test"]
        # Align indices if possible (same record split order)
    except Exception:
        x_test_3beat = None

    results = {}
    for entry in registry["models"]:
        mid = entry["id"]
        if entry.get("metrics") and not entry.get("h5"):
            results[mid] = {"metrics": entry["metrics"]}
            continue
        if not entry.get("h5"):
            continue
        print(f"[Eval] {entry['name']}...")

        is_3beat = entry.get("input_window") == 750
        x_test = x_test_3beat if is_3beat else x_test_1beat
        yt = y_test_3beat if is_3beat else y_test
        if is_3beat and x_test_3beat is None:
            print(f"  SKIP: no 3-beat test data")
            continue

        result = evaluate_model(entry, x_test, yt)
        if result:
            results[mid] = {"metrics": result["metrics"]}
            results[mid].update({
                "name": entry["name"],
                "family": entry["family"],
                "color": entry.get("color", "#666"),
                "phase": entry.get("phase", ""),
                "params": entry.get("params", 0),
            })
            results[mid]["prob_ab"] = result["prob_ab"]
            results[mid]["y_test"] = result["y_test"]

    CACHE_PATH.write_text(json.dumps(
        {mid: {k: v for k, v in r.items()
               if k not in ("prob_ab", "y_test")}
         for mid, r in results.items()}, indent=2))
    print(f"[Eval] Saved cache: {CACHE_PATH}")
    return results


def ensure_probs(results, registry):
    needed = [(mid, r) for mid, r in results.items()
              if r.get("prob_ab") is None and r.get("metrics")]
    if not needed:
        return
    print("[Probs] Re-evaluating for ROC data...")
    data = load_mit_incart_merged()
    splits = train_val_test_split(data["beats"], data["labels"],
                                  record_ids=data.get("record_ids"))
    x1, y1 = splits["test"]
    try:
        d3 = load_3beat_merged()
        s3 = train_val_test_split(d3["beats"], d3["labels"],
                                  record_ids=d3.get("record_ids"))
        x3, y3 = s3["test"]
    except Exception:
        x3, y3 = None, None
    for mid, r in needed:
        e = next((x for x in registry["models"] if x["id"] == mid), {})
        is3 = e.get("input_window") == 750
        xt, yt = (x3, y3) if is3 else (x1, y1)
        if is3 and x3 is None:
            continue
        res = evaluate_model(e, xt, yt)
        if res:
            r["prob_ab"] = res["prob_ab"]
            r["y_test"] = res["y_test"]


# ============================================================================
# Figure: Bar Chart
# ============================================================================

def plot_bar_chart(results, registry):
    entries = registry["models"]
    sorted_models = sorted(results.items(),
                           key=lambda kv: next((e.get("params", 0) for e in entries if e["id"] == kv[0]), 0))
    names = [r.get("name", mid) for mid, r in sorted_models]
    families = [r.get("family", "?") for r in [r for _, r in sorted_models]]
    colors = [r.get("color", "#666") for _, r in sorted_models]
    metrics_keys = ["acc", "auc", "prec", "recall", "f1"]
    labels = ["Accuracy", "AUC", "Precision", "Recall", "F1"]
    bar_colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(14, 6))
    n = len(names)
    x = np.arange(n)
    w = 0.14

    for i, (mk, label, bc) in enumerate(zip(metrics_keys, labels, bar_colors)):
        vals = [results[mid]["metrics"][mk] for mid, _ in sorted_models]
        bars = ax.bar(x + i * w - w * 2, vals, w, label=label, color=bc,
                      edgecolor="white", linewidth=0.5, alpha=0.9)
        for bar, val in zip(bars, vals):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.1%}", ha="center", va="bottom", fontsize=7,
                        fontweight="bold", color=bc)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, fontweight="bold", rotation=15, ha="right")
    for tick, color in zip(ax.get_xticklabels(), colors):
        tick.set_color(color)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("ECG Arrhythmia Detection — Model Performance Comparison",
                 fontsize=13, fontweight="bold", pad=12)
    # ax.legend(loc="upper center", fontsize=8, ncol=5, framealpha=0.9,
    #           bbox_to_anchor=(0.5, 1.15))
    ax.grid(axis="y", alpha=0.12, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Family separator lines
    unique_families = []
    last_idx = -1
    for i, (mid, _) in enumerate(sorted_models):
        fam = results[mid].get("family", "")
        if fam != (unique_families[-1] if unique_families else ""):
            if last_idx >= 0:
                ax.axvline(x=i - 0.5, color="#999", linewidth=1, linestyle="--", alpha=0.4)
            unique_families.append(fam)
            last_idx = i

    # Family legend
    # fam_patches = []
    # seen = set()
    # for mid, r in sorted_models:
    #     fam = r.get("family", "")
    #     if fam not in seen:
    #         seen.add(fam)
    #         fam_patches.append(Patch(color=r.get("color", "#666"), alpha=0.3, label=fam))
    # if len(fam_patches) > 1:
    #     leg2 = ax.legend(handles=fam_patches, loc="upper left", fontsize=9,
    #                      framealpha=0.9, title="Model Family")
    #     ax.add_artist(leg2)

    plt.tight_layout()
    path = FIGURES_DIR / "compare_bars.png"
    path.parent.mkdir(exist_ok=True)
    plt.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Figure] {path}")


# ============================================================================
# Figure: ROC Curves
# ============================================================================

def plot_roc_curves(results, registry):
    fig, ax = plt.subplots(figsize=(8, 7))

    for mid, r in results.items():
        entry = next((e for e in registry["models"] if e["id"] == mid), {})
        name = entry.get("name", mid)
        color = entry.get("color", "#666")
        auc_val = r["metrics"].get("auc", 0)

        if r.get("prob_ab") is None or r.get("y_test") is None:
            continue

        fpr, tpr, _ = roc_curve(r["y_test"], r["prob_ab"])
        ax.plot(fpr, tpr, linewidth=2.2, color=color,
                label=f"{name} (AUC={auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    ax.set_title("ROC Curves — ECG Arrhythmia Detection",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.12, linestyle="--")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    plt.tight_layout()
    path = FIGURES_DIR / "roc_curves.png"
    plt.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Figure] {path}")


# ============================================================================
# Figure: Radar Chart
# ============================================================================

def plot_radar_chart(results, registry):
    metrics = ["acc", "auc", "prec", "recall", "f1"]
    labels = ["Accuracy", "AUC", "Precision", "Recall", "F1"]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})

    for mid, r in results.items():
        entry = next((e for e in registry["models"] if e["id"] == mid), {})
        name = entry.get("name", mid)
        color = entry.get("color", "#666")
        vals = [r["metrics"][m] for m in metrics]
        vals += vals[:1]
        ax.fill(angles, vals, alpha=0.06, color=color)
        ax.plot(angles, vals, "o-", linewidth=2.5, label=name, color=color,
                markersize=6, markerfacecolor="white", markeredgewidth=2,
                markeredgecolor=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0.60, 1.02)
    ax.set_yticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    ax.set_yticklabels(["65%", "70%", "75%", "80%", "85%", "90%", "95%", "100%"],
                       fontsize=8, color="gray")
    ax.set_title("Holistic Model Comparison", fontsize=13, fontweight="bold", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=9,
              framealpha=0.9)
    ax.grid(True, alpha=0.15, linestyle="--")

    plt.tight_layout()
    path = FIGURES_DIR / "compare_radar.png"
    plt.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Figure] {path}")


# ============================================================================
# Figure: Training History
# ============================================================================

def plot_training_history(results, registry):
    ssl_stage1_csv = MODELS_DIR / "ssl_stage1_history.csv"
    ssl_stage2_csv = MODELS_DIR / "ssl_ft_history.csv"

    has_stage1 = ssl_stage1_csv.exists()
    has_stage2 = ssl_stage2_csv.exists()

    if not has_stage1 and not has_stage2:
        print("[Figure] No training history CSVs found, skipping")
        return

    n_panels = (1 if has_stage1 else 0) + (1 if has_stage2 else 0)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    if has_stage1:
        ax = axes[panel_idx]
        panel_idx += 1
        with open(ssl_stage1_csv) as f:
            lines = f.readlines()[1:]
        epochs, losses = [], []
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                epochs.append(int(parts[0]))
                losses.append(float(parts[1]))
        color = "#1B9E77"
        ax.plot(epochs, losses, color=color, linewidth=2)
        ax.fill_between(epochs, losses, alpha=0.1, color=color)
        best_idx = np.argmin(losses)
        ax.scatter([epochs[best_idx]], [losses[best_idx]], color=color, s=80,
                   zorder=5, marker="*", edgecolors="white", linewidth=1)
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("NT-Xent Loss", fontsize=11, fontweight="bold")
        ax.set_title("(a) SimCLR SSL Pre-training", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.12, linestyle="--")
        ax.annotate(f"Best: {losses[best_idx]:.4f}",
                    xy=(epochs[best_idx], losses[best_idx]),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=9, color=color, fontweight="bold")

    if has_stage2:
        ax = axes[panel_idx]
        with open(ssl_stage2_csv) as f:
            all_lines = f.readlines()
        header = all_lines[0].strip().split(",")
        data_lines = all_lines[1:]
        data = {}
        for line in data_lines:
            parts = line.strip().split(",")
            for i, val in enumerate(parts):
                if i < len(header):
                    key = header[i]
                    data.setdefault(key, []).append(float(val))

        epochs = list(range(1, len(data.get("loss", [])) + 1))
        color_loss, color_acc, color_auc = "#C73E1D", "#2E86AB", "#1B9E77"

        ax2_loss = ax.twinx()
        ax.plot(epochs, data.get("loss", []), color=color_loss, linewidth=2,
                label="Loss")
        ax2_loss.plot(epochs, data.get("val_loss", []), color=color_loss,
                      linewidth=1.5, linestyle="--", alpha=0.5, label="Val Loss")
        ax2_loss.plot(epochs, data.get("val_auc", []), color=color_auc,
                      linewidth=2, label="Val AUC")
        ax.plot(epochs, data.get("val_accuracy", []), color=color_acc,
                linewidth=2, label="Val Acc")

        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("Loss / Accuracy", fontsize=11, fontweight="bold")
        ax2_loss.set_ylabel("AUC", fontsize=11, fontweight="bold", color=color_auc)
        ax.set_title("(b) SSL Fine-tuning (MIT+INCART)", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.12, linestyle="--")

        lines_a, labels_a = ax.get_legend_handles_labels()
        lines_b, labels_b = ax2_loss.get_legend_handles_labels()
        ax.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8, loc="center right")

    plt.suptitle("Phase 2C — SSL Training History", fontsize=14, fontweight="bold",
                 y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "training_history.png"
    plt.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Figure] {path}")


# ============================================================================
# Figure: Confusion Matrices
# ============================================================================

def plot_confusion_matrices(results, registry):
    models_with_probs = [(mid, r) for mid, r in results.items()
                         if r.get("prob_ab") is not None]
    if not models_with_probs:
        print("[Figure] No models with prediction data, skipping CM")
        return

    n = len(models_with_probs)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if rows * cols == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for idx, (mid, r) in enumerate(models_with_probs):
        ax = axes[idx // cols][idx % cols]
        entry = next((e for e in registry["models"] if e["id"] == mid), {})
        name = entry.get("name", mid)
        y_pred = (r["prob_ab"] >= 0.5).astype(int)
        cm = confusion_matrix(r["y_test"], y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                text = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Abnormal"], fontsize=10)
        ax.set_yticklabels(["Normal", "Abnormal"], fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=10, fontweight="bold")
        acc = r["metrics"]["acc"]
        ax.set_title(f"{name} (Acc={acc:.1%})", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    plt.suptitle("Confusion Matrices (Threshold = 0.50)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "confusion_matrices.png"
    plt.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Figure] {path}")


# ============================================================================
# Figure: Parameters vs Performance Scatter
# ============================================================================

def plot_params_vs_perf(results, registry):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    for mid, r in results.items():
        entry = next((e for e in registry["models"] if e["id"] == mid), {})
        name = entry.get("name", mid)
        color = entry.get("color", "#666")
        params = entry.get("params", 0) / 1000
        recall = r["metrics"]["recall"]
        auc = r["metrics"]["auc"]
        acc = r["metrics"]["acc"]
        marker = entry.get("marker", "o")

        ax1.scatter(params, recall, s=200, c=color, edgecolors="white",
                    linewidth=1.5, zorder=5, alpha=0.85,
                    marker=marker, label=f"{name} ({params:.0f}K)")
        ax2.scatter(params, auc, s=200, c=color, edgecolors="white",
                    linewidth=1.5, zorder=5, alpha=0.85,
                    marker=marker, label=f"{name} ({params:.0f}K)")
        ax3.scatter(params, acc, s=200, c=color, edgecolors="white",
                    linewidth=1.5, zorder=5, alpha=0.85,
                    marker=marker, label=f"{name} ({params:.0f}K)")

    ax1.set_xlabel("Parameters (K)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Abnormal Recall", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Params vs Recall", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=7, framealpha=0.9, loc="lower right")
    ax1.grid(alpha=0.12, linestyle="--")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    ax2.set_xlabel("Parameters (K)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("AUC", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Params vs AUC", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=7, framealpha=0.9, loc="lower right")
    ax2.grid(alpha=0.12, linestyle="--")

    ax3.set_xlabel("Parameters (K)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    ax3.set_title("(c) Params vs Accuracy", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=7, framealpha=0.9, loc="lower right")
    ax3.grid(alpha=0.12, linestyle="--")
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.suptitle("Model Efficiency — Parameters vs Performance",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "params_vs_perf.png"
    plt.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Figure] {path}")


# ============================================================================
# Figure: Recall vs Threshold
# ============================================================================

def plot_recall_vs_threshold(results, registry):
    models_with_probs = [(mid, r) for mid, r in results.items()
                         if r.get("prob_ab") is not None]
    if not models_with_probs:
        print("[Figure] No models with probs, skipping recall-threshold")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    for mid, r in models_with_probs:
        entry = next((e for e in registry["models"] if e["id"] == mid), {})
        name, color = entry.get("name", mid), entry.get("color", "#666")
        thresholds = np.arange(0.05, 0.96, 0.02)
        recalls = []
        for th in thresholds:
            m = compute_metrics(r["y_test"], r["prob_ab"], th)
            recalls.append(m["recall"])
        ax.plot(thresholds, recalls, linewidth=2, color=color, label=name)
    ax.set_xlabel("Threshold", fontsize=11, fontweight="bold")
    ax.set_ylabel("Abnormal Recall", fontsize=11, fontweight="bold")
    ax.set_title("(a) Recall vs Threshold", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.12, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.axvline(x=0.35, color="gray", linestyle="--", alpha=0.5)
    ax.annotate("deploy θ=0.35", xy=(0.35, 0.98), xycoords=("data", "axes fraction"),
                fontsize=7, color="gray", ha="center")
    ax.legend(fontsize=8, framealpha=0.9, loc="lower left")

    ax = axes[1]
    for mid, r in models_with_probs:
        entry = next((e for e in registry["models"] if e["id"] == mid), {})
        best_f1, best_th = 0, 0.5
        for th in np.arange(0.10, 0.90, 0.01):
            m = compute_metrics(r["y_test"], r["prob_ab"], th)
            if m["f1"] > best_f1:
                best_f1, best_th = m["f1"], th
        m = compute_metrics(r["y_test"], r["prob_ab"], best_th)
        name, color = entry.get("name", mid), entry.get("color", "#666")
        marker = entry.get("marker", "o")
        label = f"{name} (θ={best_th:.2f})"
        ax.scatter(m["prec"], m["recall"], s=140, c=color,
                   edgecolors="white", linewidth=1.5, zorder=5,
                   marker=marker, label=label)
    ax.set_xlabel("Precision", fontsize=11, fontweight="bold")
    ax.set_ylabel("Recall", fontsize=11, fontweight="bold")
    ax.set_title("(b) Best F1 Operating Points", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.9, edgecolor="#ccc",
              loc="lower left", ncol=2)
    ax.grid(alpha=0.12, linestyle="--")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    plt.suptitle("Threshold Analysis — Abnormal Recall vs Precision Trade-off",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "recall_vs_threshold.png"
    plt.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Figure] {path}")


# ============================================================================
# Figure: Consolidated Paper Figure (4-panel)
# ============================================================================

def plot_paper_figure(results, registry):
    entries = registry["models"]
    sorted_ids = sorted(results.keys(),
                        key=lambda mid: next((e.get("params", 0) for e in entries if e["id"] == mid), 0))
    names = [results[mid].get("name", mid) for mid in sorted_ids]
    colors = [results[mid].get("color", "#666") for mid in sorted_ids]

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.32)

    # (a) Bar chart
    ax_bar = fig.add_subplot(gs[0, 0])
    metrics_keys = ["acc", "auc", "prec", "recall", "f1"]
    mlabels = ["Acc", "AUC", "Prec", "Recall", "F1"]
    mcolors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
    n = len(names)
    x = np.arange(n)
    w = 0.14
    for i, (mk, lb, bc) in enumerate(zip(metrics_keys, mlabels, mcolors)):
        vals = [results[mid]["metrics"][mk] for mid in sorted_ids]
        ax_bar.bar(x + i * w - w * 2, vals, w, label=lb, color=bc,
                   edgecolor="white", linewidth=0.5, alpha=0.9)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, fontsize=8, fontweight="bold", rotation=15, ha="right")
    for tick, c in zip(ax_bar.get_xticklabels(), colors):
        tick.set_color(c)
    ax_bar.set_ylim(0, 1.08)
    ax_bar.set_ylabel("Score", fontsize=10, fontweight="bold")
    ax_bar.set_title("(a) Performance Metrics", fontsize=12, fontweight="bold")
    ax_bar.grid(axis="y", alpha=0.12, linestyle="--")
    ax_bar.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # (b) ROC curves
    ax_roc = fig.add_subplot(gs[0, 1])
    for mid in sorted_ids:
        r = results[mid]
        if r.get("prob_ab") is None:
            continue
        entry = next((e for e in entries if e["id"] == mid), {})
        fpr, tpr, _ = roc_curve(r["y_test"], r["prob_ab"])
        ax_roc.plot(fpr, tpr, linewidth=2.2, color=entry.get("color", "#666"),
                    label=f"{entry.get('name', mid)} ({r['metrics']['auc']:.4f})")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.3)
    ax_roc.set_xlabel("FPR", fontsize=10, fontweight="bold")
    ax_roc.set_ylabel("TPR", fontsize=10, fontweight="bold")
    ax_roc.set_title("(b) ROC Curves", fontsize=12, fontweight="bold")
    ax_roc.legend(fontsize=7, framealpha=0.9)
    ax_roc.grid(alpha=0.12, linestyle="--")

    # (c) Recalls at different thresholds
    ax_rec = fig.add_subplot(gs[1, 0])
    thresholds = [0.30, 0.35, 0.40, 0.50]
    x_pos = np.arange(len(names))
    width = 0.18
    for ti, th in enumerate(thresholds):
        recalls = []
        for mid in sorted_ids:
            if results[mid].get("prob_ab") is not None:
                m = compute_metrics(results[mid]["y_test"], results[mid]["prob_ab"], th)
                recalls.append(m["recall"])
            else:
                recalls.append(0)
        bars = ax_rec.bar(x_pos + ti * width - width * 1.5, recalls, width,
                          label=f"θ={th:.2f}", edgecolor="white", linewidth=0.5,
                          alpha=0.85, color=plt.cm.viridis(ti / len(thresholds)))
    ax_rec.set_xticks(x_pos)
    ax_rec.set_xticklabels(names, fontsize=8, fontweight="bold", rotation=15, ha="right")
    for tick, c in zip(ax_rec.get_xticklabels(), colors):
        tick.set_color(c)
    ax_rec.set_ylabel("Abnormal Recall", fontsize=10, fontweight="bold")
    ax_rec.set_title("(c) Recall vs Threshold", fontsize=12, fontweight="bold")
    ax_rec.grid(axis="y", alpha=0.12, linestyle="--")
    ax_rec.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # (d) Params vs Performance
    ax_sc = fig.add_subplot(gs[1, 1])
    for mid in sorted_ids:
        entry = next((e for e in entries if e["id"] == mid), {})
        params_k = entry.get("params", 0) / 1000
        recall = results[mid]["metrics"]["recall"]
        c = entry.get("color", "#666")
        name = entry.get("name", mid)
        marker = entry.get("marker", "o")
        ax_sc.scatter(params_k, recall, s=200, c=c, edgecolors="white",
                      linewidth=1.5, zorder=5, alpha=0.85,
                      marker=marker, label=f"{name} ({params_k:.0f}K)")
    ax_sc.set_xlabel("Parameters (K)", fontsize=10, fontweight="bold")
    ax_sc.set_ylabel("Recall @0.5", fontsize=10, fontweight="bold")
    ax_sc.set_title("(d) Params vs Recall", fontsize=12, fontweight="bold")
    ax_sc.legend(fontsize=7, framealpha=0.9, loc="lower right")
    ax_sc.grid(alpha=0.12, linestyle="--")
    ax_sc.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.suptitle("ECG Arrhythmia Detection — Comprehensive Model Comparison",
                 fontsize=15, fontweight="bold", y=1.01)
    path = FIGURES_DIR / "paper_figure.png"
    plt.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Figure] {path}")


# ============================================================================
# Figure: Summary Table (text output + LaTeX)
# ============================================================================

def print_summary_table(results, registry):
    entries = {e["id"]: e for e in registry["models"]}
    print(f"\n{'='*85}")
    print(f"  {'Model':<18} {'Params':>7}  {'Acc':>7}  {'AUC':>7}  "
          f"{'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'Phase'}")
    print(f"  {'-'*85}")

    for mid, r in sorted(results.items(), key=lambda kv: kv[1]["metrics"]["auc"], reverse=True):
        m = r["metrics"]
        entry = entries.get(mid, {})
        params = f"{entry.get('params', 0)/1000:.0f}K"
        print(f"  {r.get('name', mid):<18} {params:>7}  {m['acc']:>6.2%}  "
              f"{m['auc']:>7.4f}  {m['prec']:>6.2%}  {m['recall']:>6.2%}  "
              f"{m['f1']:>6.2%}  {entry.get('phase', '')}")
    print(f"  {'-'*85}")

    # LaTeX table
    latex_path = FIGURES_DIR / "model_table.tex"
    lines = [r"\begin{table}[htbp]",
             r"  \centering",
             r"  \caption{ECG Arrhythmia Detection — Model Performance Comparison}",
             r"  \label{tab:model_comparison}",
             r"  \begin{tabular}{lcccccc}",
             r"    \toprule",
             r"    Model & Params & Acc & AUC & Prec & Recall & F1 \\",
             r"    \midrule"]
    for mid, r in sorted(results.items(), key=lambda kv: kv[1]["metrics"]["auc"], reverse=True):
        m = r["metrics"]
        entry = entries.get(mid, {})
        params = f"{entry.get('params', 0)/1000:.0f}K"
        lines.append(f"    {r.get('name', mid)} & {params} & "
                     f"{m['acc']:.1%} & {m['auc']:.4f} & {m['prec']:.1%} & "
                     f"{m['recall']:.1%} & {m['f1']:.2f} \\\\")
    lines.extend([r"    \bottomrule",
                  r"  \end{tabular}",
                  r"\end{table}"])
    latex_path.write_text("\n".join(lines) + "\n")
    print(f"\n[Table] LaTeX: {latex_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ECG Model Figure Generator")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument("--bars", action="store_true", help="Bar chart")
    parser.add_argument("--roc", action="store_true", help="ROC curves")
    parser.add_argument("--radar", action="store_true", help="Radar chart")
    parser.add_argument("--history", action="store_true", help="Training history")
    parser.add_argument("--confusion", action="store_true", help="Confusion matrices")
    parser.add_argument("--params", action="store_true", help="Params vs Performance")
    parser.add_argument("--paper", action="store_true", help="Consolidated paper figure")
    parser.add_argument("--threshold", action="store_true", help="Recall vs Threshold")
    parser.add_argument("--table", action="store_true", help="Summary table + LaTeX")
    parser.add_argument("--eval-only", action="store_true", help="Re-evaluate & cache only")
    parser.add_argument("--force-eval", action="store_true", help="Force re-evaluation")

    args = parser.parse_args()
    run_all = args.all or not any([args.bars, args.roc, args.radar, args.history,
                                    args.confusion, args.params, args.paper, args.table,
                                    args.threshold, args.eval_only])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    registry = load_registry()
    print(f"[Registry] {len(registry['models'])} models loaded")

    results = evaluate_all(registry, force=args.force_eval)

    if args.eval_only:
        print("[Eval] Done. Run with --all to generate figures.")
        return

    if run_all or args.roc or args.confusion or args.threshold or args.paper:
        ensure_probs(results, registry)

    if run_all or args.bars:
        plot_bar_chart(results, registry)
    if run_all or args.roc:
        plot_roc_curves(results, registry)
    if run_all or args.radar:
        plot_radar_chart(results, registry)
    if run_all or args.history:
        plot_training_history(results, registry)
    if run_all or args.confusion:
        plot_confusion_matrices(results, registry)
    if run_all or args.params:
        plot_params_vs_perf(results, registry)
    if run_all or args.threshold:
        plot_recall_vs_threshold(results, registry)
    if run_all or args.paper:
        plot_paper_figure(results, registry)

    if run_all or args.table:
        print_summary_table(results, registry)

    print(f"\n[DONE] Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
