#!/usr/bin/env python3
"""
Plot SSL training history from CSV log.

Can be run DURING training to check progress, or after completion.

Usage:
  python plot_ssl_history.py                          # Plot ssl_stage1_history.csv
  python plot_ssl_history.py --csv path/to/log.csv    # Custom CSV
  python plot_ssl_history.py --watch 10               # Auto-refresh every 10s
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR


def plot_history(csv_path, output_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    epochs, losses = [], []
    with open(csv_path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                epochs.append(int(parts[0]))
                losses.append(float(parts[1]))

    if not epochs:
        print("[ERROR] CSV is empty")
        return

    best_idx = np.argmin(losses)
    best_loss = losses[best_idx]
    best_epoch = epochs[best_idx]
    final_loss = losses[-1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Linear loss
    ax = axes[0]
    ax.plot(epochs, losses, color="#2E86AB", linewidth=1.5, alpha=0.8)
    ax.scatter([best_epoch], [best_loss], color="#C73E1D", s=100, zorder=5,
               marker="*", edgecolors="white", linewidth=1)
    ax.axhline(y=best_loss, color="#C73E1D", linestyle="--", alpha=0.4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("NT-Xent Loss")
    ax.set_title(f"(a) SimCLR Training Loss ({len(epochs)} epochs)")
    ax.annotate(f"Best: {best_loss:.4f} (ep {best_epoch})",
                xy=(best_epoch, best_loss), xytext=(best_epoch + 3, best_loss + 0.05),
                fontsize=9, color="#C73E1D", fontweight="bold")
    ax.grid(alpha=0.15, linestyle="--")

    # Panel B: Log-scale loss
    ax = axes[1]
    ax.semilogy(epochs, losses, color="#2E86AB", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log scale)")
    ax.set_title("(b) Log-Scale View")
    ax.grid(alpha=0.15, linestyle="--")

    # Panel C: Recent loss (last 50 epochs or all if fewer)
    n_recent = min(50, len(epochs))
    recent_ep = epochs[-n_recent:]
    recent_loss = losses[-n_recent:]
    ax = axes[2]
    ax.plot(recent_ep, recent_loss, color="#2E86AB", linewidth=2.0, marker="o",
            markersize=3, markerfacecolor="white")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"(c) Recent {n_recent} Epochs (Final: {final_loss:.4f})")
    ax.grid(alpha=0.15, linestyle="--")

    # Add convergence trend line for recent window
    if n_recent >= 5:
        z = np.polyfit(recent_ep, recent_loss, 1)
        p = np.poly1d(z)
        ax.plot(recent_ep, p(recent_ep), color="#C73E1D", linestyle="--",
                linewidth=1, alpha=0.6,
                label=f"Trend: {z[0]:.2e}/epoch")
        ax.legend(fontsize=8)

    plt.suptitle(f"Phase 2C — SimCLR SSL Pre-training History\n"
                 f"Start Loss: {losses[0]:.4f}  →  Final: {final_loss:.4f}  "
                 f"(best: {best_loss:.4f} @ epoch {best_epoch})",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    if output_path is None:
        output_path = csv_path.with_suffix(".png")
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"[Plot] Saved: {output_path}")

    # Console summary
    print(f"\n  Epochs: {len(epochs)} | Start: {losses[0]:.4f} | "
          f"Best: {best_loss:.4f} @ ep{best_epoch} | Final: {final_loss:.4f}")
    if n_recent >= 10:
        delta = np.mean(losses[-10:]) - np.mean(losses[-20:-10])
        status = "▼ converging" if delta < 0 else "▲ plateau/stalling"
        print(f"  Last-10 avg: {np.mean(losses[-10:]):.4f}  "
              f"Δ vs prev-10: {delta:+.4f}  ({status})")


def main():
    parser = argparse.ArgumentParser(
        description="Plot SSL training history from CSV log")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV file path (default: models/ssl_stage1_history.csv)")
    parser.add_argument("--watch", type=int, default=0,
                        help="Watch mode: auto-refresh every N seconds")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path")

    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = MODELS_DIR / "ssl_stage1_history.csv"

    if args.watch > 0:
        print(f"[Watch] Refreshing every {args.watch}s. Ctrl+C to stop.\n")
        try:
            while True:
                plot_history(csv_path, args.output)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n[Watch] Stopped.")
    else:
        plot_history(csv_path, args.output)


if __name__ == "__main__":
    main()
