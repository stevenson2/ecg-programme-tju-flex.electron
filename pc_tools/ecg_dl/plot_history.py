#!/usr/bin/env python3
"""
Plot training history from Keras CSVLogger output.

Usage:
  python plot_history.py --csv models/pretrain_history.csv
  python plot_history.py --csv models/pretrain_history.csv --watch 30
  python plot_history.py --csv models/multitask_history.csv
"""

import sys, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR
FIGURES_DIR = MODELS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot(csv_path, output_path=None, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[ERROR] Not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    cols = df.columns.tolist()
    epochs = np.arange(1, len(df) + 1)

    loss_cols = [c for c in cols if 'loss' in c.lower()]
    metric_cols = [c for c in cols if c not in loss_cols and c != 'epoch'
                   and c not in ['learning_rate', 'lr']]

    n_rows = 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 5 * n_rows))

    # Row 1: Loss curves
    ax = axes[0]
    for c in loss_cols:
        vals = pd.to_numeric(df[c], errors='coerce').values
        ax.plot(epochs, vals, linewidth=1.5, label=c, alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss — {csv_path.name}")
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Row 2: Metrics
    ax = axes[1]
    for c in metric_cols:
        vals = pd.to_numeric(df[c], errors='coerce').values
        if vals.shape[-1] > 0 and not np.all(np.isnan(vals)):
            ax.plot(epochs, vals, linewidth=1.5, label=c, alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_title(f"Metrics — {csv_path.name}")
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if show:
        plt.draw()
        plt.pause(0.5)
    else:
        name = Path(csv_path).stem
        out = output_path or str(FIGURES_DIR / f"{name}.png")
        out_dir = Path(out).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close()


def watch(csv_path, interval=30):
    print(f"Watching {csv_path} (refresh every {interval}s)...")
    last_mtime = 0
    while True:
        p = Path(csv_path)
        if p.exists():
            mtime = p.stat().st_mtime
            if mtime > last_mtime:
                last_mtime = mtime
                name = p.stem
                plot(csv_path, str(FIGURES_DIR / f"{name}.png"))
                print(f"  Updated — {_count_rows(csv_path)} epochs")
        time.sleep(interval)


def _count_rows(csv_path):
    try:
        return sum(1 for _ in open(csv_path)) - 1
    except Exception:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--watch", type=int, default=0, help="Auto-refresh seconds")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    csv_path = args.csv
    if not Path(csv_path).is_absolute():
        csv_path = str(MODELS_DIR / csv_path) if not Path(args.csv).exists() else args.csv

    if args.watch > 0:
        watch(csv_path, args.watch)
    else:
        plot(csv_path, args.output)
