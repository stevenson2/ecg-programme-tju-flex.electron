#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2B: 单拍 → 3拍序列拼接预处理

从现有 MIT-BIH + INCART 预处理 .npz 文件重构 3-beat 序列。
每个序列 = [前拍(250), 当前拍(250), 后拍(250)] = (750,)。
标签 = 当前拍 (中心拍) 的标签。

原始 .npz 中 beats 按 record 分组且保持时间序，
无需重跑全量预处理。
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DIR, CLASS_NAMES, BEAT_WINDOW_SAMPLES

BEAT_LEN = BEAT_WINDOW_SAMPLES  # 250
THREE_BEAT_LEN = BEAT_LEN * 3   # 750


def stitch_3beat(beats, labels, record_ids):
    """
    Stitch consecutive beats into 3-beat sequences within each record.

    Args:
        beats:      (N, 250) — single-beat windows
        labels:     (N,)    — binary labels
        record_ids: (N,)    — record ID per beat

    Returns:
        beats_3:  (M, 750) — 3-beat sequences (M = N - 2×num_records at boundaries)
        labels_3: (M,)     — labels of the center beat
        record_3: (M,)     — record IDs
    """
    unique_recs = np.unique(record_ids)
    beats_list, labels_list, recs_list = [], [], []

    total_beats = len(beats)
    skipped_boundary = 0

    for rec in unique_recs:
        mask = record_ids == rec
        idx = np.where(mask)[0]

        rec_beats = beats[idx]    # (k, 250)
        rec_labels = labels[idx]  # (k,)
        k = len(rec_beats)

        if k < 3:
            skipped_boundary += k
            continue

        for i in range(1, k - 1):
            prev_b = rec_beats[i - 1]
            curr_b = rec_beats[i]
            next_b = rec_beats[i + 1]
            triple = np.concatenate([prev_b, curr_b, next_b], axis=0)  # (750,)
            beats_list.append(triple)
            labels_list.append(rec_labels[i])
            recs_list.append(rec)

    beats_3 = np.array(beats_list, dtype=np.float32)
    labels_3 = np.array(labels_list, dtype=np.int32)
    record_3 = np.array(recs_list, dtype=np.int32)

    print(f"[3-beat] 输入: {total_beats} 单拍 → 输出: {len(beats_3)} 3拍序列")
    if skipped_boundary:
        print(f"[3-beat]   跳过边界: {skipped_boundary} 拍 "
              f"(记录首尾各1拍无法构成三拍序列)")
    return beats_3, labels_3, record_3


def preprocess_3beat_mit_bih():
    """Stitch MIT-BIH 3-beat sequences."""
    npz = PROCESSED_DIR / "mit_bih_processed.npz"
    if not npz.exists():
        raise FileNotFoundError(f"请先运行 python data/preprocess.py → {npz}")

    data = np.load(npz)
    beats_3, labels_3, record_3 = stitch_3beat(
        data["beats"], data["labels"], data["record_ids"]
    )

    out = PROCESSED_DIR / "mit_bih_3beat.npz"
    np.savez_compressed(out, beats=beats_3, labels=labels_3, record_ids=record_3)
    print(f"[3-beat] MIT-BIH 已保存: {out}")
    print_stats(labels_3)
    return out


def preprocess_3beat_incart():
    """Stitch INCART 3-beat sequences."""
    npz = PROCESSED_DIR / "incart_processed.npz"
    if not npz.exists():
        raise FileNotFoundError(f"请先运行 python data/preprocess_incart.py → {npz}")

    data = np.load(npz)
    beats_3, labels_3, record_3 = stitch_3beat(
        data["beats"], data["labels"], data["record_ids"]
    )

    out = PROCESSED_DIR / "incart_3beat.npz"
    np.savez_compressed(out, beats=beats_3, labels=labels_3, record_ids=record_3)
    print(f"[3-beat] INCART 已保存: {out}")
    print_stats(labels_3)
    return out


def preprocess_3beat_merged():
    """Merge MIT-BIH + INCART 3-beat into one file."""
    mit_path = PROCESSED_DIR / "mit_bih_3beat.npz"
    inc_path = PROCESSED_DIR / "incart_3beat.npz"

    for p in [mit_path, inc_path]:
        if not p.exists():
            raise FileNotFoundError(f"请先运行 3-beat 预处理: {p}")

    mit = np.load(mit_path)
    inc = np.load(inc_path)

    # Offset INCART record IDs to avoid collision
    inc_rids = inc["record_ids"] + 100000

    beats = np.concatenate([mit["beats"], inc["beats"]], axis=0)
    labels = np.concatenate([mit["labels"], inc["labels"]], axis=0)
    rids = np.concatenate([mit["record_ids"], inc_rids], axis=0)

    out = PROCESSED_DIR / "mit_incart_3beat.npz"
    np.savez_compressed(out, beats=beats, labels=labels, record_ids=rids)
    print(f"[3-beat] 合并已保存: {out}")
    print_stats(labels)
    return out


def print_stats(labels):
    nN = int((labels == 0).sum())
    nA = int((labels == 1).sum())
    print(f"  Total: {len(labels)}, Normal: {nN} ({nN/len(labels)*100:.1f}%), "
          f"Abnormal: {nA} ({nA/len(labels)*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="3-beat 序列预处理")
    parser.add_argument("--all", action="store_true", default=True,
                        help="处理 MIT-BIH + INCART 并合并")
    parser.add_argument("--mit", action="store_true", help="仅 MIT-BIH")
    parser.add_argument("--incart", action="store_true", help="仅 INCART")
    args = parser.parse_args()

    if args.mit:
        preprocess_3beat_mit_bih()
    if args.incart:
        preprocess_3beat_incart()
    if args.all or (not args.mit and not args.incart):
        preprocess_3beat_mit_bih()
        preprocess_3beat_incart()
        preprocess_3beat_merged()

    print("[3-beat] ✅ Done")
