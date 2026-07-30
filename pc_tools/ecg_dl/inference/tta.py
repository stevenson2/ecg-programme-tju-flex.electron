#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Route G: Test-Time Augmentation (TTA) for ECG beat inference.

Strategies:
  1. Sliding window: 3 overlapping views per beat (stride=0.25s), max-aggregation
  2. Mild augmentations: noise + amplitude scaling, averaged predictions
  3. Multi-beat confirmation: N consecutive abnormal beats required

Reference:
  - PTB-XL Benchmark: TTA sliding window + max aggregation
  - Nature SCD: TTA sliding window inference for streaming data
"""

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# TTA: Sliding Window (3-view, stride=0.25s)
# ---------------------------------------------------------------------------

def tta_sliding_window(beat, stride_samples=62, n_views=3):
    """
    Create multiple overlapping views of a single beat.

    For a 250-sample beat (1s at 250Hz), stride=62 ~ 0.25s.
    Pads edges by reflecting to keep output size consistent.

    Args:
        beat: (250,) or (250, 1) float32 array.
        stride_samples: Stride in samples (default 62 for 0.25s).
        n_views: Number of windows (default 3).

    Returns:
        ndarray (n_views, 250, 1) float32.
    """
    beat = np.asarray(beat, dtype=np.float32)
    if beat.ndim == 1:
        beat = beat.reshape(-1, 1)
    win = beat.shape[0]
    pad_left = stride_samples * (n_views // 2)
    pad_right = stride_samples * (n_views // 2)
    padded = np.pad(beat[:, 0], (pad_left, pad_right), mode='reflect')
    views = np.zeros((n_views, win, 1), dtype=np.float32)
    for i in range(n_views):
        start = i * stride_samples
        views[i, :, 0] = padded[start:start + win]
    return views


def tta_predict_sliding(model, beat, stride_samples=62, n_views=3,
                         aggregation='max'):
    views = tta_sliding_window(beat, stride_samples=stride_samples,
                               n_views=n_views)
    probs = model.predict(views, verbose=0)
    if probs.shape[-1] >= 2:
        abnormal_probs = probs[:, 1]
    else:
        abnormal_probs = probs[:, 0]

    if aggregation == 'max':
        return float(np.max(abnormal_probs))
    else:
        return float(np.mean(abnormal_probs))


def tta_predict_batch_sliding(model, beats, stride_samples=62, n_views=3,
                                aggregation='max', batch_size=32):
    beats = np.asarray(beats, dtype=np.float32)
    if beats.ndim == 2:
        beats = beats[..., np.newaxis]

    n_beats = len(beats)
    all_views = np.zeros((n_beats * n_views, beats.shape[1], 1),
                         dtype=np.float32)
    for i in range(n_beats):
        all_views[i * n_views:(i + 1) * n_views] = tta_sliding_window(
            beats[i, :, 0], stride_samples=stride_samples, n_views=n_views)

    probs = model.predict(all_views, batch_size=batch_size, verbose=0)
    if probs.shape[-1] >= 2:
        abnormal_probs = probs[:, 1]
    else:
        abnormal_probs = probs[:, 0]

    result = np.zeros(n_beats, dtype=np.float32)
    for i in range(n_beats):
        win_probs = abnormal_probs[i * n_views:(i + 1) * n_views]
        if aggregation == 'max':
            result[i] = np.max(win_probs)
        else:
            result[i] = np.mean(win_probs)
    return result


# ---------------------------------------------------------------------------
# TTA: Mild Augmentation (view augmentation)
# ---------------------------------------------------------------------------

def tta_augmented_views(beat, n_aug=5, noise_std=0.01, amp_range=0.05):
    beat = np.asarray(beat, dtype=np.float32)
    if beat.ndim == 1:
        beat = beat.reshape(1, -1, 1)
    elif beat.ndim == 2 and beat.shape[-1] != 1:
        beat = beat[np.newaxis, ...]
    if beat.ndim == 2:
        beat = beat[np.newaxis, ...]

    views = np.tile(beat, (n_aug + 1, 1, 1))
    rng = np.random.RandomState(42)
    for i in range(1, n_aug + 1):
        noise = rng.randn(*beat.shape[1:]).astype(np.float32) * noise_std
        scale = 1.0 + rng.uniform(-amp_range, amp_range)
        views[i] = views[i] * scale + noise
    return views


def tta_predict_augmented(model, beat, n_aug=5, noise_std=0.01,
                           amp_range=0.05, aggregation='mean'):
    views = tta_augmented_views(beat, n_aug=n_aug, noise_std=noise_std,
                                 amp_range=amp_range)
    probs = model.predict(views, verbose=0)
    if probs.shape[-1] >= 2:
        abnormal_probs = probs[:, 1]
    else:
        abnormal_probs = probs[:, 0]
    if aggregation == 'max':
        return float(np.max(abnormal_probs))
    else:
        return float(np.mean(abnormal_probs))


def tta_predict_batch_augmented(model, beats, n_aug=5, noise_std=0.01,
                                  amp_range=0.05, aggregation='mean',
                                  batch_size=64):
    beats = np.asarray(beats, dtype=np.float32)
    if beats.ndim == 2:
        beats = beats[..., np.newaxis]
    n_beats = len(beats)
    views_all = np.zeros((n_beats * (n_aug + 1), beats.shape[1], 1),
                         dtype=np.float32)
    rng = np.random.RandomState(42)
    for i in range(n_beats):
        base = i * (n_aug + 1)
        views_all[base] = beats[i]
        for j in range(1, n_aug + 1):
            noise = rng.randn(beats.shape[1], 1).astype(np.float32) * noise_std
            scale = 1.0 + rng.uniform(-amp_range, amp_range)
            views_all[base + j] = beats[i] * scale + noise

    probs = model.predict(views_all, batch_size=batch_size, verbose=0)
    if probs.shape[-1] >= 2:
        abnormal_probs = probs[:, 1]
    else:
        abnormal_probs = probs[:, 0]

    result = np.zeros(n_beats, dtype=np.float32)
    for i in range(n_beats):
        base = i * (n_aug + 1)
        aug_probs = abnormal_probs[base:base + n_aug + 1]
        if aggregation == 'max':
            result[i] = np.max(aug_probs)
        else:
            result[i] = np.mean(aug_probs)
    return result


# ---------------------------------------------------------------------------
# Multi-beat Confirmation
# ---------------------------------------------------------------------------

def multi_beat_confirm(beat_probs, threshold=0.35, n_confirm=3):
    probs = np.asarray(beat_probs, dtype=np.float32)
    flags = np.zeros(len(probs), dtype=np.int8)

    for i in range(len(probs)):
        if i < n_confirm - 1:
            flags[i] = 0
            continue
        window = probs[max(0, i - n_confirm + 1):i + 1]
        if np.all(window > threshold):
            flags[i] = 1
        if flags[i] == 1:
            for j in range(i - n_confirm + 1, i):
                if j >= 0 and np.all(probs[j:i + 1] > threshold):
                    flags[j] = 1

    return flags


# ---------------------------------------------------------------------------
# Full TTA Pipeline
# ---------------------------------------------------------------------------

def tta_evaluate(model, beats, labels, threshold=0.35,
                 use_sliding=False, use_augmented=True,
                 n_confirm=3, stride_samples=62, n_views=3,
                 n_aug=5, batch_size=64, record_ids=None):
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, roc_auc_score, f1_score)

    beats = np.asarray(beats, dtype=np.float32)
    if beats.ndim == 2:
        beats_input = beats[..., np.newaxis]
    else:
        beats_input = beats

    std_probs = model.predict(beats_input, batch_size=batch_size, verbose=0)
    std_abnormal = std_probs[:, 1] if std_probs.shape[-1] >= 2 else std_probs[:, 0]
    std_preds = (std_abnormal > threshold).astype(int)

    if use_augmented:
        tta_probs = tta_predict_batch_augmented(
            model, beats, n_aug=n_aug, noise_std=0.01, amp_range=0.05,
            aggregation='mean', batch_size=batch_size)
        if use_sliding:
            slide_probs = tta_predict_batch_sliding(
                model, beats, stride_samples=stride_samples, n_views=n_views,
                aggregation='mean', batch_size=batch_size)
            tta_probs = (tta_probs + slide_probs) / 2.0
    elif use_sliding:
        tta_probs = tta_predict_batch_sliding(
            model, beats, stride_samples=stride_samples, n_views=n_views,
            aggregation='mean', batch_size=batch_size)
    else:
        tta_probs = std_abnormal

    tta_preds = (tta_probs > threshold).astype(int)

    tta_confirmed_probs = tta_probs.copy()
    if record_ids is not None:
        for rid in np.unique(record_ids):
            mask = record_ids == rid
            if np.sum(mask) < n_confirm:
                continue
            confirmed = multi_beat_confirm(
                tta_probs[mask], threshold=threshold, n_confirm=n_confirm)
            tta_confirmed_probs[mask] = np.where(confirmed == 1,
                                                  np.maximum(tta_probs[mask], 0.5),
                                                  tta_probs[mask])
    confirmed_preds = (tta_confirmed_probs > max(threshold, 0.5)).astype(int)

    def compute_metrics(preds, probs):
        return {
            'accuracy': float(accuracy_score(labels, preds)),
            'precision': float(precision_score(labels, preds, zero_division=0)),
            'recall': float(recall_score(labels, preds, zero_division=0)),
            'f1': float(f1_score(labels, preds, zero_division=0)),
            'auc': float(roc_auc_score(labels, probs)),
        }

    return {
        'standard': compute_metrics(std_preds, std_abnormal),
        'tta': compute_metrics(tta_preds, tta_probs),
        'tta_confirmed': compute_metrics(confirmed_preds, tta_confirmed_probs),
    }


# ---------------------------------------------------------------------------
# ESP32-compatible Streaming TTA
# ---------------------------------------------------------------------------

def tta_streaming_buffer(window_size=250, stride=62, n_windows=3):
    buf = np.zeros(window_size + stride * (n_windows - 1), dtype=np.float32)
    idx = 0

    def push(sample):
        nonlocal idx
        buf[:-1] = buf[1:]
        buf[-1] = sample
        idx += 1
        return idx >= len(buf)

    def get_windows():
        windows = np.zeros((n_windows, window_size), dtype=np.float32)
        for i in range(n_windows):
            start = i * stride
            windows[i] = buf[start:start + window_size]
        return windows[..., np.newaxis]

    return push, get_windows


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print("[TTA] Unit tests...")

    beat = np.sin(np.linspace(0, 4 * np.pi, 250)).astype(np.float32)
    views = tta_sliding_window(beat, stride_samples=62, n_views=3)
    assert views.shape == (3, 250, 1), f"Expected (3,250,1), got {views.shape}"
    print(f"  Sliding window: OK ({views.shape})")

    probs = np.array([0.1, 0.4, 0.5, 0.6, 0.3, 0.8, 0.9, 0.95], dtype=np.float32)
    flags = multi_beat_confirm(probs, threshold=0.35, n_confirm=3)
    assert flags[3] == 1, f"Expected confirmed at idx 3, got {flags}"
    assert flags[7] == 1, f"Expected confirmed at idx 7, got {flags}"
    print(f"  Multi-beat: {probs} -> {flags}")

    aug_views = tta_augmented_views(beat, n_aug=3)
    assert aug_views.shape == (4, 250, 1), f"Expected (4,250,1), got {aug_views.shape}"
    print(f"  Augmented views: OK ({aug_views.shape})")

    print("[TTA] All tests passed!")
