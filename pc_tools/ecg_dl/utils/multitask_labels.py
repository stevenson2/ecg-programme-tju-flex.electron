#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-task auxiliary label generation for Route F.

Generates pseudo-labels from beat waveforms:
  - BPM: Dominant frequency via FFT (Hz → BPM × 60), normalized to [0, 1]
  - SQI: Kurtosis-based signal quality [0, 1]

All labels are normalised to [0, 1] so loss scales are comparable
across classification, regression, and quality tasks.
"""

import numpy as np
from scipy.stats import kurtosis

BPM_MIN = 30.0
BPM_MAX = 180.0


def compute_pseudo_bpm(beat, fs=250):
    n = len(beat)
    beat_centered = beat - np.mean(beat)
    fft = np.abs(np.fft.rfft(beat_centered))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    mask = (freqs >= 0.5) & (freqs <= 3.0)
    if not np.any(mask):
        return 72.0
    peak_idx = np.argmax(fft[mask])
    dom_freq = freqs[mask][peak_idx]
    bpm = dom_freq * 60.0
    return float(np.clip(bpm, BPM_MIN, BPM_MAX))


def bpm_normalize(bpm_raw):
    return (bpm_raw - BPM_MIN) / (BPM_MAX - BPM_MIN)


def bpm_denormalize(bpm_norm):
    return bpm_norm * (BPM_MAX - BPM_MIN) + BPM_MIN


def compute_pseudo_sqi(beat, fs=250):
    beat_centered = beat - np.mean(beat)
    rms = np.sqrt(np.mean(beat_centered ** 2))
    if rms < 1e-6:
        return 0.0
    beat_norm = beat_centered / rms
    k = kurtosis(beat_norm, fisher=True)
    sqi = 1.0 / (1.0 + np.exp(-(k - 5.0) / 3.0))
    return float(np.clip(sqi, 0.0, 1.0))


def compute_multitask_labels(beats, fs=250, normalize_bpm=True):
    n = len(beats)
    bpm_raw = np.zeros(n, dtype=np.float32)
    sqi = np.zeros(n, dtype=np.float32)
    for i in range(n):
        bpm_raw[i] = compute_pseudo_bpm(beats[i], fs=fs)
        sqi[i] = compute_pseudo_sqi(beats[i], fs=fs)
    if normalize_bpm:
        bpm = bpm_normalize(bpm_raw)
    else:
        bpm = bpm_raw
    return bpm, sqi


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    fake = rng.randn(4, 250).astype(np.float32)
    b, s = compute_multitask_labels(fake)
    for i in range(4):
        print(f"Beat {i}: BPM={b[i]:.1f}, SQI={s[i]:.3f}")
    print("[multitask_labels] OK")
