#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECGdata_1000 Preprocessing -> 250Hz Beat-Level Dataset

Dataset: 1000 x 12-lead ECG recordings
  - 600 labeled (300 Normal + 300 Abnormal)
  - 500 Hz, 10-sec segments, MAT format
  - Channels: Lead I(0), II(1), III, aVR, aVL, aVF, V1-V6

Output: data/processed/ecg1000_processed.npz
"""

import sys, re
from pathlib import Path
import numpy as np
from scipy import signal as scipy_signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, TARGET_FS, BEAT_WINDOW_SAMPLES

DATA_DIR = Path(r"C:\Users\cai\OneDrive\Desktop\Fe programme 25261\ecg-programme-tju-flex.electron-master\ECGdata_1000")
TRAIN_DIR = DATA_DIR / "TRAIN"
REF_FILE = DATA_DIR / "reference.txt"
ORIG_FS = 500


def load_labels():
    """Parse reference.txt -> {TRAINxxx: label}."""
    labels = {}
    with open(REF_FILE) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                name, label = parts[0], int(parts[1])
                if name.startswith("TRAIN"):
                    labels[name] = label
    return labels


def apply_esp32_filters(signal, fs):
    """HP(0.5Hz) -> LP(40Hz) -> Notch(50Hz)."""
    bh, ah = scipy_signal.butter(2, 0.5/(0.5*fs), btype='high')
    bl, al = scipy_signal.butter(2, 40.0/(0.5*fs), btype='low')
    bn, an = scipy_signal.iirnotch(50.0, 20.0, fs)
    sig = scipy_signal.filtfilt(bh, ah, signal)
    sig = scipy_signal.filtfilt(bl, al, sig)
    sig = scipy_signal.filtfilt(bn, an, sig)
    return sig.astype(np.float32)


def detect_r_peaks(signal, fs=250):
    """Pan-Tompkins R-peak detection (relaxed for short recordings)."""
    nyq = 0.5*fs
    b, a = scipy_signal.butter(2, [5.0/nyq, 15.0/nyq], btype='band')
    f = scipy_signal.filtfilt(b, a, signal)
    ma = np.convolve(np.diff(f, prepend=f[0])**2,
                     np.ones(int(0.080*fs))/int(0.080*fs), mode='same')
    # Relaxed threshold: 15% of max OR 10% of std (was 30%)
    th = max(0.15 * np.max(ma), 0.1 * np.std(signal))
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(ma, height=th, distance=int(0.15*fs))
    # Fallback: if <3 peaks found, lower threshold further
    if len(peaks) < 3:
        th2 = 0.05 * np.max(ma) if np.max(ma) > 0 else 0.01
        peaks, _ = find_peaks(ma, height=th2, distance=int(0.12*fs))
    return peaks


def extract_beats_from_signal(signal_250, r_peaks):
    """Extract 250-sample beats around R-peaks (relaxed edge margin)."""
    half = BEAT_WINDOW_SAMPLES // 2
    beats = []
    for r in r_peaks:
        lo = r - half
        hi = lo + BEAT_WINDOW_SAMPLES
        # Allow beats closer to edges (only need 200 samples, pad the rest)
        if lo < 0 or hi > len(signal_250):
            # Try padding
            beat_raw = signal_250[max(0, lo):min(len(signal_250), hi)]
            if len(beat_raw) < 150:  # Too short, skip
                continue
            beat = np.zeros(BEAT_WINDOW_SAMPLES, dtype=np.float32)
            start = -lo if lo < 0 else 0
            beat[start:start+len(beat_raw)] = beat_raw
        else:
            beat = signal_250[lo:hi].copy()
        s = beat.std()
        if s > 1e-8:
            beats.append((beat - beat.mean()) / s)
    return beats


def process_ecg1000(dual_lead=False):
    """Process ECGdata_1000 -> beat-level dataset."""
    import scipy.io as sio
    labels_map = load_labels()
    nN = sum(1 for v in labels_map.values() if v == 0)
    nA = sum(1 for v in labels_map.values() if v == 1)
    print(f"[ECG1000] {len(labels_map)} records (N={nN}, A={nA})")

    all_beats, all_labels, all_rids = [], [], []
    failed, total = 0, 0
    files = sorted(TRAIN_DIR.glob("TRAIN*.mat"))

    for i, fp in enumerate(files):
        name = fp.stem
        label = labels_map.get(name)
        if label is None:
            continue
        try:
            mat = sio.loadmat(str(fp))
            data = mat["data"]  # (12, 5000)
            n250 = int(data.shape[1] * TARGET_FS / ORIG_FS)
            data_250 = scipy_signal.resample(data, n250, axis=1)

            # Lead II (ch1) + filters
            lead2 = apply_esp32_filters(data_250[1], TARGET_FS)
            rp = detect_r_peaks(lead2, TARGET_FS)
            beats_l2 = extract_beats_from_signal(lead2, rp)

            # Dual-lead: also Lead I (ch0)
            beats_l1 = []
            if dual_lead:
                lead1 = apply_esp32_filters(data_250[0], TARGET_FS)
                beats_l1 = extract_beats_from_signal(lead1, rp)

            all_rec = beats_l2 + beats_l1
            if all_rec:
                b = np.array(all_rec, dtype=np.float32)
                all_beats.append(b)
                all_labels.append(np.full(len(b), label, dtype=np.int32))
                rid = int(re.search(r'\d+', name).group())
                all_rids.append(np.full(len(b), rid, dtype=np.int32))
                total += len(b)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  [SKIP] {name}: {e}")
        if (i+1) % 100 == 0:
            print(f"  [{i+1}/{len(files)}] {total} beats, {failed} failed")

    fb = np.concatenate(all_beats) if all_beats else np.empty((0,BEAT_WINDOW_SAMPLES), np.float32)
    fl = np.concatenate(all_labels) if all_labels else np.empty((0,), np.int32)
    fr = np.concatenate(all_rids) if all_rids else np.empty((0,), np.int32)
    nN, nA = (fl==0).sum(), (fl==1).sum()
    print(f"\n[ECG1000] Total: {len(fb)} beats (N={nN}, A={nA}), failed={failed}")
    return {"beats": fb, "labels": fl, "record_ids": fr}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dual", action="store_true", help="Dual-lead (Lead I+II)")
    p.add_argument("--merge", action="store_true", help="Merge with MIT-BIH")
    args = p.parse_args()

    r = process_ecg1000(dual_lead=args.dual)

    out = PROCESSED_DIR / "ecg1000_processed.npz"
    np.savez_compressed(out, beats=r["beats"], labels=r["labels"],
                        record_ids=r["record_ids"])
    print(f"[SAVED] {out} ({out.stat().st_size/1024/1024:.1f} MB)")

    if args.merge:
        mit = np.load(PROCESSED_DIR / "mit_bih_processed.npz")
        b_all = np.concatenate([mit["beats"], r["beats"]])
        l_all = np.concatenate([mit["labels"], r["labels"]])
        r_all = np.concatenate([mit["record_ids"], r["record_ids"] + 200000])
        mp = PROCESSED_DIR / "mit_bih_ecg1000_merged.npz"
        np.savez_compressed(mp, beats=b_all, labels=l_all, record_ids=r_all)
        nN, nA = (l_all==0).sum(), (l_all==1).sum()
        print(f"[MERGED] {len(b_all)} beats (N={nN}, A={nA})")

    print("\n[DONE] Next: python train.py --ecg1000")

