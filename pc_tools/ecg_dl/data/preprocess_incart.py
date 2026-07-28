#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INCART (St. Petersburg INCART 12-lead Arrhythmia Database) Preprocessing
-> Unified 250Hz Beat-Level Dataset

Pipeline:
  1. Download INCART from PhysioNet (if not present)
  2. Load WFDB records (257Hz, 12-lead, Lead II extraction)
  3. Resample 257->250Hz
  4. ESP32-matched IIR filters: HP 0.5Hz + LP 40Hz + Notch 50Hz
  5. R-peak centered beat cutting (250 samples)
  6. AAMI label mapping -> Normal(0)/Abnormal(1)
  7. Save .npz compatible with dataset.py

Output: data/processed/incart_processed.npz

INCART Database:
  - 75 records (I01-I75), ~30 min each
  - 257 Hz, 12-lead, beat-level annotations
  - ~175K annotated beats
  - Complements MIT-BIH with long-duration records
"""

import sys
from pathlib import Path
import numpy as np
from scipy import signal as scipy_signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, TARGET_FS, BEAT_WINDOW_SAMPLES, AAMI_CLASSES

INCART_DIR = Path(r"C:\Users\cai\OneDrive\Desktop\ecg-programme-tju-flex.electron-master\st-petersburg-incart-12-lead-arrhythmia-database-1.0.0\files")
INCART_RECORDS = [f"I{i:02d}" for i in range(1, 76)]


def _butter_hp(cutoff, fs, order=2):
    b, a = scipy_signal.butter(order, cutoff / (0.5 * fs), btype='high')
    return b, a

def _butter_lp(cutoff, fs, order=2):
    b, a = scipy_signal.butter(order, cutoff / (0.5 * fs), btype='low')
    return b, a

def _notch(f0, fs, Q=20.0):
    return scipy_signal.iirnotch(f0, Q, fs)

def apply_filters(signal, fs):
    """HP(0.5Hz) -> LP(40Hz) -> Notch(50Hz) - ESP32 IIR Biquad matched."""
    bh, ah = _butter_hp(0.5, fs); bl, al = _butter_lp(40.0, fs)
    bn, an = _notch(50.0, fs)
    sig = scipy_signal.filtfilt(bh, ah, signal)
    sig = scipy_signal.filtfilt(bl, al, sig)
    sig = scipy_signal.filtfilt(bn, an, sig)
    return sig.astype(np.float32)

def resample_ecg(signal, orig_fs, target_fs):
    n_target = int(len(signal) * target_fs / orig_fs)
    return scipy_signal.resample(signal, n_target)

def download_incart():
    """Download INCART database from PhysioNet if not present."""
    if INCART_DIR.exists() and list(INCART_DIR.glob("*.dat")):
        print(f"[INCART] 数据目录已存在: {INCART_DIR}")
        return
    INCART_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import wfdb
        print("[INCART] 从 PhysioNet 下载 INCART 数据库...")
        wfdb.dl_database("incartdb", str(INCART_DIR))
        print("[INCART] download done")
    except Exception as e:
        print(f"[INCART] download failed: {e}")
        print("[INCART] 请手动下载: https://physionet.org/content/incartdb/")
        print(f"[INCART] 解压到: {INCART_DIR}")
        raise

def load_incart_record(record_name):
    """Load a single INCART record."""
    import wfdb
    rec = wfdb.rdrecord(str(INCART_DIR / record_name))
    ann = wfdb.rdann(str(INCART_DIR / record_name), 'atr')
    signal = rec.p_signal[:, 0].astype(np.float64)
    ann_sym = [s.decode() if isinstance(s, bytes) else s for s in ann.symbol]
    return signal, ann.sample, ann_sym, rec.fs

def extract_beats(signal, ann_idx, ann_sym, orig_fs, target_fs):
    """Extract beats centered on R-peaks with ESP32 filters."""
    sig250 = resample_ecg(signal, orig_fs, target_fs)
    sig_f = apply_filters(sig250, target_fs)
    ratio = target_fs / orig_fs
    half = BEAT_WINDOW_SAMPLES // 2
    beats, labels, skipped = [], [], 0
    for idx, sym in zip(ann_idx, ann_sym):
        if sym not in AAMI_CLASSES:
            skipped += 1; continue
        ri = int(idx * ratio)
        lo, hi = ri - half, ri - half + BEAT_WINDOW_SAMPLES
        if lo < 0 or hi > len(sig_f):
            skipped += 1; continue
        beat = sig_f[lo:hi]
        s = beat.std()
        if s < 1e-8:
            skipped += 1; continue
        beat = (beat - beat.mean()) / s
        beats.append(beat); labels.append(AAMI_CLASSES[sym])
    return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int32), skipped


def process_all_incart(records=None):
    """Process all INCART records and return beats, labels, rec_ids."""
    if records is None:
        records = INCART_RECORDS
    all_beats, all_labels, all_rec_ids = [], [], []
    failed, total_skip = [], 0
    for i, rec_name in enumerate(records):
        rid = int(rec_name[1:])
        print(f"[INCART] [{i+1}/{len(records)}] {rec_name}...", end=" ")
        try:
            sig, idx, sym, fs = load_incart_record(rec_name)
            beats, labels, skipped = extract_beats(sig, idx, sym, fs, TARGET_FS)
            total_skip += skipped
            if len(beats) == 0:
                print("[SKIP] no beats"); failed.append(rec_name); continue
            all_beats.append(beats); all_labels.append(labels)
            all_rec_ids.append(np.full(len(beats), rid, dtype=np.int32))
            print(f"[OK] N={(labels==0).sum()} A={(labels==1).sum()}")
        except Exception as e:
            print(f"[FAIL] {e}"); failed.append(rec_name)
    if not all_beats:
        raise RuntimeError("[INCART] All records failed!")
    fb = np.concatenate(all_beats); fl = np.concatenate(all_labels)
    fr = np.concatenate(all_rec_ids)
    nN, nA = int((fl==0).sum()), int((fl==1).sum())
    print(f"\n[INCART] Total: {len(fb)}, N={nN}, A={nA}, skip={total_skip}")
    if failed:
        print(f"[INCART] Failed: {failed}")
    return fb, fl, fr


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="INCART preprocessing")
    parser.add_argument("--download", action="store_true", help="Download INCART")
    parser.add_argument("--merge", action="store_true", help="Merge with MIT-BIH after")
    args = parser.parse_args()

    if args.download:
        download_incart()
    beats, labels, rec_ids = process_all_incart()

    out = PROCESSED_DIR / "incart_processed.npz"
    np.savez_compressed(out, beats=beats, labels=labels, record_ids=rec_ids)
    print(f"\n[INCART] Saved: {out} ({out.stat().st_size/1024/1024:.1f} MB)")

    if args.merge:
        mit = np.load(PROCESSED_DIR / "mit_bih_processed.npz")
        b_all = np.concatenate([mit["beats"], beats])
        l_all = np.concatenate([mit["labels"], labels])
        r_all = np.concatenate([mit["record_ids"], rec_ids + 100000])
        mp = PROCESSED_DIR / "mit_bih_incart_merged.npz"
        np.savez_compressed(mp, beats=b_all, labels=l_all, record_ids=r_all)
        print(f"[INCART] Merged: {mp} ({mp.stat().st_size/1024/1024:.1f} MB)")

    print("\n[DONE] Run: python train.py --incart")

