#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTB-XL Rhythm-Only Preprocessing -> Beat-Level Dataset
=====================================================

TARGET: Only extract records with RHYTHM diagnoses where the label
        plausibly applies to ALL extracted beats.

STRATEGY (fundamentally different from preprocess_ptbxl.py):
  1. FILTER: Keep only records with rhythm arrhythmia diagnoses
     (AFIB, AFLT, SVT, PVC, PAC, AV block, etc.)
  2. EXCLUDE: Structural/morphological diagnoses
     (MI, LVH, ischemia, etc. - these are NOT beat-level labels)
  3. EXTRACT: ALL detected beats (not just 3 per record)
  4. LABEL: Beat inherits record rhythm label (reasonable assumption
     for rhythm diagnoses)

Output: data/processed/ptbxl_rhythm_processed.npz
"""

import sys, ast
from pathlib import Path
import numpy as np
from scipy import signal as scipy_signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, TARGET_FS, BEAT_WINDOW_SAMPLES

import os

# Auto-detect WSL2 vs Windows path
_RAW = r"C:\Users\cai\OneDrive\Desktop\Fe programme 25261\ecg-programme-tju-flex.electron-master\PTB-XL_ECG"
_WSL = "/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/PTB-XL_ECG"
PTBXL_DIR = Path(_WSL if os.path.exists(_WSL) else _RAW)
PTBXL_CSV = PTBXL_DIR / "ptbxl_database.csv"

# Rhythm diagnoses: apply to ALL beats in the record
RHYTHM_NORMAL = {"NORM", "SR", "SBRAD", "STACH", "SARRH"}
RHYTHM_ABNORMAL = {
    "AFIB", "AFLT",                         # Afib/Aflutter - chaotic, all beats
    "SVARR", "SVTAC", "PSVT",              # SV arrhythmias
    "PVC", "PAC", "BIGU", "TRIGU",         # Ectopic beats
    "1AVB", "2AVB", "3AVB",               # AV block - affects all beats
    "PACE", "WPW", "LNGQT",               # Paced / WPW / Long QT
}

# Structural diagnoses: NOT beat-level, skip these records
STRUCTURAL_EXCLUDE = {
    "IMI", "ASMI", "AMI", "ALMI", "ILMI", "LMI", "IPLMI", "PMI",
    "LVH", "RVH", "LAO/LAE", "RAO/RAE", "SEHYP", "VCLVH",
    "ISC_", "ISCAL", "ISCIN", "ISCIL", "ISCAS", "ISCLA",
    "INJAS", "INJAL", "INJIN", "INJIL",
    "NST_", "DIG", "LOWT", "NT_", "INVT", "TAB_", "STE_", "STD_",
    "ABQRS", "QWAVE", "LVOLT", "HVOLT", "NDT", "ANEUR", "EL", "DTI",
    "CRBBB", "CLBBB", "IRBBB", "IVCD", "LAFB", "LPFB", "LPR",
}



def classify_rhythm_only(scp_str):
    """Return 0=Normal, 1=Abnormal, -1=Skip (structural only)."""
    try:
        codes = set(ast.literal_eval(scp_str).keys())
    except:
        return -1
    has_rhythm_ab = bool(codes & RHYTHM_ABNORMAL)
    has_structural = bool(codes & STRUCTURAL_EXCLUDE)
    if has_structural and not has_rhythm_ab:
        return -1
    if has_rhythm_ab:
        return 1
    if codes.issubset(RHYTHM_NORMAL):
        return 0
    return -1


def apply_filters(signal, fs):
    """ESP32-matched: HP 0.5 + LP 40 + Notch 50."""
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
    # Relaxed: 10% of max energy (was 30%)
    th = max(0.10 * np.max(ma), 0.05 * np.std(signal))
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(ma, height=th, distance=int(0.15*fs))
    # Fallback: ultra-relaxed if <3 peaks
    if len(peaks) < 3:
        th2 = 0.03 * np.max(ma) if np.max(ma) > 0 else 0.005
        peaks, _ = find_peaks(ma, height=th2, distance=int(0.10*fs))
    return peaks


def process_ptbxl_rhythm(max_records=None):
    """Extract beat-level data from PTB-XL rhythm subset."""
    import pandas as pd, wfdb
    print("="*60 + "\n PTB-XL Rhythm-Only -> Beat Dataset\n" + "="*60)
    db = pd.read_csv(PTBXL_CSV)
    db = db[db["validated_by_human"] == True].copy()
    db["rhythm_label"] = db["scp_codes"].apply(classify_rhythm_only)
    n_skip = (db["rhythm_label"] == -1).sum()
    db = db[db["rhythm_label"] >= 0].copy()
    nN, nA = (db["rhythm_label"]==0).sum(), (db["rhythm_label"]==1).sum()
    print(f"[FILTER] {len(db)} rhythm records (N={nN}, A={nA}), "
          f"skipped {n_skip} structural")
    if max_records:
        db = db.head(max_records)

    all_beats, all_labels, all_rids = [], [], []
    failed, total = 0, 0
    half = BEAT_WINDOW_SAMPLES // 2

    for i, (_, row) in enumerate(db.iterrows()):
        try:
            rec_path = str(PTBXL_DIR / row["filename_hr"])
            rec = wfdb.rdrecord(rec_path, channels=[1])
            sig = rec.p_signal[:,0].astype(np.float64)

            # Debug first 3 records
            if i < 3:
                print(f"  [DEBUG] {row['ecg_id']}: sig_len={len(sig)}, "
                      f"fs={rec.fs}, min={sig.min():.3f}, max={sig.max():.3f}")
            n_tgt = int(len(sig)*TARGET_FS/rec.fs)
            sig250 = scipy_signal.resample(sig, n_tgt)
            sig_f = apply_filters(sig250, TARGET_FS)
            rp = detect_r_peaks(sig_f, TARGET_FS)
            if i < 3:
                print(f"          after resample+filter: len={len(sig_f)}, "
                      f"R-peaks found: {len(rp)}")
            rec_beats = []
            for r in rp:
                lo = r - half
                if 0 <= lo and lo+BEAT_WINDOW_SAMPLES <= len(sig_f):
                    beat = sig_f[lo:lo+BEAT_WINDOW_SAMPLES]
                    s = beat.std()
                    if s > 1e-8:
                        rec_beats.append((beat-beat.mean())/s)
            if rec_beats:
                b = np.array(rec_beats, dtype=np.float32)
                all_beats.append(b)
                all_labels.append(np.full(len(b), row["rhythm_label"], dtype=np.int32))
                all_rids.append(np.full(len(b), row["strat_fold"], dtype=np.int32))
                total += len(b)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  [SKIP] {row['ecg_id']}: {e}")
        if (i+1) % 1000 == 0:
            print(f"  [{i+1}/{len(db)}] {total} beats, {failed} failed")

    fb = np.concatenate(all_beats) if all_beats else np.empty((0,BEAT_WINDOW_SAMPLES), np.float32)
    fl = np.concatenate(all_labels) if all_labels else np.empty((0,), np.int32)
    fr = np.concatenate(all_rids) if all_rids else np.empty((0,), np.int32)
    nN, nA = (fl==0).sum(), (fl==1).sum()
    print(f"\n[DONE] {len(fb)} beats (N={nN}, A={nA}), {failed} failed")
    return {"beats": fb, "labels": fl, "record_ids": fr}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=None, help="Max records to process")
    args = p.parse_args()
    r = process_ptbxl_rhythm(max_records=args.max)
    out = PROCESSED_DIR / "ptbxl_rhythm_processed.npz"
    np.savez_compressed(out, beats=r["beats"], labels=r["labels"],
                        record_ids=r["record_ids"])
    print(f"\n[SAVED] {out} ({out.stat().st_size/1024/1024:.1f} MB)")
    print("[NEXT] python train.py --incart (先跑 INCART, PTB-XL 备选)")
