#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTB-XL ECG Preprocessing → Unified 250Hz Beat-Level Dataset

Pipeline:
  1. Parse ptbxl_database.csv + scp_statements.csv
  2. Filter: validated_by_human=True
  3. Label: NORM-only → Normal(0), diagnostic codes → Abnormal(1)
  4. Load WFDB → Extract Lead II → 500→250Hz resample
     → HP 0.5Hz+LP 40Hz+Notch 50Hz → R-peak detect → beat cutting
  5. Save .npz compatible with dataset.py

Output: data/processed/ptbxl_processed.npz
"""

import sys, ast
from pathlib import Path
import numpy as np
from scipy import signal as scipy_signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, TARGET_FS, BEAT_WINDOW_SAMPLES

# Paths
PTBXL_DIR = Path(r"C:\Users\cai\OneDrive\Desktop\ecg-programme-tju-flex.electron-master\PTB-XL_ECG")
PTBXL_CSV = PTBXL_DIR / "ptbxl_database.csv"
SCP_CSV = PTBXL_DIR / "scp_statements.csv"

# SCP label mapping
NORMAL_SCP = {"NORM", "SBRAD", "STACH", "SARRH", "SR"}
ABNORMAL_SCP = {
    "IMI","ASMI","AMI","ALMI","ILMI","LMI","IPLMI","PMI",
    "LVH","RVH","LAO/LAE","RAO/RAE","SEHYP","VCLVH",
    "CRBBB","CLBBB","IRBBB","IVCD","LAFB","LPFB","WPW",
    "1AVB","2AVB","3AVB","LPR",
    "ISC_","ISCAL","ISCIN","ISCIL","ISCAS","ISCLA",
    "INJAS","INJAL","INJIN","INJIL",
    "NST_","DIG","LNGQT","LOWT","NT_","INVT","TAB_","STE_","STD_",
    "AFIB","AFLT","SVARR","SVTAC","PSVT",
    "PVC","PAC","PRC(S)","BIGU","TRIGU",
    "ABQRS","QWAVE","LVOLT","HVOLT","NDT","ANEUR","EL","DTI","PACE",
}

_scp_df = None
def _load_scp():
    global _scp_df
    if _scp_df is None:
        import pandas as pd
        _scp_df = pd.read_csv(SCP_CSV, index_col=0)
    return _scp_df

def parse_scp(scp_str):
    try: return ast.literal_eval(scp_str)
    except: return {}

def classify(codes):
    c = set(codes.keys())
    if c & ABNORMAL_SCP: return 1
    if c.issubset(NORMAL_SCP): return 0
    df = _load_scp()
    for code in c - NORMAL_SCP:
        if code in df.index and df.loc[code].get("diagnostic", 0) == 1.0:
            return 1
    return 0 if c.issubset(NORMAL_SCP) else 1


# ===========================================================================
# Digital Filters (ESP32 IIR Biquad matched)
# ===========================================================================

def _butter_hp(cutoff, fs, order=2):
    b, a = scipy_signal.butter(order, cutoff/(0.5*fs), btype='high')
    return b, a

def _butter_lp(cutoff, fs, order=2):
    b, a = scipy_signal.butter(order, cutoff/(0.5*fs), btype='low')
    return b, a

def _notch(f0, fs, Q=20.0):
    return scipy_signal.iirnotch(f0, Q, fs)

def apply_filters(signal, fs):
    bh, ah = _butter_hp(0.5, fs); bl, al = _butter_lp(40.0, fs)
    bn, an = _notch(50.0, fs)
    sig = scipy_signal.filtfilt(bh, ah, signal)
    sig = scipy_signal.filtfilt(bl, al, sig)
    sig = scipy_signal.filtfilt(bn, an, sig)
    return sig.astype(np.float32)


# ===========================================================================
# R-Peak Detection (simplified Pan-Tompkins)
# ===========================================================================

def detect_r(signal, fs=250, thresh=0.3, min_dist_ms=200):
    sig = np.asarray(signal, dtype=np.float32)
    nyq = 0.5 * fs
    b, a = scipy_signal.butter(2, [5.0/nyq, 15.0/nyq], btype='band')
    f = scipy_signal.filtfilt(b, a, sig)
    d = np.diff(f, prepend=f[0])
    s = d ** 2
    win = int(0.080 * fs)
    ma = np.convolve(s, np.ones(win)/win, mode='same')
    th = max(thresh * np.max(ma), 0.3 * np.std(sig)) if np.max(ma) > 0 else 0.5*np.std(sig)
    min_d = int(min_dist_ms / 1000.0 * fs)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(ma, height=th, distance=min_d)
    if len(peaks) == 0:
        peaks, _ = find_peaks(ma, height=0.1*np.max(ma), distance=min_d)
    rw = int(0.020 * fs)
    refined = []
    for p in peaks:
        lo, hi = max(0, p - rw), min(len(sig), p + rw)
        refined.append(lo + np.argmax(np.abs(sig[lo:hi])))
    return np.array(refined, dtype=int)


def extract_beats(signal, r_peaks, win=250, max_per_record=3):
    """Extract up to max_per_record non-overlapping beats."""
    half = win // 2
    beats = []
    last_end = -win
    for r in r_peaks:
        lo = r - half
        if lo >= 0 and lo + win <= len(signal) and lo >= last_end + win:
            beats.append(signal[lo:lo+win])
            last_end = lo + win
            if len(beats) >= max_per_record:
                break
    return np.array(beats, dtype=np.float32) if beats else np.empty((0, win), dtype=np.float32)


# ===========================================================================
# Main Pipeline
# ===========================================================================

def process_ptbxl(max_records=None, lead_idx=1):
    """Full PTB-XL → 250Hz beat dataset."""
    import pandas as pd
    import wfdb

    print(f"{'='*60}\n PTB-XL → 250Hz Beat Dataset\n{'='*60}\n")

    db = pd.read_csv(PTBXL_CSV)
    db = db[db["validated_by_human"] == True].copy()
    print(f"[1/4] 有效记录: {len(db)} (human-validated only)")

    if max_records:
        db = db.head(max_records)
        print(f"      [LIMIT] first {max_records} records")

    db["label"] = db["scp_codes"].apply(lambda s: classify(parse_scp(s)))
    nN, nA = (db["label"]==0).sum(), (db["label"]==1).sum()
    print(f"[2/4] Labels: Normal={nN}, Abnormal={nA}")

    print(f"[3/4] Processing records (Lead II, 250Hz, beat cutting)...")
    all_beats, all_labels, all_rids, all_folds = [], [], [], []
    failed = 0

    for i, (_, row) in enumerate(db.iterrows()):
        try:
            rec = wfdb.rdrecord(str(PTBXL_DIR / row["filename_hr"]), channels=[lead_idx])
            sig = rec.p_signal[:, 0].astype(np.float64)
            n_tgt = int(len(sig) * TARGET_FS / rec.fs)
            sig250 = scipy_signal.resample(sig, n_tgt)
            sig_f = apply_filters(sig250, TARGET_FS)
            rp = detect_r(sig_f, TARGET_FS)
            beats = extract_beats(sig_f, rp, BEAT_WINDOW_SAMPLES)
            if len(beats) > 0:
                all_beats.append(beats)
                all_labels.append(np.full(len(beats), row["label"], dtype=np.int32))
                # Use strat_fold as record_id to enable proper per-fold splitting
                all_rids.append(np.full(len(beats), row["strat_fold"], dtype=np.int32))
                all_folds.append(np.full(len(beats), row["strat_fold"], dtype=np.int32))
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  [SKIP] ecg_id={row['ecg_id']}: {e}")
        if (i+1) % 3000 == 0:
            print(f"  [{i+1}/{len(db)}] {sum(len(b) for b in all_beats)} beats, {failed} failed")

    fb = np.concatenate(all_beats) if all_beats else np.empty((0, BEAT_WINDOW_SAMPLES), np.float32)
    fl = np.concatenate(all_labels) if all_labels else np.empty((0,), np.int32)
    fr = np.concatenate(all_rids) if all_rids else np.empty((0,), np.int32)

    nN, nA = (fl==0).sum(), (fl==1).sum()
    print(f"\n[4/4] Done: {len(fb)} beats (N={nN}, A={nA}), {failed}/{len(db)} failed\n{'='*60}")
    return {"beats": fb, "labels": fl, "record_ids": fr}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="PTB-XL preprocessing")
    p.add_argument("--max", type=int, default=None)
    p.add_argument("--lead", type=int, default=1)
    p.add_argument("--output", type=str, default="ptbxl_processed.npz")
    args = p.parse_args()

    r = process_ptbxl(max_records=args.max, lead_idx=args.lead)
    out = PROCESSED_DIR / args.output
    np.savez_compressed(out, beats=r["beats"], labels=r["labels"], record_ids=r["record_ids"])
    mb = out.stat().st_size / 1024 / 1024
    print(f"[OK] Saved: {out} ({mb:.1f} MB)")

