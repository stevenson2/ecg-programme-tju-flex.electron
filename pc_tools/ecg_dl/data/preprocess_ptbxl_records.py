#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTB-XL Record-Level Preprocessing for Route K Supervised Pretraining.

Pipeline:
  1. Load ptbxl_database.csv + scp_statements.csv
  2. Filter: validated_by_human=True
  3. Map SCP codes → 5 superclasses (NORM/MI/CD/STTC/HYP)
  4. Load 100Hz WFDB records → Extract Lead I + Lead II
  5. 10s records → 1000 samples per lead
  6. Save .npz compatible with pytorch-style training

Output: data/processed/ptbxl_records_100hz.npz
  - signals: (N, 1000, 2) float32  [Lead I, Lead II]
  - labels:  (N, 5) float32        multi-label one-hot
  - folds:   (N,) int32             strat_fold 1-10
  - ecg_ids: (N,) int32
"""

import sys, os, ast
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR

PTBXL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "PTB-XL_ECG"
PTBXL_CSV = PTBXL_DIR / "ptbxl_database.csv"
SCP_CSV = PTBXL_DIR / "scp_statements.csv"

SUPERCLASS_MAP = {
    'NORM': 0, 'MI': 1, 'CD': 2, 'STTC': 3, 'HYP': 4,
}
SUPERCLASS_NAMES = ['NORM', 'MI', 'CD', 'STTC', 'HYP']


def load_metadata():
    import pandas as pd
    ptbxl = pd.read_csv(PTBXL_CSV)
    scp = pd.read_csv(SCP_CSV, index_col=0)
    # Filter: only diagnostic-level SCP codes (exclude form/rhythm)
    scp_diag = scp[scp['diagnostic'] == 1.0].copy()
    return ptbxl, scp_diag


def scp_to_superclass(scp_codes_str, scp_df):
    """Map SCP codes string to 5-dim multi-label vector, diagnostic-only."""
    try:
        codes = ast.literal_eval(scp_codes_str)
    except Exception:
        codes = {}

    labels = np.zeros(5, dtype=np.float32)
    for code in codes:
        if code in scp_df.index:
            dclass = scp_df.loc[code, 'diagnostic_class']
            if dclass in SUPERCLASS_MAP:
                labels[SUPERCLASS_MAP[dclass]] = 1.0
    return labels


def load_record(filename_lr):
    """Load a single 100Hz WFDB record, return (2, 1000) float32 [Lead I, Lead II]."""
    import wfdb
    record_path = str(PTBXL_DIR / filename_lr)
    record = wfdb.rdrecord(record_path, channels=[0, 1])
    sig = record.p_signal.astype(np.float32)  # (1000, 2)

    sig = np.nan_to_num(sig, nan=0.0).T  # (2, 1000)

    # Z-score normalize per lead
    for ch in range(sig.shape[0]):
        mean = np.mean(sig[ch])
        std = np.std(sig[ch])
        if std > 1e-8:
            sig[ch] = (sig[ch] - mean) / std

    return sig


def preprocess(max_records=None):
    print("[PTB-XL Records] Loading metadata...")
    ptbxl, scp = load_metadata()

    # Filter validated
    ptbxl = ptbxl[ptbxl['validated_by_human'] == True].copy()
    print(f"  Validated records: {len(ptbxl)}")

    if max_records:
        ptbxl = ptbxl.iloc[:max_records]

    n = len(ptbxl)
    signals = np.zeros((n, 2, 1000), dtype=np.float32)
    labels = np.zeros((n, 5), dtype=np.float32)
    folds = np.zeros(n, dtype=np.int32)
    ecg_ids = np.zeros(n, dtype=np.int32)

    skip = 0
    for i, (_, row) in enumerate(ptbxl.iterrows()):
        if i % 2000 == 0 and i > 0:
            print(f"  Processed {i}/{n}...")

        try:
            sig = load_record(row['filename_lr'])
            signals[i - skip] = sig
        except Exception as e:
            skip += 1
            continue

        labels[i - skip] = scp_to_superclass(row['scp_codes'], scp)
        folds[i - skip] = int(row['strat_fold'])
        ecg_ids[i - skip] = int(row['ecg_id'])

    if skip > 0:
        signals = signals[:n - skip]
        labels = labels[:n - skip]
        folds = folds[:n - skip]
        ecg_ids = ecg_ids[:n - skip]
        print(f"  Skipped {skip} unreadable records")

    print(f"\n  Final: {len(signals)} records")
    print(f"  Shape: signals={signals.shape}, labels={labels.shape}")
    print(f"  Class distribution:")
    for i, name in enumerate(SUPERCLASS_NAMES):
        count = int(labels[:, i].sum())
        print(f"    {name}: {count} ({count/len(labels)*100:.1f}%)")

    out_path = PROCESSED_DIR / "ptbxl_records_100hz.npz"
    np.savez_compressed(out_path,
                        signals=signals, labels=labels,
                        folds=folds, ecg_ids=ecg_ids,
                        class_names=SUPERCLASS_NAMES)
    print(f"\n  Saved: {out_path}")
    return signals, labels, folds, ecg_ids


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=None,
                        help='Max records (for testing)')
    args = parser.parse_args()
    preprocess(max_records=args.max)
