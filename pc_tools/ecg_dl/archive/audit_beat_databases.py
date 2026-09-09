#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_beat_databases.py -- inventory of locally available beat-level ECG data.

The purpose is to plan a comprehensive mining campaign over all existing
annotated beat-level databases before deciding which normal/abnormal beats to
add to v3-A training.  This script only reads file listings/headers; it does
not train and does not touch test splits.
"""
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
# Use WSL paths when available.
WSL_VFDB = Path("/home/devcontainers/vfdb")
WSL_AFDB = Path("/home/devcontainers/afdb")
WSL_AFDB_WFDB = Path("/home/devcontainers/afdb_wfdb")
WSL_ECG_DATA = Path("/home/devcontainers/ecg_data")

# Windows-side paths exposed through /mnt/c in WSL.
ROOT = Path("/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261")
REPO = ROOT / "ecg-programme-tju-flex.electron-master"
MIT_DIR = REPO / "pc_tools/ecg_dl/data/raw/mit-bih-arrhythmia-database"
INCART_DIR = REPO / "st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/files"
INCART_DIR2 = REPO / "pc_tools/ecg_dl/data/raw/incartdb"
PTB_DIR = REPO / "ECG-Database"
LUDB_DIR = REPO / "ECG-Database/lobachevsky-university-electrocardiography-database-1.0.1/lobachevsky-university-electrocardiography-database-1.0.1/data"
PTBXL_DIR = REPO / "PTB-XL_ECG"
SVDB_DIR = REPO / "pc_tools/ecg_dl/data/raw/svdb"


def count_dat(d):
    if not d.exists():
        return None, 0
    return str(d), len(list(d.glob("*.dat")))


def summarize_dir(name, d, note=""):
    dat_path, n_dat = count_dat(d)
    n_atr = len(list(d.glob("*.atr"))) if d.exists() else 0
    n_qrs = len(list(d.glob("*.qrs"))) if d.exists() else 0
    n_hea = len(list(d.glob("*.hea"))) if d.exists() else 0
    return {
        "name": name,
        "path": dat_path,
        "n_dat": n_dat,
        "n_atr": n_atr,
        "n_qrs": n_qrs,
        "n_hea": n_hea,
        "note": note,
    }


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = [
        summarize_dir("MIT-BIH Arrhythmia", MIT_DIR,
                      "48 records, beat-level .atr (AAMI labels)"),
        summarize_dir("INCART full", INCART_DIR,
                      "75 records, beat-level .atr (AAMI labels)"),
        summarize_dir("INCART data/raw copy", INCART_DIR2,
                      "partial legacy copy; use full INCART if present"),
        summarize_dir("PTB (ECG-Database)", PTB_DIR,
                      "record-level diagnosis; no per-beat .atr by default"),
        summarize_dir("SVDB", SVDB_DIR,
                      "78 Holter records, beat-level .atr, record ids 800-899"),
        summarize_dir("LUDB", LUDB_DIR,
                      "Lobachevsky University DB, .qrs heart-rate gold annotations"),
        summarize_dir("PTB-XL", PTBXL_DIR,
                      "record-level diagnosis; 10s/12-lead; not beat-level"),
        summarize_dir("VFDB+CUDB", WSL_VFDB,
                      "22 VFDB + 35 CUDB; rhythm/boundary .atr; CUDB has beat N + VF []"),
        summarize_dir("AFDB WSL", WSL_AFDB,
                      "AFDB raw in WSL; .atr/.qrs present"),
        summarize_dir("AFDB wfdb WSL", WSL_AFDB_WFDB,
                      "AFDB wfdb-format copy; .qrs beat annotations"),
    ]
    # Existing clean processed arrays (deploy-causal) inventory.
    processed = []
    if WSL_ECG_DATA.exists():
        for pat in ("*_processed_deploy_causal_beats.npy",
                    "*_processed_deploy_causal_labels.npy",
                    "*_processed_deploy_causal_record_ids.npy"):
            for f in sorted(WSL_ECG_DATA.glob(pat)):
                processed.append({
                    "file": f.name,
                    "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
                    "path": str(f),
                })
    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Inventory of locally available beat-level ECG data for mining planning",
        "databases": rows,
        "processed_deploy_causal_arrays": processed,
        "elapsed_s": round(time.time() - t0, 1),
        "note": "This inventory does not decide labels or training; use SplitGuard before any mining.",
    }
    out = CACHE / "beat_db_inventory.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[BDI] saved {out}", flush=True)
    for r in rows:
        print(f"[BDI] {r['name']}: dat={r['n_dat']} atr={r['n_atr']} "
              f"qrs={r['n_qrs']} hea={r['n_hea']}", flush=True)


if __name__ == "__main__":
    main()
