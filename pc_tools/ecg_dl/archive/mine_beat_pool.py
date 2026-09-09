#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mine_beat_pool.py -- Step 1: build the train-safe beat-pool manifest.

For each locally available beat-level database this script:
  - reads the existing deploy-causal arrays (or the ones we just built),
  - applies the authoritative train mask (SplitGuard for MIT/INCART/PTB/SVDB,
    record-level split for VFDB/CUDB and AFDB),
  - counts train-safe beats and records,
  - writes a manifest under models/deploy_match/.

No combined NPZ is written; the later hard-normal mining step streams each
source independently so memory stays bounded.  No test/val contact.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.split_guard import get_guard
from data.patient_split import patient_level_split, SEED as SPLIT_SEED

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
OUT = CACHE / "beat_pool_train_manifest.json"

ECG_DATA = Path("/home/devcontainers/ecg_data")


def load_source(name, beats_path, labels_path, rids_path):
    return (np.load(beats_path, mmap_mode="r"),
            np.load(labels_path, mmap_mode="r"),
            np.load(rids_path, mmap_mode="r"))


def train_mask_splitguard(tag):
    return get_guard(tag).train_mask


def train_mask_record_ids(record_ids, train_ids):
    return np.isin(np.asarray(record_ids), np.asarray(train_ids, dtype=np.int64))


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # VFDB/CUDB split audit already exists.
    vfdb_split = json.loads((CACHE / "vfdb_split_audit.json").read_text(encoding="utf-8"))

    sources = []

    # --- MIT, INCART, PTB, SVDB from the authoritative guard ---
    for tag, name in [("mit_bih", "MIT-BIH"), ("incart", "INCART"),
                      ("ptb", "PTB"), ("svdb", "SVDB")]:
        b, l, r = load_source(
            name,
            ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy",
            ECG_DATA / f"{tag}_processed_deploy_causal_labels.npy",
            ECG_DATA / f"{tag}_processed_deploy_causal_record_ids.npy",
        )
        mask = np.asarray(train_mask_splitguard(tag), dtype=bool)
        n = int(mask.sum())
        n_norm = int((mask & (l == 0)).sum())
        n_abn = int((mask & (l == 1)).sum())
        recs = np.unique(np.asarray(r)[mask])
        code_map = {"mit_bih": 0, "incart": 1, "ptb": 2, "svdb": 3}
        sources.append({
            "name": name,
            "tag": tag,
            "source_code": code_map[tag],
            "n_train_beats": n,
            "n_train_normal": n_norm,
            "n_train_abnormal": n_abn,
            "n_train_records": int(len(recs)),
            "record_ids": sorted(int(x) for x in recs[:2000]),
            "note": "authoritative SplitGuard train mask",
        })
        print(f"[BP] {name}: train beats={n} N={n_norm} A={n_abn} records={len(recs)}",
              flush=True)

    # --- VFDB/CUDB combined from our processed NPZ + split audit ---
    v = np.load(CACHE / "vfdb_processed_deploy_causal.npz")
    vb, vl, vr = v["beats"], v["labels"], v["record_ids"]
    v_train_ids = vfdb_split["train_record_ids"]
    vm = train_mask_record_ids(vr, v_train_ids)
    vn = int(vm.sum())
    sources.append({
        "name": "VFDB+CUDB",
        "tag": "vfdb_cudb",
        "source_code": 4,
        "n_train_beats": vn,
        "n_train_normal": int((vm & (vl == 0)).sum()),
        "n_train_abnormal": int((vm & (vl == 1)).sum()),
        "n_train_records": len(v_train_ids),
        "record_ids": v_train_ids,
        "note": "record-level split from audit_vfdb_split.py; train only",
    })
    print(f"[BP] VFDB+CUDB: train beats={vn} N={int((vm & (vl==0)).sum())} "
          f"A={int((vm & (vl==1)).sum())} records={len(v_train_ids)}", flush=True)

    # --- AFDB from our new processed NPZ + record=patient split ---
    a = np.load(CACHE / "afdb_processed_deploy_causal.npz")
    ab, al, ar = a["beats"], a["labels"], a["record_ids"]
    unique_ar = np.unique(ar)
    amap = {int(rid): f"afdb_{int(rid)}" for rid in unique_ar}
    # patient_level_split expects int rids; ar already internal ints.
    tr_mask, va_mask, te_mask, astats = patient_level_split(
        np.asarray(ar, dtype=np.int64), amap, seed=SPLIT_SEED)
    tr_mask = np.asarray(tr_mask, dtype=bool)
    an = int(tr_mask.sum())
    sources.append({
        "name": "AFDB",
        "tag": "afdb",
        "source_code": 5,
        "n_train_beats": an,
        "n_train_normal": int((tr_mask & (al == 0)).sum()),
        "n_train_abnormal": int((tr_mask & (al == 1)).sum()),
        "n_train_records": int(len(np.unique(np.asarray(ar)[tr_mask]))),
        "record_ids": sorted(int(x) for x in np.unique(np.asarray(ar)[tr_mask])),
        "note": "AFDB .qrs peaks + rhythm labels; record-level split via patient_level_split",
        "split_stats": astats,
    })
    print(f"[BP] AFDB: train beats={an} N={int((tr_mask & (al==0)).sum())} "
          f"A={int((tr_mask & (al==1)).sum())}", flush=True)

    # Summary
    total_beats = sum(s["n_train_beats"] for s in sources if s.get("n_train_beats"))
    total_norm = sum(s["n_train_normal"] for s in sources if s.get("n_train_normal"))
    total_abn = sum(s["n_train_abnormal"] for s in sources if s.get("n_train_abnormal"))
    total_records = sum(s["n_train_records"] for s in sources if s.get("n_train_records"))

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Train-safe beat pool manifest over local beat-level databases",
        "source_info": {
            "0": "MIT-BIH",
            "1": "INCART",
            "2": "PTB",
            "3": "SVDB",
            "4": "VFDB+CUDB",
            "5": "AFDB",
        },
        "sources": sources,
        "totals": {
            "train_beats": total_beats,
            "train_normal": total_norm,
            "train_abnormal": total_abn,
            "train_records": total_records,
        },
        "excluded_sources": [
            {
                "name": "LUDB",
                "reason": "QRS gold-standard annotations only; no per-beat normal/abnormal class labels for the binary anomaly task.",
            },
            {
                "name": "PTB-XL",
                "reason": "Record-level diagnosis only; not beat-level annotated for this task.",
            },
            {
                "name": "PTB",
                "reason": "Record-level diagnosis (current pipeline already uses it); included above only as record-level PTB source.",
            },
        ],
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[BP] saved {OUT}", flush=True)
    print(f"[BP] totals: beats={total_beats} N={total_norm} A={total_abn} "
          f"records={total_records}", flush=True)


if __name__ == "__main__":
    main()
