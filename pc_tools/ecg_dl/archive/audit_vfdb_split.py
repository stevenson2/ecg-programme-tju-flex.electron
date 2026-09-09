#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_vfdb_split.py -- patient/record-level split audit for VFDB/CUDB.

Uses the same patient_level_split core as data/split_guard.py (seed=42,
60/20/20) but does NOT modify split_guard.py. Each VFDB/CUDB record is treated
as one patient (conservative record-level split).  The output records the
train/val/test masks and verifies zero train/test, train/val, val/test overlap.

No training.  No test-set contact.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.patient_split import patient_level_split, SEED as SPLIT_SEED

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
NPZ = CACHE / "vfdb_processed_deploy_causal.npz"
OUT = CACHE / "vfdb_split_audit.json"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = np.load(NPZ)
    record_ids = np.asarray(d["record_ids"]).astype(np.int64)
    labels = np.asarray(d["labels"]).astype(np.int32)

    # Internal ids are 0..n_records-1; map each to a distinct "patient" so no
    # two records are treated as the same patient (conservative).
    unique_rids = np.unique(record_ids)
    pmap = {int(rid): f"vfdb_rec_{int(rid)}" for rid in unique_rids}
    train_mask, val_mask, test_mask, stats = patient_level_split(
        record_ids, pmap, seed=SPLIT_SEED)

    # Zero-overlap assertions.
    overlaps = {
        "train_val": int((train_mask & val_mask).sum()),
        "train_test": int((train_mask & test_mask).sum()),
        "val_test": int((val_mask & test_mask).sum()),
    }
    coverage = int((train_mask | val_mask | test_mask).sum())
    n_records = len(record_ids)
    ok = (overlaps["train_val"] == 0 and
          overlaps["train_test"] == 0 and
          overlaps["val_test"] == 0 and
          coverage == n_records)

    def class_counts(mask):
        return {
            "beats": int(mask.sum()),
            "normal": int((mask & (labels == 0)).sum()),
            "abnormal": int((mask & (labels == 1)).sum()),
            "excluded": int((mask & (labels == -1)).sum()),
        }

    # The arrays currently exclude excluded beats (label -1 not saved);
    # count only 0/1 as stored.
    splits = {
        "train": class_counts(train_mask),
        "val": class_counts(val_mask),
        "test": class_counts(test_mask),
    }

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "VFDB/CUDB record-level split audit using patient_level_split core",
        "seed": SPLIT_SEED,
        "split_rule": "60/20/20 patient-level; each record = one patient (conservative)",
        "source": str(NPZ.relative_to(BASE)),
        "n_records_with_beats": int(len(unique_rids)),
        "n_beats_total": int(len(record_ids)),
        "overlaps": overlaps,
        "coverage": {"covered": coverage, "total": n_records, "ok": coverage == n_records},
        "split_ok": ok,
        "stats": stats,
        "splits": splits,
        "patient_map": {str(k): v for k, v in sorted(pmap.items())},
        "train_record_ids": sorted(int(r) for r in unique_rids[train_mask[:len(unique_rids)]]),
        "val_record_ids": sorted(int(r) for r in unique_rids[val_mask[:len(unique_rids)]]),
        "test_record_ids": sorted(int(r) for r in unique_rids[test_mask[:len(unique_rids)]]),
        "elapsed_s": round(time.time() - t0, 1),
    }

    # Fix: masks are per-beat; use unique rids per split properly.
    tr_rids = np.unique(record_ids[train_mask])
    va_rids = np.unique(record_ids[val_mask])
    te_rids = np.unique(record_ids[test_mask])
    report["train_record_ids"] = sorted(int(r) for r in tr_rids)
    report["val_record_ids"] = sorted(int(r) for r in va_rids)
    report["test_record_ids"] = sorted(int(r) for r in te_rids)
    report["split_record_counts"] = {
        "train": len(tr_rids), "val": len(va_rids), "test": len(te_rids),
        "union": len(set(tr_rids) | set(va_rids) | set(te_rids)),
    }
    report["record_split_ok"] = (
        len(set(tr_rids) & set(va_rids)) == 0 and
        len(set(tr_rids) & set(te_rids)) == 0 and
        len(set(va_rids) & set(te_rids)) == 0 and
        len(set(tr_rids) | set(va_rids) | set(te_rids)) == len(unique_rids)
    )

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[VFS] saved {OUT}", flush=True)
    print(f"[VFS] records={len(unique_rids)} beats={len(record_ids)} "
          f"train={len(tr_rids)} val={len(va_rids)} test={len(te_rids)} "
          f"overlaps={overlaps} ok={ok and report['record_split_ok']}",
          flush=True)
    for name, c in splits.items():
        print(f"[VFS] {name}: beats={c['beats']} N={c['normal']} A={c['abnormal']}",
              flush=True)


if __name__ == "__main__":
    main()
