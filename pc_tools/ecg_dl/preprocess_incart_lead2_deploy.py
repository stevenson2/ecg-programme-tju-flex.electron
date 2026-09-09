#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preprocess_incart_lead2_deploy.py -- rebuild INCART deploy-causal arrays
using Lead II (channel index 1) instead of the historical Lead I channel.

The existing INCART pipeline used `p_signal[:,0]` (Lead I) despite comments
claiming Lead II.  This new script creates a corrected INCART Lead II dataset
under models/deploy_match/.  It does not modify any frozen script.
"""
import json
import sys
import time
from pathlib import Path
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import ecg_refs as _REFS  # noqa: E402


import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TARGET_FS, BEAT_WINDOW_SAMPLES, AAMI_CLASSES
from eval_deploy_match import corrected_deployment_chain, extract_beats_deploy

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
INCART_DIR = _REFS.INCART_DIR
OUT_NPZ = CACHE / "incart_lead2_processed_deploy_causal.npz"
OUT_AUDIT = CACHE / "incart_lead2_preprocess_audit.json"
EXCLUDE_RECORDS = {3, 73}  # user visual review: I03 and I73 not Lead II-like


def _aami_r_idx(ann_idx, ann_sym, fs):
    """AAMI-filtered R-peak indices at TARGET_FS (same as build_deploy_npz)."""
    aami_mask = np.array([s in AAMI_CLASSES for s in ann_sym])
    return (ann_idx[aami_mask] * (TARGET_FS / fs)).astype(np.int64)


def _aami_labels(ann_sym):
    syms = [s.decode() if isinstance(s, bytes) else s for s in ann_sym]
    return np.array([AAMI_CLASSES[s] for s in syms if s in AAMI_CLASSES],
                    dtype=np.int32)


def main():
    import wfdb
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_beats, all_labels, all_rec_ids = [], [], []
    counts = {}
    errors = []

    for i in range(1, 76):
        rec_name = f"I{i:02d}"
        if i in EXCLUDE_RECORDS:
            print(f"[IL2] {rec_name}: SKIP user-excluded (not Lead II-like)", flush=True)
            continue
        try:
            rec = wfdb.rdrecord(str(INCART_DIR / rec_name))
            ann = wfdb.rdann(str(INCART_DIR / rec_name), "atr")
            fs = int(rec.fs)
            lead = rec.p_signal[:, 1].astype(np.float64)  # Lead II

            labels_b = _aami_labels(ann.symbol)
            if len(labels_b) == 0:
                print(f"[IL2] {rec_name}: SKIP 0 AAMI beats")
                continue
            r_idx_250 = _aami_r_idx(np.asarray(ann.sample), ann.symbol, fs)
            deploy_250 = corrected_deployment_chain(lead, fs)

            # Refine annotation indices to actual positive R peak in the
            # deployment-filtered Lead II stream (compensates causal group delay).
            refined = []
            search = 14
            for ri in r_idx_250:
                if ri - search < 0 or ri + search + 1 > len(deploy_250):
                    refined.append(int(ri))
                    continue
                seg = deploy_250[ri - search:ri + search + 1]
                rel = int(np.argmax(seg))
                refined.append(int(ri - search + rel))
            r_idx_250 = np.asarray(refined, dtype=np.int64)

            half = BEAT_WINDOW_SAMPLES // 2
            keep = ((r_idx_250 - half >= 0) &
                    (r_idx_250 - half + BEAT_WINDOW_SAMPLES <= len(deploy_250)))
            r_idx_use = r_idx_250[keep]
            beats = extract_beats_deploy(deploy_250, r_idx_use, "incart")
            labels_use = labels_b[keep]

            if len(beats) != int(keep.sum()):
                errors.append(f"{rec_name}: deploy {len(beats)} != keep {int(keep.sum())}")
                continue

            all_beats.append(beats)
            all_labels.append(labels_use)
            all_rec_ids.append(np.full(len(beats), i, dtype=np.int32))
            counts[rec_name] = len(beats)
            print(f"[IL2] {rec_name}: {len(beats)} beats "
                  f"(N={int((labels_use==0).sum())}, A={int((labels_use==1).sum())})",
                  flush=True)
        except Exception as e:
            errors.append(f"{rec_name}: {e}")
            print(f"[IL2] {rec_name}: ERROR {e}", flush=True)

    if not all_beats:
        raise SystemExit("[IL2] no records processed")

    beats = np.concatenate(all_beats).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int32)
    rec_ids = np.concatenate(all_rec_ids).astype(np.int32)
    np.savez_compressed(OUT_NPZ, beats=beats, labels=labels, record_ids=rec_ids)

    audit = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "INCART Lead II deploy-causal rebuild",
        "channel": "II (p_signal[:,1])",
        "source_dir": str(INCART_DIR),
        "n_records_processed": len(counts),
        "n_errors": len(errors),
        "total_beats": int(len(beats)),
        "total_normal": int((labels == 0).sum()),
        "total_abnormal": int((labels == 1).sum()),
        "per_record_counts": counts,
        "errors": errors,
        "output": str(OUT_NPZ.relative_to(BASE)),
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"[IL2] saved {OUT_NPZ.name}: beats={len(beats)} "
          f"N={int((labels==0).sum())} A={int((labels==1).sum())} "
          f"errors={len(errors)}", flush=True)


if __name__ == "__main__":
    main()
