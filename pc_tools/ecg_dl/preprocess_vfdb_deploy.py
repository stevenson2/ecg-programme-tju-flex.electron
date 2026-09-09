#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preprocess_vfdb_deploy.py -- VFDB/CUDB deploy-causal beat extraction audit.

This is the second step of the VFDB route: raw metadata was already audited.
This script:
  1. loads each VFDB/CUDB raw record (lead 0, mV, 250 Hz),
  2. applies the same corrected deployment chain used by the clean pipeline,
  3. detects R peaks with XQRS,
  4. extracts 250-point strict-edge beats with firmware z-score,
  5. assigns provisional binary labels from rhythm/beat annotations,
  6. writes arrays + a per-record audit JSON under models/deploy_match/.

No training, no test-set contact, no modification of frozen scripts.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BEAT_WINDOW_SAMPLES
from eval_deploy_match import corrected_deployment_chain, extract_beats_deploy
from data.preprocess_ptb import detect_r_peaks

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
VFDB_DIR = Path("/home/devcontainers/vfdb")
OUT_NPZ = CACHE / "vfdb_processed_deploy_causal.npz"
OUT_AUDIT = CACHE / "vfdb_preprocess_audit.json"

# Provisional label map for feasibility audit only.
# Final label map must be re-confirmed in vfdb_calibration_prereg.json.
VFDB_NORMAL_NOTES = {"N", "NSR"}
VFDB_EXCLUDE_NOTES = {"NOISE", "ASYS"}
# Everything else with a rhythm annotation is treated as abnormal in this
# provisional audit. This is a conservative feasibility count, not a final
# training-label decision.
CUDB_EXCLUDE_AFTER = True  # after the closing ']' of a CUDB VF episode, drop


def clean_note(s):
    if s is None:
        return ""
    # WFDB rhythm notes look like "(N", "(VT", "(NOISE"; reduce to bare code.
    return ("".join(ch for ch in str(s) if ch.isalnum() or ch in "_")
            .replace("\x00", "").strip())


def label_for_vfdb_time(t, ann, fs):
    """Return (label, note) for a beat center time in a VFDB record.

    label: 0 normal, 1 abnormal, -1 excluded.
    Uses the last annotation whose sample <= t (seconds); before the first
    annotation the first note is used.
    """
    if ann is None or len(ann.sample) == 0:
        return -1, "NO_ANNOTATION"
    note = clean_note(ann.aux_note[0]) if ann.aux_note else ""
    # t in seconds
    times = np.asarray(ann.sample, dtype=np.float64) / fs
    idx = int(np.searchsorted(times, t, side="right") - 1)
    if idx < 0:
        idx = 0
    note = clean_note(ann.aux_note[idx]) if ann.aux_note else ""
    if note in VFDB_NORMAL_NOTES:
        return 0, note
    if note in VFDB_EXCLUDE_NOTES:
        return -1, note
    # Unknown/empty note: do not silently include as normal.
    if not note:
        return -1, "EMPTY_NOTE"
    return 1, note


def label_for_cudb_time(t, ann, fs):
    """CUDB: use '[' and ']' annotations as VF episode boundaries.

    Before '[' -> 0 (normal), inside [ '[' , ']' ] -> 1 (abnormal),
    after ']' -> -1 (excluded, usually nothing).
    """
    if ann is None or len(ann.sample) == 0:
        return -1, "NO_ANNOTATION"
    times = np.asarray(ann.sample, dtype=np.float64) / fs
    syms = list(ann.symbol)
    open_t = None
    close_t = None
    for smp, sym in zip(times, syms):
        if sym == "[":
            open_t = smp
        elif sym == "]":
            close_t = smp
    if open_t is None:
        return -1, "NO_VF_OPEN"
    if t < open_t:
        return 0, "PRE_VF"
    if close_t is not None and t > close_t:
        return -1, "POST_VF"
    return 1, "VF"


def process_record(dat_path):
    import wfdb
    stem = dat_path.stem
    is_cudb = stem.startswith("cu")
    rec = wfdb.rdrecord(str(dat_path.with_suffix("")), channels=[0])
    fs = int(rec.fs)
    sig = rec.p_signal[:, 0].astype(np.float64)

    # Deployment-causal 250 Hz stream (same as clean pipeline).
    chain = corrected_deployment_chain(sig, fs)

    # XQRS R-peak detection on the deploy chain (same as PTB/real-AFE usage).
    r_idx = np.asarray(detect_r_peaks(chain), dtype=np.int64)

    # Strict-edge keep mask (same as extract_beats_deploy domain='ptb').
    half = BEAT_WINDOW_SAMPLES // 2
    keep = ((r_idx - half >= 0) &
            (r_idx - half + BEAT_WINDOW_SAMPLES <= len(chain)))
    r_idx_use = r_idx[keep]

    ann = None
    if dat_path.with_suffix(".atr").exists():
        try:
            ann = wfdb.rdann(str(dat_path.with_suffix("")), "atr")
        except Exception as e:
            ann = None

    beats = extract_beats_deploy(chain, r_idx_use, "ptb")
    labels = []
    notes = []
    for ri in r_idx_use:
        # Use beat center on original time scale, not shifted window.
        t = float(ri) / 250.0
        if is_cudb:
            lab, note = label_for_cudb_time(t, ann, fs)
        else:
            lab, note = label_for_vfdb_time(t, ann, fs)
        labels.append(lab)
        notes.append(note)
    labels = np.asarray(labels, dtype=np.int32)
    return stem, is_cudb, beats, labels, notes, int(len(r_idx)), int(fs)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dat_files = sorted(VFDB_DIR.glob("*.dat"))
    all_beats, all_labels, all_rec_ids, all_notes = [], [], [], []
    per_record = []
    label_counts = {"normal": 0, "abnormal": 0, "excluded": 0}

    for i, dat in enumerate(dat_files):
        r0 = time.time()
        try:
            stem, is_cudb, beats, labels, notes, n_peaks, fs = process_record(dat)
        except Exception as e:
            per_record.append({"record": dat.stem, "error": str(e)})
            print(f"[VFP] {dat.stem}: ERROR {e}", flush=True)
            continue
        n = len(beats)
        n_norm = int((labels == 0).sum())
        n_abn = int((labels == 1).sum())
        n_excl = int((labels == -1).sum())
        label_counts["normal"] += n_norm
        label_counts["abnormal"] += n_abn
        label_counts["excluded"] += n_excl
        if n_norm + n_abn > 0:
            all_beats.append(beats)
            # Keep excluded beats out of arrays; provenance records original index.
            keep = labels >= 0
            all_labels.append(labels[keep])
            all_rec_ids.append(np.full(int(keep.sum()), len(all_rec_ids), dtype=np.int32))
            # We append only kept note strings; not used beyond audit.
        per_record.append({
            "record": stem,
            "type": "cudb" if is_cudb else "vfdb",
            "fs": fs,
            "n_xqrs_peaks": int(n_peaks),
            "n_beats_extracted": n,
            "n_normal": n_norm,
            "n_abnormal": n_abn,
            "n_excluded": n_excl,
            "notes_top": sorted(
                {note: int(notes.count(note)) for note in set(notes)}.items(),
                key=lambda kv: -kv[1])[:8],
            "elapsed_s": round(time.time() - r0, 1),
        })
        print(f"[VFP] {stem}: peaks={n_peaks} beats={n} "
              f"N={n_norm} A={n_abn} X={n_excl} "
              f"({time.time()-r0:.0f}s)", flush=True)
        if (i + 1) % 10 == 0:
            print(f"[VFP] progress {i+1}/{len(dat_files)}", flush=True)

    all_beats = np.concatenate(all_beats).astype(np.float32)
    all_labels = np.concatenate(all_labels).astype(np.int32)
    all_rec_ids = np.concatenate(all_rec_ids).astype(np.int32)
    np.savez_compressed(
        OUT_NPZ,
        beats=all_beats,
        labels=all_labels,
        record_ids=all_rec_ids,
    )
    audit = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "VFDB/CUDB deploy-causal feasibility build: beats + provisional labels",
        "provisional_label_map": {
            "vfdb_normal_notes": sorted(VFDB_NORMAL_NOTES),
            "vfdb_abnormal_notes": "all non-normal non-excluded rhythm notes",
            "vfdb_exclude_notes": sorted(VFDB_EXCLUDE_NOTES),
            "cudb_fepisode": "'['->']' abnormal, pre-'[' normal, post-']' excluded",
            "note": "THIS IS NOT A FINAL TRAINING LABEL DECISION; confirm in prereg",
        },
        "records_processed": len(per_record),
        "records_with_error": sum(1 for r in per_record if "error" in r),
        "total_beats": int(len(all_beats)),
        "label_counts": label_counts,
        "per_record": per_record,
        "outputs": {
            "npz": str(OUT_NPZ.relative_to(BASE)),
            "audit_json": str(OUT_AUDIT.relative_to(BASE)),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"[VFP] saved {OUT_NPZ.name} + {OUT_AUDIT.name}", flush=True)
    print(f"[VFP] total beats={len(all_beats)} "
          f"N={label_counts['normal']} A={label_counts['abnormal']} "
          f"X={label_counts['excluded']}", flush=True)


if __name__ == "__main__":
    main()
