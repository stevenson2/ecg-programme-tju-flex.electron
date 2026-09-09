#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preprocess_afdb_deploy.py -- AFDB deploy-causal beat extraction audit.

AFDB has long 250 Hz single-lead recordings, .qrs beat annotations and .atr
rhythm annotations.  This script builds 250-point deploy-causal beat windows
with provisional binary labels from the rhythm segments.

No training, no test contact.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BEAT_WINDOW_SAMPLES
from eval_deploy_match import corrected_deployment_chain, extract_beats_deploy

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
AFDB_DIR = Path("/home/devcontainers/afdb_wfdb")
OUT_NPZ = CACHE / "afdb_processed_deploy_causal.npz"
OUT_AUDIT = CACHE / "afdb_preprocess_audit.json"

AFDB_NORMAL_NOTES = {"N"}
AFDB_EXCLUDE_NOTES = {"NOISE", "ASYS", ""}


def clean_note(s):
    if s is None:
        return ""
    return "".join(ch for ch in str(s) if ch.isalnum() or ch in "_").replace("\x00", "").strip()


def label_for_afdb_time(t, ann, fs):
    if ann is None or len(ann.sample) == 0:
        return -1, "NO_ANNOTATION"
    times = np.asarray(ann.sample, dtype=np.float64) / fs
    idx = int(np.searchsorted(times, t, side="right") - 1)
    if idx < 0:
        idx = 0
    note = clean_note(ann.aux_note[idx]) if ann.aux_note else ""
    if note in AFDB_NORMAL_NOTES:
        return 0, note
    if note in AFDB_EXCLUDE_NOTES:
        return -1, note
    if not note:
        return -1, "EMPTY_NOTE"
    return 1, note


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    import wfdb
    dat_files = sorted(AFDB_DIR.glob("*.dat"))
    all_beats, all_labels, all_rec_ids = [], [], []
    per_record = []
    label_counts = {"normal": 0, "abnormal": 0, "excluded": 0}
    record_id_counter = 0

    for i, dat in enumerate(dat_files):
        stem = dat.stem
        r0 = time.time()
        try:
            rec = wfdb.rdrecord(str(dat.with_suffix("")), channels=[0])
            qrs = wfdb.rdann(str(dat.with_suffix("")), "qrs")
            ann = None
            if dat.with_suffix(".atr").exists():
                ann = wfdb.rdann(str(dat.with_suffix("")), "atr")
            fs = int(rec.fs)
            sig = rec.p_signal[:, 0].astype(np.float64)
            chain = corrected_deployment_chain(sig, fs)
            r_idx = np.asarray(qrs.sample, dtype=np.int64)
            half = BEAT_WINDOW_SAMPLES // 2
            keep = ((r_idx - half >= 0) &
                    (r_idx - half + BEAT_WINDOW_SAMPLES <= len(chain)))
            r_idx_use = r_idx[keep]
            beats = extract_beats_deploy(chain, r_idx_use, "ptb")
            labels = []
            notes = []
            for ri in r_idx_use:
                t = float(ri) / 250.0
                lab, note = label_for_afdb_time(t, ann, fs)
                labels.append(lab)
                notes.append(note)
            labels = np.asarray(labels, dtype=np.int32)
            keep2 = labels >= 0
            n_norm = int((labels == 0).sum())
            n_abn = int((labels == 1).sum())
            n_excl = int((labels == -1).sum())
            label_counts["normal"] += n_norm
            label_counts["abnormal"] += n_abn
            label_counts["excluded"] += n_excl
            if n_norm + n_abn > 0:
                all_beats.append(beats[keep2])
                all_labels.append(labels[keep2])
                all_rec_ids.append(np.full(int(keep2.sum()), record_id_counter, dtype=np.int32))
                record_id_counter += 1
            per_record.append({
                "record": stem,
                "fs": fs,
                "n_qrs": int(len(r_idx)),
                "n_beats": int(len(beats)),
                "n_normal": n_norm,
                "n_abnormal": n_abn,
                "n_excluded": n_excl,
                "notes_top": sorted(
                    {note: int(notes.count(note)) for note in set(notes)}.items(),
                    key=lambda kv: -kv[1])[:8],
                "elapsed_s": round(time.time() - r0, 1),
            })
            print(f"[AFP] {stem}: qrs={len(r_idx)} beats={len(beats)} "
                  f"N={n_norm} A={n_abn} X={n_excl} ({time.time()-r0:.0f}s)",
                  flush=True)
        except Exception as e:
            per_record.append({"record": stem, "error": str(e)})
            print(f"[AFP] {stem}: ERROR {e}", flush=True)
        if (i + 1) % 5 == 0:
            print(f"[AFP] progress {i+1}/{len(dat_files)}", flush=True)

    if all_beats:
        beats = np.concatenate(all_beats).astype(np.float32)
        labels = np.concatenate(all_labels).astype(np.int32)
        rec_ids = np.concatenate(all_rec_ids).astype(np.int32)
    else:
        beats = np.zeros((0, 250), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int32)
        rec_ids = np.zeros((0,), dtype=np.int32)
    np.savez_compressed(OUT_NPZ, beats=beats, labels=labels, record_ids=rec_ids)
    audit = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "AFDB deploy-causal feasibility build with .qrs peaks + rhythm labels",
        "label_map": {
            "normal_notes": sorted(AFDB_NORMAL_NOTES),
            "exclude_notes": sorted(AFDB_EXCLUDE_NOTES),
            "abnormal": "all other non-excluded rhythm notes (AFIB/AFL/J etc)",
        },
        "records_processed": len(dat_files),
        "records_with_error": sum(1 for r in per_record if "error" in r),
        "total_beats": int(len(beats)),
        "label_counts": label_counts,
        "per_record": per_record,
        "output": str(OUT_NPZ.relative_to(BASE)),
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[AFP] saved {OUT_NPZ.name}, total_beats={len(beats)}, "
          f"N={label_counts['normal']} A={label_counts['abnormal']} "
          f"X={label_counts['excluded']}", flush=True)


if __name__ == "__main__":
    main()
