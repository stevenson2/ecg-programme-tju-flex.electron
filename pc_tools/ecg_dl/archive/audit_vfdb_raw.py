#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_vfdb_raw.py -- VFDB/CUDB raw data feasibility audit (no training).

Reads the PhysioNet VFDB + CUDB raw directory (/home/devcontainers/vfdb) and
writes a metadata/label audit JSON under models/deploy_match/.  This is the
first step of the VFDB route: before any preprocessing/split/training we must
know what is actually on disk, what annotation vocabulary exists, and whether
the data can plausibly supply additional normal/abnormal beats for the 250-point
single-lead pipeline.

No test-set contact, no model evaluation, no training.  This script only reads
raw WFDB headers and annotations.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import wfdb

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
VFDB_DIR = Path("/home/devcontainers/vfdb")
OUT_JSON = CACHE / "vfdb_raw_audit.json"


def clean_note(s):
    if s is None:
        return ""
    return str(s).replace("\x00", "").strip()


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    dat_files = sorted(VFDB_DIR.glob("*.dat"))
    if not dat_files:
        raise SystemExit("[VFA] no .dat files found in VFDB_DIR")

    records = []
    summary = {
        "n_dat_files": len(dat_files),
        "n_vfdb": 0,
        "n_cudb": 0,
        "fs_values": {},
        "sig_names": {},
        "units": {},
        "total_duration_s": 0.0,
        "annotation_present": 0,
        "annotation_missing": 0,
        "aoi_notes": {},
        "symbol_counts": {},
    }

    for dat in dat_files:
        stem = dat.stem
        is_cudb = stem.startswith("cu")
        try:
            rec = wfdb.rdrecord(str(dat.with_suffix("")), channels=[0])
            fs = int(rec.fs)
            dur_s = float(len(rec.p_signal) / fs)
            sig_name = list(rec.sig_name) if rec.sig_name else []
            units = list(rec.units) if rec.units else []
        except Exception as e:
            records.append({
                "record": stem, "type": "cudb" if is_cudb else "vfdb",
                "error": f"rdrecord failed: {e}",
            })
            continue

        rec_row = {
            "record": stem,
            "type": "cudb" if is_cudb else "vfdb",
            "fs": fs,
            "n_sig": int(rec.n_sig),
            "sig_name": sig_name,
            "units": units,
            "n_samples": int(len(rec.p_signal)),
            "duration_s": dur_s,
            "duration_min": round(dur_s / 60.0, 2),
        }

        atr_path = dat.with_suffix(".atr")
        if not atr_path.exists():
            rec_row["annotation"] = None
            summary["annotation_missing"] += 1
        else:
            try:
                ann = wfdb.rdann(str(dat.with_suffix("")), "atr")
                notes = [clean_note(x) for x in (ann.aux_note or [])]
                syms = list(ann.symbol)
                rec_row["annotation"] = {
                    "n_annotations": int(len(ann.sample)),
                    "symbol_counts": dict(sorted(
                        {s: int(syms.count(s)) for s in set(syms)}.items(),
                        key=lambda kv: -kv[1])),
                    "note_counts": dict(sorted(
                        {n: int(notes.count(n)) for n in set(notes) if n}.items(),
                        key=lambda kv: -kv[1])),
                    "first_notes": notes[:10],
                }
                summary["annotation_present"] += 1
                for s, c in rec_row["annotation"]["symbol_counts"].items():
                    summary["symbol_counts"][s] = summary["symbol_counts"].get(s, 0) + c
                for n, c in rec_row["annotation"]["note_counts"].items():
                    summary["aoi_notes"][n] = summary["aoi_notes"].get(n, 0) + c
            except Exception as e:
                rec_row["annotation"] = {"error": f"rdann failed: {e}"}
                summary["annotation_missing"] += 1

        if is_cudb:
            summary["n_cudb"] += 1
        else:
            summary["n_vfdb"] += 1

        summary["fs_values"][str(fs)] = summary["fs_values"].get(str(fs), 0) + 1
        summary["total_duration_s"] += dur_s
        for s in sig_name:
            summary["sig_names"][s] = summary["sig_names"].get(s, 0) + 1
        for u in units:
            summary["units"][u] = summary["units"].get(u, 0) + 1

        records.append(rec_row)

    summary["total_duration_h"] = round(summary["total_duration_s"] / 3600.0, 2)
    summary["total_records_ok"] = sum(1 for r in records if "error" not in r)

    # Also list the simple RECORDS listing files if present.
    for name in ("RECORDS", "CUDB_RECORDS"):
        f = VFDB_DIR / name
        summary[name] = None
        if f.exists():
            lines = [line.strip() for line in f.read_text(errors="replace").splitlines() if line.strip()]
            summary[name] = {"n_lines": len(lines), "first": lines[:10], "last": lines[-10:]}

    # Record-number style overlap notes with existing clean-split families.
    # The raw VFDB/CUDB record IDs are a different naming family from the
    # MIT/INCART/PTB splits; this is a metadata-level statement only.
    vfdb_ids = sorted({r["record"] for r in records if r.get("type") == "vfdb"})
    cudb_ids = sorted({r["record"] for r in records if r.get("type") == "cudb"})
    summary["record_ids"] = {"vfdb": vfdb_ids, "cudb": cudb_ids}
    summary["id_family_note"] = (
        "VFDB IDs (418-430, 602-615) and CUDB IDs (cu01-cu35) are PhysioNet "
        "records unrelated by record-number family to MIT-BIH (100-124), "
        "INCART (numerous integer records), or PTB (patientNNN). This is not "
        "independent proof of zero patient overlap; patient-level overlap must "
        "be addressed by dataset provenance (separate PhysioNet collections)."
    )

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "VFDB/CUDB raw feasibility audit stage 1: metadata + annotation vocabulary only",
        "source_dir": str(VFDB_DIR),
        "zero_test_contact": True,
        "training_performed": False,
        "summary": summary,
        "records": records,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    print(f"[VFA] wrote {OUT_JSON}", flush=True)
    print(f"[VFA] n_dat={len(dat_files)} vfdb={summary['n_vfdb']} "
          f"cudb={summary['n_cudb']} total_h={summary['total_duration_h']} "
          f"fs={summary['fs_values']} sig={summary['sig_names']} "
          f"units={summary['units']}", flush=True)
    print(f"[VFA] top annotation notes: "
          f"{sorted(summary['aoi_notes'].items(), key=lambda kv: -kv[1])[:15]}",
          flush=True)


if __name__ == "__main__":
    main()
