#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_incart_lead2_quality.py -- per-record Lead II quality/polarity audit
for INCART.  Helps identify which records actually show Lead II-like features.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TARGET_FS
from eval_deploy_match import corrected_deployment_chain

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
INCART_DIR = Path(
    "/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/"
    "ecg-programme-tju-flex.electron-master/"
    "st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/files"
)
OUT = CACHE / "incart_lead2_quality_audit.json"
SEARCH = 14
MAX_BEATS = 200


def main():
    t0 = time.time()
    rows = []
    for i in range(1, 76):
        rec_name = f"I{i:02d}"
        try:
            rec = wfdb.rdrecord(str(INCART_DIR / rec_name))
            ann = wfdb.rdann(str(INCART_DIR / rec_name), "atr")
            fs = int(rec.fs)
            lead2 = rec.p_signal[:, 1].astype(np.float64)
            filtered = corrected_deployment_chain(lead2, fs)

            # Sample up to MAX_BEATS annotations, refine to positive R peak.
            samples = np.asarray(ann.sample, dtype=np.int64)
            peaks = []
            r_amps = []
            base = np.median(filtered[int(10*TARGET_FS):int(20*TARGET_FS)])
            use = samples[:MAX_BEATS]
            for s in use:
                idx = int(s * TARGET_FS / fs)
                if idx - SEARCH < 0 or idx + SEARCH + 1 > len(filtered):
                    continue
                seg = filtered[idx - SEARCH:idx + SEARCH + 1]
                rel = int(np.argmax(seg))
                center = idx - SEARCH + rel
                peak_val = seg[rel]
                peaks.append(peak_val)
                # local baseline around R ± 40-60 samples (before QRS)
                lo = max(0, center - 60)
                hi = max(0, center - 25)
                local_base = np.median(filtered[lo:hi]) if hi > lo else base
                r_amps.append(peak_val - local_base)
            peaks = np.asarray(peaks)
            r_amps = np.asarray(r_amps)
            frac_pos = float((peaks > 0).mean()) if len(peaks) else np.nan
            rows.append({
                "record": rec_name,
                "n_beats_sampled": int(len(peaks)),
                "frac_positive_peak": frac_pos,
                "median_peak_mv": float(np.median(peaks)) if len(peaks) else None,
                "median_r_amp_mv": float(np.median(r_amps)) if len(r_amps) else None,
                "mean_r_amp_mv": float(np.mean(r_amps)) if len(r_amps) else None,
            })
        except Exception as e:
            rows.append({"record": rec_name, "error": str(e)})
        if i % 10 == 0:
            print(f"[ILQ] {i}/75", flush=True)

    # Sort by a simple Lead II-likeness heuristic: positive R frac then amplitude.
    ok_rows = [r for r in rows if "frac_positive_peak" in r and r["n_beats_sampled"] > 0]
    ok_rows.sort(key=lambda r: (-r["frac_positive_peak"], -r["median_r_amp_mv"]))
    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "INCART Lead II per-record polarity/amplitude audit",
        "channel": "II (p_signal[:,1])",
        "records": rows,
        "sorted_by_lead2_likeness": ok_rows,
        "summary": {
            "n_records": len(rows),
            "frac_positive_peak_ge_0.8": int(sum(1 for r in ok_rows if r["frac_positive_peak"] >= 0.8)),
            "frac_positive_peak_lt_0.8": int(sum(1 for r in ok_rows if r["frac_positive_peak"] < 0.8)),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ILQ] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
