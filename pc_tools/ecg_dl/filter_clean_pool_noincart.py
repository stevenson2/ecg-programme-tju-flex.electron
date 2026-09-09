#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""filter_clean_pool_noincart.py -- create a clean hard-normal pool excluding
INCART after user visual review reported INCART is still not Lead II in the
preview.

Keeps only MIT-BIH, PTB, SVDB.  Saves filtered scores/meta and an audit.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
SCORES = CACHE / "beat_pool_clean_normal_scores.npy"
META = CACHE / "beat_pool_clean_normal_meta.npz"
OUT_SCORES = CACHE / "beat_pool_noincart_normal_scores.npy"
OUT_META = CACHE / "beat_pool_noincart_normal_meta.npz"
OUT_AUDIT = CACHE / "hard_normal_noincart_audit.json"


def main():
    scores = np.load(SCORES)
    meta = np.load(META)
    keep = meta["source"] != 1
    scores_f = scores[keep]
    meta_f = {k: v[keep] for k, v in meta.items()}
    np.save(OUT_SCORES, scores_f)
    np.savez_compressed(OUT_META, **meta_f)

    complete = meta_f["complete"] == 1
    overall = {
        "normal_total": int(len(scores_f)),
        "complete_total": int(complete.sum()),
        "hard_gt_0_5": int((scores_f > 0.5).sum()),
        "hard_gt_0_95": int((scores_f > 0.95).sum()),
        "hard_gt_0_5_complete": int(((scores_f > 0.5) & complete).sum()),
        "hard_gt_0_95_complete": int(((scores_f > 0.95) & complete).sum()),
    }
    per_source = []
    names = {0: "MIT-BIH", 2: "PTB", 3: "SVDB"}
    for code in [0, 2, 3]:
        m = meta_f["source"] == code
        c = complete & m
        src_complete = complete[m]
        overall_src = {
            "name": names[code],
            "source_code": code,
            "normal": int(m.sum()),
            "complete": int(src_complete.sum()),
            "hard_gt_0_5": int((scores_f[m] > 0.5).sum()),
            "hard_gt_0_95": int((scores_f[m] > 0.95).sum()),
            "hard_gt_0_5_complete": int(((scores_f[m] > 0.5) & src_complete).sum()),
            "hard_gt_0_95_complete": int(((scores_f[m] > 0.95) & src_complete).sum()),
        }
        per_source.append(overall_src)
        print(f"[NOINC] {names[code]}: {overall_src}", flush=True)

    audit = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Clean hard-normal pool excluding INCART due to unresolved Lead II visual review",
        "excluded_source": "INCART (source_code=1)",
        "overall": overall,
        "per_source": per_source,
        "outputs": {
            "scores": str(OUT_SCORES.relative_to(BASE)),
            "meta": str(OUT_META.relative_to(BASE)),
            "audit": str(OUT_AUDIT.relative_to(BASE)),
        },
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[NOINC] saved {OUT_SCORES.name}, {OUT_META.name}, {OUT_AUDIT.name}", flush=True)


if __name__ == "__main__":
    main()
