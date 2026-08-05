#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build deploy-chain 3-beat dataset (mit_incart_3beat_deploy.npz).

Stitches *_deploy single-beat windows (MIT 6x-augmented, INCART raw) into
750-pt 3-beat sequences per record, keeping the center-beat label — same
semantics as data/preprocess_3beat.stitch_3beat.

Note: MIT beats are 6x block-tiled (raw,noise,scale,scale,drift,drift).
Stitching on the augmented array keeps label copies consistent (center label
copied with the augmented center beat), so per-AAMI alignment via the same
tile structure stays valid.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR
from data.dataset import set_npz_suffix
from data.preprocess_3beat import stitch_3beat

set_npz_suffix("_deploy")
from data.dataset import load_mit_incart_merged

data = load_mit_incart_merged()
beats, labels, rids = data["beats"], data["labels"], data["record_ids"]
print(f"Source: {len(beats)} beats ({len(np.unique(rids))} records)")

beats_3, labels_3, record_3 = stitch_3beat(beats, labels, rids)
out = PROCESSED_DIR / "mit_incart_3beat_deploy.npz"
np.savez_compressed(out, beats=beats_3, labels=labels_3, record_ids=record_3)
print(f"Saved: {out}  ({len(beats_3)} sequences)")
n_abn = int((labels_3 == 1).sum())
print(f"  labels: N={len(labels_3)-n_abn:,}  A={n_abn:,}  "
      f"(abn {n_abn/len(labels_3)*100:.1f}%)")
