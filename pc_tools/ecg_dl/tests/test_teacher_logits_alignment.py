#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alignment test for precompute_teacher_logits.py.

Rebuilds the same arrays via the same imports, then asserts:
  1. npy row counts == (len(a_train), len(a_val), len(b_train))
  2. Teacher re-predicted logits for rows [0:16] match npy rows (atol=1e-5)
  3. Manifest sha1 matches each npy file

Plain-assert script: prints per-test status, sys.exit(1) on first
failure, sys.exit(0) when all pass. Run from the ecg_dl directory:

    cd pc_tools/ecg_dl
    python3 tests/test_teacher_logits_alignment.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Allow running this file directly: tests/ sits next to data/.
_HERE = Path(__file__).resolve().parent
_ECG_DL = _HERE.parent
if str(_ECG_DL) not in sys.path:
    sys.path.insert(0, str(_ECG_DL))

import numpy as np  # noqa: E402

from data.dataset import (  # noqa: E402
    set_npz_suffix,
    load_mit_incart_merged,
    load_ptb_data,
    add_channel_dim,
)
from data.patient_split import (  # noqa: E402
    build_mit_patient_map,
    build_incart_patient_map,
    build_ptb_patient_map,
    patient_level_split,
)

# ---------------------------------------------------------------------------
# Config — must match the precompute invocation you want to verify.
# Adjust TEACHER / PREFIX / PTB_ABN_MAX if you used different flags.
# ---------------------------------------------------------------------------
MODELS_DIR = _ECG_DL / "models"
TEACHER = MODELS_DIR / "final_ssl_finetuned.h5"
PREFIX = "teacher_logits_ssl"
EPS = 1e-7
BATCH = 1024
PTB_ABN_MAX = 10000
CHECK_ROWS = 16  # how many leading rows to spot-check logits

# Only the logit arrays (not labels) for sha1 / row-count checks
LOGIT_KEYS = ("a_train", "a_val", "b_train")
# All keys including labels
ALL_KEYS = ("a_train", "a_val", "b_train",
            "a_train_labels", "a_val_labels", "b_train_labels")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _pass(msg: str) -> None:
    print(f"  PASS: {msg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Teacher-logits alignment test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load manifest
    # ------------------------------------------------------------------
    manifest_path = MODELS_DIR / f"{PREFIX}_manifest.json"
    if not manifest_path.exists():
        _fail(f"Manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"Loaded manifest: {manifest_path.name}")

    # ------------------------------------------------------------------
    # Load saved npy arrays
    # ------------------------------------------------------------------
    saved: dict[str, np.ndarray] = {}
    for key in ALL_KEYS:
        npy_path = MODELS_DIR / f"{PREFIX}_{key}.npy"
        if not npy_path.exists():
            _fail(f"Missing npy: {npy_path}")
        saved[key] = np.load(str(npy_path))
    print(f"Loaded {len(saved)} npy arrays")

    # ------------------------------------------------------------------
    # Rebuild the exact same splits (mirror precompute logic)
    # ------------------------------------------------------------------
    print("\n--- Rebuilding splits ---")
    set_npz_suffix("_deploy")

    # A-domain
    data = load_mit_incart_merged()
    beats = data["beats"]
    labels = data["labels"]
    record_ids = data["record_ids"]

    _pmap: dict = {}
    _pmap.update(build_mit_patient_map())
    _pmap.update({
        rid + 100000: "inc_" + pat
        for rid, pat in build_incart_patient_map().items()
    })
    tr_m, va_m, te_m, pstats = patient_level_split(record_ids, _pmap)
    a_train_beats = beats[tr_m]
    a_train_labels = labels[tr_m]
    a_val_beats = beats[va_m]
    a_val_labels = labels[va_m]

    # B-domain
    ptb = load_ptb_data()
    _trp, _, _, _ps = patient_level_split(
        ptb["record_ids"], build_ptb_patient_map()
    )
    ptb_beats = ptb["beats"][_trp]
    ptb_labels = ptb["labels"][_trp]

    mask_n = ptb_labels == 0
    ptb_normals = ptb_beats[mask_n]
    ptb_normal_labels = ptb_labels[mask_n]

    idx_a = np.where(ptb_labels == 1)[0]
    if len(idx_a) > PTB_ABN_MAX:
        rng = np.random.default_rng(42)
        idx_a = rng.choice(idx_a, PTB_ABN_MAX, replace=False)
    ptb_abnormals = ptb_beats[idx_a]
    ptb_abnormal_labels = ptb_labels[idx_a]

    b_train_beats = np.concatenate([ptb_normals, ptb_abnormals], axis=0)
    b_train_labels_arr = np.concatenate(
        [ptb_normal_labels, ptb_abnormal_labels], axis=0
    )

    # ------------------------------------------------------------------
    # Test 1: Row counts match
    # ------------------------------------------------------------------
    print("\n--- Test 1: Row counts ---")
    expected_counts = {
        "a_train": len(a_train_beats),
        "a_val": len(a_val_beats),
        "b_train": len(b_train_beats),
        "a_train_labels": len(a_train_labels),
        "a_val_labels": len(a_val_labels),
        "b_train_labels": len(b_train_labels_arr),
    }
    for key, expected in expected_counts.items():
        actual = len(saved[key])
        if actual != expected:
            _fail(f"{key}: expected {expected} rows, got {actual}")
        _pass(f"{key}: {actual} rows == {expected}")

    # ------------------------------------------------------------------
    # Test 2: Label arrays match
    # ------------------------------------------------------------------
    print("\n--- Test 2: Label arrays ---")
    label_checks = [
        ("a_train_labels", a_train_labels),
        ("a_val_labels", a_val_labels),
        ("b_train_labels", b_train_labels_arr),
    ]
    for key, expected_arr in label_checks:
        if not np.array_equal(saved[key], expected_arr):
            _fail(f"{key}: array mismatch")
        _pass(f"{key}: arrays equal ({len(expected_arr)} elements)")

    # ------------------------------------------------------------------
    # Test 3: Re-predict logits for first CHECK_ROWS, compare
    # ------------------------------------------------------------------
    print("\n--- Test 3: Logit spot-check (first {CHECK_ROWS} rows) ---")
    import tensorflow as tf  # noqa: E402

    if not TEACHER.exists():
        _fail(f"Teacher model not found: {TEACHER}")
    teacher = tf.keras.models.load_model(str(TEACHER), compile=False)

    beat_arrays = {
        "a_train": a_train_beats,
        "a_val": a_val_beats,
        "b_train": b_train_beats,
    }
    for key in LOGIT_KEYS:
        beats_arr = beat_arrays[key]
        x = add_channel_dim(beats_arr[:CHECK_ROWS].astype(np.float32))
        p_t = teacher.predict(x, batch_size=BATCH, verbose=0)
        z = np.log(np.clip(p_t, EPS, 1.0 - EPS))
        if not np.allclose(saved[key][:CHECK_ROWS], z, atol=1e-5):
            max_diff = float(np.max(np.abs(saved[key][:CHECK_ROWS] - z)))
            _fail(f"{key}: logits mismatch for rows [0:{CHECK_ROWS}], "
                  f"max_diff={max_diff:.2e}")
        _pass(f"{key}: logits match for rows [0:{CHECK_ROWS}] (atol=1e-5)")

    # ------------------------------------------------------------------
    # Test 4: Manifest sha1 matches each npy file
    # ------------------------------------------------------------------
    print("\n--- Test 4: Manifest SHA-1 checksums ---")
    for key in ALL_KEYS:
        npy_path = MODELS_DIR / f"{PREFIX}_{key}.npy"
        expected_sha1 = manifest["arrays"][key]["sha1"]
        actual_sha1 = _sha1_of_file(npy_path)
        if actual_sha1 != expected_sha1:
            _fail(f"{key}: sha1 mismatch "
                  f"(expected {expected_sha1[:12]}..., got {actual_sha1[:12]}...)")
        _pass(f"{key}: sha1 = {actual_sha1[:12]}... OK")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
