#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute teacher soft-target logits for knowledge distillation.

Mirrors the exact data path of train.py --deploy-chain --incart --ptb-beat
(patient-level split, PTB train-patient filter, abnormal cap).

Usage:
    python precompute_teacher_logits.py \
        --teacher models/final_ssl_finetuned.h5 \
        --out-prefix teacher_logits_ssl \
        --eps 1e-7 --batch 1024
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: make sibling packages importable (same as train.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import (
    set_npz_suffix,
    load_mit_incart_merged,
    load_ptb_data,
    add_channel_dim,
)
from data.patient_split import (
    build_mit_patient_map,
    build_incart_patient_map,
    build_ptb_patient_map,
    patient_level_split,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha1_of_file(path: Path) -> str:
    """Return hex SHA-1 digest of a file (streaming, constant-memory)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_npy(arr: np.ndarray, path: Path) -> None:
    """Save array and print shape."""
    np.save(str(path), arr)
    print(f"  saved {path.name}: shape={arr.shape}, dtype={arr.dtype}")


def _log_stats(name: str, z: np.ndarray) -> tuple:
    """Print and return (mean, std) of logit array z."""
    mean = float(np.mean(z))
    std = float(np.std(z))
    print(f"  {name}: mean(z)={mean:.6f}, std(z)={std:.6f}, "
          f"min(z)={float(np.min(z)):.4f}, max(z)={float(np.max(z)):.4f}")
    return mean, std


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Precompute teacher soft-target logits for KD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--teacher", type=str, required=True,
                    help="Path to teacher .h5 model (softmax output).")
    ap.add_argument("--out-prefix", type=str, default="teacher_logits_ssl",
                    help="Output filename prefix (inside models/ dir).")
    ap.add_argument("--eps", type=float, default=1e-7,
                    help="Clip epsilon for log(p) stability.")
    ap.add_argument("--batch", type=int, default=1024,
                    help="Batch size for teacher.predict().")
    ap.add_argument("--ptb-abn-max", type=int, default=10000,
                    help="Max PTB abnormal beats to keep (cap).")
    args = ap.parse_args()

    teacher_path = Path(args.teacher)
    if not teacher_path.exists():
        ap.error(f"Teacher model not found: {teacher_path}")

    out_dir = Path(__file__).resolve().parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix
    eps = args.eps

    # ------------------------------------------------------------------
    # 1. Switch to deploy-chain data sources
    # ------------------------------------------------------------------
    print("=" * 60)
    print("[1/5] Setting npz suffix to '_deploy' ...")
    set_npz_suffix("_deploy")

    # ------------------------------------------------------------------
    # 2. A-domain: MIT + INCART, patient-level split
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[2/5] Loading A-domain (MIT+INCART) ...")
    data = load_mit_incart_merged()
    beats = data["beats"]
    labels = data["labels"]
    record_ids = data["record_ids"]

    # Build combined patient map exactly like dataset.py prepare_datasets
    _pmap: dict = {}
    _pmap.update(build_mit_patient_map())
    _pmap.update({
        rid + 100000: "inc_" + pat
        for rid, pat in build_incart_patient_map().items()
    })

    tr_m, va_m, te_m, pstats = patient_level_split(record_ids, _pmap)
    print(f"  Patient split: {pstats['n_patients']} patients = "
          f"train {pstats['n_train']} / val {pstats['n_val']} / test {pstats['n_test']}")
    print(f"  Beat counts:   train {pstats['beats_train']:,} / "
          f"val {pstats['beats_val']:,} / test {pstats['beats_test']:,}")

    a_train_beats = beats[tr_m]
    a_train_labels = labels[tr_m]
    a_val_beats = beats[va_m]
    a_val_labels = labels[va_m]
    print(f"  A-train: {len(a_train_beats):,} beats")
    print(f"  A-val:   {len(a_val_beats):,} beats")

    # ------------------------------------------------------------------
    # 3. B-domain: PTB, train-patient filter + abnormal cap
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[3/5] Loading B-domain (PTB) ...")
    ptb = load_ptb_data()

    # Patient-level filter: keep only TRAIN patients (same as dataset.py)
    _trp, _, _, _ps = patient_level_split(
        ptb["record_ids"], build_ptb_patient_map()
    )
    ptb_beats = ptb["beats"][_trp]
    ptb_labels = ptb["labels"][_trp]
    ptb_rids = ptb["record_ids"][_trp]
    print(f"  PTB train-patient filter: {_ps['n_train']}/{_ps['n_patients']} "
          f"patients, {len(ptb_beats):,} beats remain")

    # Normal beats: all kept (label == 0)
    mask_n = ptb_labels == 0
    ptb_normals = ptb_beats[mask_n]
    ptb_normal_labels = ptb_labels[mask_n]
    n_ptb_n = len(ptb_normals)
    print(f"  PTB normals: {n_ptb_n:,}")

    # Abnormal beats: capped at --ptb-abn-max with rng seed 42
    idx_a = np.where(ptb_labels == 1)[0]
    ptb_abn_max = args.ptb_abn_max
    if len(idx_a) > ptb_abn_max:
        rng = np.random.default_rng(42)
        idx_a = rng.choice(idx_a, ptb_abn_max, replace=False)
    ptb_abnormals = ptb_beats[idx_a]
    ptb_abnormal_labels = ptb_labels[idx_a]
    n_ptb_a = len(ptb_abnormals)
    print(f"  PTB abnormals: {n_ptb_a:,} (capped at {ptb_abn_max})")

    # Concatenate: normals first, then abnormals (same order as dataset.py)
    b_train_beats = np.concatenate([ptb_normals, ptb_abnormals], axis=0)
    b_train_labels = np.concatenate([ptb_normal_labels, ptb_abnormal_labels],
                                    axis=0)
    print(f"  B-train: {len(b_train_beats):,} beats "
          f"(normal {n_ptb_n} + abnormal {n_ptb_a})")

    # ------------------------------------------------------------------
    # 4. Load teacher, predict, compute logits
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"[4/5] Loading teacher model: {teacher_path} ...")
    import tensorflow as tf
    teacher = tf.keras.models.load_model(str(teacher_path), compile=False)
    teacher.summary(print_fn=lambda s: print(f"  {s}"))

    def _predict_logits(beats_arr: np.ndarray, name: str) -> np.ndarray:
        """Run teacher softmax, clip, log -> logits z."""
        x = add_channel_dim(beats_arr.astype(np.float32))
        print(f"\n  Predicting {name} ({len(x):,} beats, batch={args.batch}) ...")
        p_t = teacher.predict(x, batch_size=args.batch, verbose=0)
        assert p_t.ndim == 2 and p_t.shape[1] == 2, (
            f"Expected (N,2) softmax output, got {p_t.shape}")
        z = np.log(np.clip(p_t, eps, 1.0 - eps))
        _log_stats(name, z)
        return z

    z_a_train = _predict_logits(a_train_beats, "A-train logits")
    z_a_val = _predict_logits(a_val_beats, "A-val logits")
    z_b_train = _predict_logits(b_train_beats, "B-train logits")

    # ------------------------------------------------------------------
    # 5. Save arrays, labels, and manifest
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[5/5] Saving outputs ...")

    paths: dict[str, Path] = {}
    arrays: dict[str, np.ndarray] = {
        "a_train": z_a_train,
        "a_val": z_a_val,
        "b_train": z_b_train,
        "a_train_labels": a_train_labels,
        "a_val_labels": a_val_labels,
        "b_train_labels": b_train_labels,
    }
    for key, arr in arrays.items():
        p = out_dir / f"{prefix}_{key}.npy"
        _save_npy(arr, p)
        paths[key] = p

    # Manifest
    manifest: dict = {
        "teacher_model": teacher_path.name,
        "eps": eps,
        "arrays": {},
    }
    for key, arr in arrays.items():
        p = paths[key]
        entry: dict = {
            "filename": p.name,
            "length": len(arr),
            "sha1": _sha1_of_file(p),
        }
        # Add mean/std for logit arrays only
        if not key.endswith("_labels"):
            entry["mean_z"] = float(np.mean(arr))
            entry["std_z"] = float(np.std(arr))
        manifest["arrays"][key] = entry

    manifest_path = out_dir / f"{prefix}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  saved {manifest_path.name}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DONE. Summary:")
    print(f"  A-train logits:  {len(z_a_train):>8,} rows")
    print(f"  A-val   logits:  {len(z_a_val):>8,} rows")
    print(f"  B-train logits:  {len(z_b_train):>8,} rows")
    print(f"  A-train labels:  {len(a_train_labels):>8,}")
    print(f"  A-val   labels:  {len(a_val_labels):>8,}")
    print(f"  B-train labels:  {len(b_train_labels):>8,}")
    print(f"  Manifest: {manifest_path}")
    print(f"  All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
