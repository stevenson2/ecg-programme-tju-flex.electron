#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balanced-Mixed Single-Model training for ECG-ResNet-Lite-Large.

One ResNet-Large model trained on MIT+INCART + PTB (abnormal-only),
class-balanced to abnormal fraction π (default 0.30).
Binary abnormal: 心律失常 ∪ 心梗 = 有问题.

Reproduces the exact data path of train_kd.py (deploy-chain, patient-level
split), but replaces KD loss / domain-balanced batch sampling with offline
class balancing (downsample normals to target π).

Usage:
    python3 train_mixed_balanced.py --abn-frac 0.30 --epochs 200 --patience 40 \
        --lr 0.01 --model-name bal_mixed.h5 --log-suffix bal_mixed
    python3 train_mixed_balanced.py --dry-run
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: make sibling packages importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorflow as tf

# 显存按需分配
_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    try:
        tf.config.experimental.set_memory_growth(_gpus[0], True)
    except Exception:
        pass

from config import MODELS_DIR, TRAIN_CONFIG  # noqa: E402
from data.dataset import (  # noqa: E402
    set_npz_suffix,
    load_mit_incart_merged,
    load_ptb_data,
    add_channel_dim,
    make_tf_dataset,
)
from data.patient_split import (  # noqa: E402
    build_mit_patient_map,
    build_incart_patient_map,
    build_ptb_patient_map,
    patient_level_split,
)
from models.resnet_lite_1d import (  # noqa: E402
    build_ecg_resnet_lite_large,
    get_callbacks as get_resnet_callbacks,
    model_summary_table,
    compile_model,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Balanced-Mixed Single-Model (MIT+INCART + PTB abnormal) "
                    "with class balancing to target abnormal fraction π.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--abn-frac", type=float, default=0.30,
                    help="Target abnormal fraction (π) after balancing.")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--model-name", type=str, default="bal_mixed.h5",
                    help="Unique checkpoint filename (e.g. bal_mixed.h5).")
    ap.add_argument("--log-suffix", type=str, default="bal_mixed",
                    help="Suffix for train_history CSV (e.g. bal_mixed).")
    ap.add_argument("--lr", type=float, default=0.01,
                    help="Learning rate (SGD+Nesterov).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load data, print census, build model, run ONE fit "
                         "step, then exit 0.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for normal downsampling reproducibility.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    tf.random.set_seed(TRAIN_CONFIG["random_seed"])
    np.random.seed(TRAIN_CONFIG["random_seed"])

    # ------------------------------------------------------------------
    # 1. Set npz suffix to deploy chain
    # ------------------------------------------------------------------
    print("=" * 60)
    print("[1/6] Setting npz suffix to '_deploy' ...")
    set_npz_suffix("_deploy")

    # ------------------------------------------------------------------
    # 2. Load A-domain (MIT+INCART), patient-level split
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[2/6] Loading A-domain (MIT+INCART) ...")
    data = load_mit_incart_merged()
    _pmap: dict = {}
    _pmap.update(build_mit_patient_map())
    _pmap.update({
        rid + 100000: "inc_" + pat
        for rid, pat in build_incart_patient_map().items()
    })
    tr_m, va_m, _te_m, pstats = patient_level_split(data["record_ids"], _pmap)
    print(f"  Patient split: {pstats['n_patients']} patients = "
          f"train {pstats['n_train']} / val {pstats['n_val']} / test {pstats['n_test']}")
    a_train_beats = data["beats"][tr_m]
    a_train_labels = data["labels"][tr_m]
    a_val_beats = data["beats"][va_m]
    a_val_labels = data["labels"][va_m]
    print(f"  A-train: {len(a_train_beats):,} beats "
          f"(N={int((a_train_labels == 0).sum()):,}, "
          f"A={int((a_train_labels == 1).sum()):,})")
    print(f"  A-val:   {len(a_val_beats):,} beats "
          f"(N={int((a_val_labels == 0).sum()):,}, "
          f"A={int((a_val_labels == 1).sum()):,})")

    # ------------------------------------------------------------------
    # 3. Load B-domain (PTB), train-patient filter, abnormal ONLY
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[3/6] Loading B-domain (PTB), abnormal beats only ...")
    ptb = load_ptb_data()
    _trp, _, _, _ps = patient_level_split(
        ptb["record_ids"], build_ptb_patient_map()
    )
    ptb_beats = ptb["beats"][_trp]
    ptb_labels = ptb["labels"][_trp]
    print(f"  PTB train-patient filter: {_ps['n_train']}/{_ps['n_patients']} "
          f"patients, {len(ptb_beats):,} beats remain")

    # Keep ABNORMAL ONLY (PTB supplies abnormal supplementation)
    idx_a = np.where(ptb_labels == 1)[0]
    ptb_abn_beats = ptb_beats[idx_a]
    ptb_abn_labels = ptb_labels[idx_a]
    print(f"  PTB abnormal kept: {len(ptb_abn_beats):,} beats "
          f"(ALL train-patient abnormals, no cap)")

    # ------------------------------------------------------------------
    # 4. Balance to target abnormal fraction π
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"[4/6] Balancing to π = {args.abn_frac:.4f} ...")

    # Concatenate all abnormals: A-train abnormal + PTB train abnormal
    a_train_abn_mask = a_train_labels == 1
    a_train_abn = a_train_beats[a_train_abn_mask]
    a_train_abn_labels = a_train_labels[a_train_abn_mask]

    abn_beats = np.concatenate([a_train_abn, ptb_abn_beats], axis=0)
    abn_labels = np.concatenate([a_train_abn_labels, ptb_abn_labels], axis=0)
    n_abn = len(abn_beats)
    print(f"  A-train abnormals:  {len(a_train_abn):,}")
    print(f"  PTB train abnormals: {len(ptb_abn_beats):,}")
    print(f"  Total abnormals:     {n_abn:,}")

    # Normal side: A-train normals only (no PTB normals)
    a_train_norm_mask = a_train_labels == 0
    norm_beats = a_train_beats[a_train_norm_mask]
    n_norm_available = len(norm_beats)
    print(f"  A-train normals available: {n_norm_available:,}")

    # Desired normal count to achieve π
    n_norm_keep = int(round(n_abn * (1.0 - args.abn_frac) / args.abn_frac))
    print(f"  Normals to keep (π={args.abn_frac}): "
          f"{n_norm_keep:,} / {n_norm_available:,} available")

    # Downsample normals if needed
    rng_norm = np.random.default_rng(args.seed)
    if n_norm_available > n_norm_keep:
        keep_idx = rng_norm.choice(n_norm_available, n_norm_keep, replace=False)
        norm_beats_kept = norm_beats[keep_idx]
    else:
        norm_beats_kept = norm_beats
        print(f"  WARNING: not enough normals ({n_norm_available:,} < "
              f"{n_norm_keep:,}), using all available.")

    n_norm_actual = len(norm_beats_kept)
    norm_labels_kept = np.zeros(n_norm_actual, dtype=np.int64)

    # Concatenate and shuffle
    train_x = np.concatenate([norm_beats_kept, abn_beats], axis=0)
    train_y = np.concatenate([norm_labels_kept, abn_labels], axis=0)

    # Global shuffle (mirrors train.py PTB branch: 防止尾部纯域批次)
    rng_shuf = np.random.default_rng(0)
    perm = rng_shuf.permutation(len(train_y))
    train_x = train_x[perm]
    train_y = train_y[perm]

    achieved_pi = train_y.mean()
    print(f"\n  --- Balanced Dataset Census ---")
    print(f"  Normals kept:     {n_norm_actual:,}")
    print(f"  Abnormals total:  {n_abn:,}")
    print(f"  Total train:      {len(train_y):,}")
    print(f"  Achieved π:       {achieved_pi:.4f} "
          f"({'✓' if abs(achieved_pi - args.abn_frac) < 0.01 else '⚠'})")
    print(f"  ==============================")

    # ------------------------------------------------------------------
    # 5. Build datasets
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[5/6] Building tf.data datasets ...")

    train_x_f32 = train_x.astype(np.float32)
    a_val_f32 = a_val_beats.astype(np.float32)

    train_ds = make_tf_dataset(train_x_f32, train_y,
                               batch_size=args.batch_size,
                               shuffle=True, augment=True)
    val_ds = make_tf_dataset(a_val_f32, a_val_labels,
                             batch_size=args.batch_size,
                             shuffle=False)
    print(f"  Train DS: {train_ds}")
    print(f"  Val DS:   {val_ds}")

    # ------------------------------------------------------------------
    # 6. Build model, compile, train
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[6/6] Building model and training ...")

    model = build_ecg_resnet_lite_large(input_shape=(250, 1))
    model_summary_table(model)

    # Compile with SGD+Nesterov (deploy standard)
    compile_model(model, learning_rate=args.lr, loss=None, optimizer="sgd")

    # Callbacks (mirror train_kd.py)
    callbacks = get_resnet_callbacks(
        model_name=args.model_name,
        early_patience=args.patience,
    )
    history_csv = str(MODELS_DIR / f"train_history_{args.log_suffix}.csv")
    callbacks.append(
        tf.keras.callbacks.CSVLogger(history_csv, append=False)
    )
    print(f"[训练] Loss 历史: {history_csv}")

    if args.dry_run:
        print("\n[Dry-run] Running ONE fit step to validate pipeline ...")
        history = model.fit(
            train_ds.take(1),
            validation_data=val_ds.take(1),
            epochs=1,
            callbacks=callbacks,
            verbose=2,
        )
    else:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=2,
        )

    # Save final model
    final_path = MODELS_DIR / f"final_{args.model_name}"
    model.save(str(final_path))
    print(f"[训练] 最终模型已保存: {final_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    best_val_auc = max(history.history.get("val_auc", [0.0]))
    epochs_run = len(history.history.get("loss", []))
    print("\n" + "=" * 60)
    print("Balanced-Mixed Training Summary:")
    print(f"  abn_frac (π):  {args.abn_frac:.4f}")
    print(f"  achieved π:    {achieved_pi:.4f}")
    print(f"  lr:            {args.lr}")
    print(f"  epochs run:    {epochs_run}")
    print(f"  best val_auc:  {best_val_auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
