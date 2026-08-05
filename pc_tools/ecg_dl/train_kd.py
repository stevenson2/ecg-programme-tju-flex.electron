#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge-Distillation (KD) training for ECG-ResNet-Lite-Large.

Reproduces the exact data path of precompute_teacher_logits.py
(deploy-chain, patient-level split, PTB train-patient filter + abnormal cap).

Usage:
    python3 train_kd.py --alpha 0.5 --temperature 3 --model-name kd_a0.5_t3.h5 \
        --log-suffix kd_a0.5_t3 --epochs 200 --patience 15
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
    make_domain_balanced_dataset_kd,
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
)
from losses.kd_loss import make_kd_loss, SlicedAUC  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Knowledge-Distillation training for ECG-ResNet-Lite-Large.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--alpha", type=float, required=True,
                    help="KD loss weight (1-alpha)*CE + alpha*KL.")
    ap.add_argument("--temperature", type=float, required=True,
                    help="Softmax temperature for KD KL term.")
    ap.add_argument("--teacher-prefix", type=str, default="teacher_logits_ssl",
                    help="Prefix for precomputed teacher .npy files in models/.")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--optimizer", choices=["adamw", "sgd"], default="sgd")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--ptb-abn-max", type=int, default=10000)
    ap.add_argument("--model-name", type=str, required=True,
                    help="Unique checkpoint filename (e.g. kd_a0.5_t3.h5).")
    ap.add_argument("--log-suffix", type=str, required=True,
                    help="Suffix for train_history CSV (e.g. kd_a0.5_t3).")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    tf.random.set_seed(TRAIN_CONFIG["random_seed"])
    np.random.seed(TRAIN_CONFIG["random_seed"])

    # ------------------------------------------------------------------
    # 1. Replicate the exact data path of precompute_teacher_logits.py
    # ------------------------------------------------------------------
    print("=" * 60)
    print("[1/7] Setting npz suffix to '_deploy' ...")
    set_npz_suffix("_deploy")

    # -- A-domain: MIT + INCART, patient-level split --
    print("\n" + "=" * 60)
    print("[2/7] Loading A-domain (MIT+INCART) ...")
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
    print(f"  A-train: {len(a_train_beats):,} beats")
    print(f"  A-val:   {len(a_val_beats):,} beats")

    # -- B-domain: PTB, train-patient filter + abnormal cap --
    print("\n" + "=" * 60)
    print("[3/7] Loading B-domain (PTB) ...")
    ptb = load_ptb_data()
    _trp, _, _, _ps = patient_level_split(
        ptb["record_ids"], build_ptb_patient_map()
    )
    ptb_beats = ptb["beats"][_trp]
    ptb_labels = ptb["labels"][_trp]
    print(f"  PTB train-patient filter: {_ps['n_train']}/{_ps['n_patients']} "
          f"patients, {len(ptb_beats):,} beats remain")

    mask_n = ptb_labels == 0
    ptb_normals = ptb_beats[mask_n]
    ptb_normal_labels = ptb_labels[mask_n]

    idx_a = np.where(ptb_labels == 1)[0]
    if len(idx_a) > args.ptb_abn_max:
        rng_abn = np.random.default_rng(42)
        idx_a = rng_abn.choice(idx_a, args.ptb_abn_max, replace=False)
    ptb_abnormals = ptb_beats[idx_a]
    ptb_abnormal_labels = ptb_labels[idx_a]

    # Normals first, then abnormals (same order as precompute script)
    b_train_beats = np.concatenate([ptb_normals, ptb_abnormals], axis=0)
    b_train_labels = np.concatenate([ptb_normal_labels, ptb_abnormal_labels],
                                    axis=0)
    print(f"  B-train: {len(b_train_beats):,} beats "
          f"(normal {len(ptb_normals)} + abnormal {len(ptb_abnormals)})")

    # ------------------------------------------------------------------
    # 2. Load precomputed teacher logits + labels, assert alignment
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    prefix = args.teacher_prefix
    print(f"[4/7] Loading teacher logits: models/{prefix}_*.npy ...")

    z_a_train = np.load(MODELS_DIR / f"{prefix}_a_train.npy")
    z_a_val = np.load(MODELS_DIR / f"{prefix}_a_val.npy")
    z_b_train = np.load(MODELS_DIR / f"{prefix}_b_train.npy")

    loaded_a_train_labels = np.load(MODELS_DIR / f"{prefix}_a_train_labels.npy")
    loaded_a_val_labels = np.load(MODELS_DIR / f"{prefix}_a_val_labels.npy")
    loaded_b_train_labels = np.load(MODELS_DIR / f"{prefix}_b_train_labels.npy")

    # Row-count assertions
    for name, loaded, local in [
        ("a_train logits", z_a_train, a_train_beats),
        ("a_val logits", z_a_val, a_val_beats),
        ("b_train logits", z_b_train, b_train_beats),
        ("a_train labels", loaded_a_train_labels, a_train_labels),
        ("a_val labels", loaded_a_val_labels, a_val_labels),
        ("b_train labels", loaded_b_train_labels, b_train_labels),
    ]:
        if len(loaded) != len(local):
            raise AssertionError(
                f"ROW COUNT MISMATCH for {name}: "
                f"loaded {len(loaded)} vs recomputed {len(local)}. "
                f"Check that precompute_teacher_logits.py and train_kd.py "
                f"use the same split recipe and npz suffix.")

    # Label equality assertions
    for name, loaded, local in [
        ("a_train labels", loaded_a_train_labels, a_train_labels),
        ("a_val labels", loaded_a_val_labels, a_val_labels),
        ("b_train labels", loaded_b_train_labels, b_train_labels),
    ]:
        if not np.array_equal(loaded, local):
            raise AssertionError(
                f"LABEL MISMATCH for {name}: loaded != recomputed. "
                f"Precomputed and local split recipes diverged.")

    print(f"  z_a_train: {z_a_train.shape}, z_a_val: {z_a_val.shape}, "
          f"z_b_train: {z_b_train.shape}")
    print("  ✓ All row counts and label arrays match.")

    # ------------------------------------------------------------------
    # 3. Build KD targets: concat([onehot(2), teacher_logits(2)])
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[5/7] Building KD targets ...")

    a_train_onehot = tf.keras.utils.to_categorical(a_train_labels, 2).astype(np.float32)
    b_train_onehot = tf.keras.utils.to_categorical(b_train_labels, 2).astype(np.float32)
    a_val_onehot = tf.keras.utils.to_categorical(a_val_labels, 2).astype(np.float32)
    a_val_targets = np.concatenate([a_val_onehot, z_a_val], axis=-1).astype(np.float32)

    print(f"  a_train_onehot: {a_train_onehot.shape}, z_a_train: {z_a_train.shape}")
    print(f"  b_train_onehot: {b_train_onehot.shape}, z_b_train: {z_b_train.shape}")
    print(f"  a_val_targets:  {a_val_targets.shape}")

    # ------------------------------------------------------------------
    # 4. Build datasets
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[6/7] Building domain-balanced KD dataset ...")

    x_a_train = a_train_beats.astype(np.float32)  # (N,250) — dataset 内部 add_channel_dim
    x_b_train = b_train_beats.astype(np.float32)
    x_a_val = add_channel_dim(a_val_beats.astype(np.float32))  # val 直接喂模型, 需 (N,250,1)

    train_ds = make_domain_balanced_dataset_kd(
        x_a_train, a_train_onehot, z_a_train,
        x_b_train, b_train_onehot, z_b_train,
        batch_size=args.batch_size,
        frac_b=0.20,
        weight_b=0.5,
    )
    val_data = (x_a_val, a_val_targets)
    print(f"  Train DS: {train_ds}")
    print(f"  Val:      x={x_a_val.shape}, y={a_val_targets.shape}")

    # ------------------------------------------------------------------
    # 5. Build model, compile, train
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[7/7] Building model and training ...")

    model = build_ecg_resnet_lite_large(input_shape=(250, 1))
    model_summary_table(model)

    # Optimizer: replicate train.py SGD branch VERBATIM
    if args.optimizer == "sgd":
        opt = tf.keras.optimizers.SGD(
            learning_rate=args.lr, momentum=0.9,
            nesterov=True, weight_decay=1e-4,
        )
        print(f"[编译] SGD+Nesterov (lr={args.lr}, momentum=0.9, wd=1e-4)")
    else:
        opt = tf.keras.optimizers.AdamW(
            learning_rate=args.lr, weight_decay=1e-4,
        )
        print(f"[编译] AdamW (lr={args.lr}, wd=1e-4)")

    model.compile(
        optimizer=opt,
        loss=make_kd_loss(args.alpha, args.temperature),
        metrics=[SlicedAUC(name="auc")],
    )

    # Callbacks
    callbacks = get_resnet_callbacks(
        model_name=args.model_name,
        early_patience=args.patience,
    )
    history_csv = str(MODELS_DIR / f"train_history_{args.log_suffix}.csv")
    callbacks.append(
        tf.keras.callbacks.CSVLogger(history_csv, append=False)
    )
    print(f"[训练] Loss 历史: {history_csv}")

    history = model.fit(
        train_ds,
        validation_data=val_data,
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
    print("KD Training Summary:")
    print(f"  alpha:       {args.alpha}")
    print(f"  temperature: {args.temperature}")
    print(f"  epochs run:  {epochs_run}")
    print(f"  best val_auc: {best_val_auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
