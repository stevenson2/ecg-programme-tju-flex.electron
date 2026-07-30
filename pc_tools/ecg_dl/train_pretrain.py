#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Route K Stage 1: PTB-XL Supervised Pretraining (5-superclass multi-label).

Input:  10s records at 100Hz → 1000 samples, Lead II only
Output: 5 sigmoid heads (NORM/MI/CD/STTC/HYP)

Architecture: ResNet-L encoder + concat pooling + 5-head sigmoid
Transfer:   Encoder weights → MIT-BIH+INCART beat-level finetune

Usage:
  # First, preprocess PTB-XL records:
  python data/preprocess_ptbxl_records.py

  # Then pretrain:
  python train_pretrain.py --epochs 100

  # Then finetune on MIT-BIH+INCART:
  python train_multitask.py --incart --pretrained models/best_ptbxl_pretrain.h5 --epochs 100
"""

import sys, os
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PROCESSED_DIR, MODELS_DIR

SUPERCLASS_NAMES = ['NORM', 'MI', 'CD', 'STTC', 'HYP']


def load_ptbxl_records():
    npz = PROCESSED_DIR / "ptbxl_records_100hz.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"PTB-XL records not found: {npz}\n"
            f"Run: python data/preprocess_ptbxl_records.py")

    data = np.load(npz)
    signals = data['signals']     # (N, 2, 1000)
    labels = data['labels']        # (N, 5)
    folds = data['folds']          # (N,) 1-10
    print(f"[PTB-XL Records] Loaded: {signals.shape}, labels={labels.shape}")
    for i, name in enumerate(SUPERCLASS_NAMES):
        c = int(labels[:, i].sum())
        print(f"  {name}: {c} ({c/len(labels)*100:.1f}%)")
    return signals, labels, folds


def build_pretrain_model(input_shape=(1000, 1), n_classes=5, dropout_rate=0.3):
    """ResNet encoder (Route H) + multi-label sigmoid heads."""
    from models.resnet_multitask_1d import (
        build_shared_encoder,
    )
    layers = tf.keras.layers

    inputs, features = build_shared_encoder(
        input_shape=input_shape,
        filters=(16, 32, 64, 128),
        blocks_per_stage=(2, 3, 3, 1),
        kernel_sizes=(5, 5, 5, 5),
        strides=(1, 2, 2, 1),
        pre_act=True,
        concat_pool=True,
    )

    x = layers.Dense(128, activation='relu', name="pretrain_fc1")(features)
    x = layers.BatchNormalization(name="pretrain_bn")(x)
    x = layers.Dropout(0.25, name="pretrain_do1")(x)
    x = layers.Dense(64, activation='relu', name="pretrain_fc2")(x)
    x = layers.Dropout(0.5, name="pretrain_do2")(x)
    outputs = layers.Dense(n_classes, activation='sigmoid', name="pretrain_out")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ptbxl_pretrain")


def compile_pretrain_model(model, learning_rate=0.001):
    opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate,
                                     weight_decay=1e-4)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='auc', multi_label=True),
            tf.keras.metrics.BinaryAccuracy(name='acc'),
        ])
    return model


def prepare_pretrain_data(signals, labels, folds):
    """Split by strat_fold: 1-8 train, 9 val, 10 test."""
    train_mask = np.isin(folds, [1, 2, 3, 4, 5, 6, 7, 8])
    val_mask = folds == 9
    test_mask = folds == 10

    x_train = signals[train_mask]
    y_train = labels[train_mask]
    x_val = signals[val_mask]
    y_val = labels[val_mask]
    x_test = signals[test_mask]
    y_test = labels[test_mask]

    # Use Lead II only: signals is (N, 2, 1000) → (N, 1000, 1)
    x_train = x_train[:, 1:2, :].transpose(0, 2, 1)
    x_val = x_val[:, 1:2, :].transpose(0, 2, 1)
    x_test = x_test[:, 1:2, :].transpose(0, 2, 1)

    print(f"\n[Split] Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
    print(f"  Train class dist: {y_train.sum(axis=0)}")
    print(f"  Val class dist:   {y_val.sum(axis=0)}")
    print(f"  Test class dist:  {y_test.sum(axis=0)}")

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(64).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(64).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, (x_test, y_test)


def get_callbacks(model_name="best_ptbxl_pretrain.h5"):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=15, mode='max',
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5,
            min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / model_name),
            monitor='val_auc', mode='max', save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(
            str(MODELS_DIR / "pretrain_history.csv")),
    ]


def evaluate_pretrain(model, x_test, y_test):
    from sklearn.metrics import roc_auc_score

    y_pred = model.predict(x_test, verbose=0)
    print(f"\n{'='*50}")
    print("  PTB-XL Pretraining Evaluation (macro AUC)")
    print(f"{'='*50}")
    for i, name in enumerate(SUPERCLASS_NAMES):
        if y_test[:, i].sum() > 0 and y_test[:, i].var() > 0:
            auc = roc_auc_score(y_test[:, i], y_pred[:, i])
            print(f"  {name:<6}: AUC={auc:.4f}")
    macro_auc = roc_auc_score(y_test, y_pred, average='macro')
    print(f"  {'MACRO':<6}: AUC={macro_auc:.4f}")
    print(f"{'='*50}")
    return macro_auc


def train(epochs=100, batch_size=64, learning_rate=0.001, max_records=None):
    print(f"\n{'='*60}")
    print("  Route K Stage 1: PTB-XL Supervised Pretraining")
    print(f"  Target: 5 superclasses {SUPERCLASS_NAMES}")
    print(f"{'='*60}\n")

    # Data
    print("[1/3] Loading PTB-XL records...")
    try:
        signals, labels, folds = load_ptbxl_records()
    except FileNotFoundError:
        print("  Preprocessing PTB-XL records first...")
        from data.preprocess_ptbxl_records import preprocess
        signals, labels, folds, _ = preprocess(max_records=max_records)

    train_ds, val_ds, test_ds, (x_test, y_test) = prepare_pretrain_data(
        signals, labels, folds)

    # Model
    print("\n[2/3] Building pretraining model...")
    model = build_pretrain_model(dropout_rate=0.3)
    model = compile_pretrain_model(model, learning_rate=learning_rate)
    model.summary()

    # Train
    print("\n[3/3] Training...")
    callbacks = get_callbacks()

    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, callbacks=callbacks, verbose=2)

    model.save(str(MODELS_DIR / "final_ptbxl_pretrain.h5"))
    print(f"\n  Saved: {MODELS_DIR / 'final_ptbxl_pretrain.h5'}")

    evaluate_pretrain(model, x_test, y_test)

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PTB-XL Pretraining")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max', type=int, default=None,
                        help='Max records for quick test')
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size,
          learning_rate=args.lr, max_records=args.max)
