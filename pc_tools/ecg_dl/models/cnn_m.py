#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG-CNN-M:  3-beat context model (M/L ~450K, S ~140K params).

Architecture per ROADMAP Phase 2B:
  Input:  3-beat concatenation (750, 1)
  Stem:   Conv1D(f, k15, s=2) → BN → ReLU          → (375, f)
  Block1: Conv1D(2f, k7) → BN → ReLU → MaxPool(2)  → (187, 2f)
  Block2: Conv1D(4f, k5) → BN → ReLU → MaxPool(2)  → (93,  4f)
  Block3: Conv1D(6f, k5) → BN → ReLU → MaxPool(2)  → (46,  6f)
  Block4: Conv1D(6f, k3) → BN → ReLU → GAP         → (6f)

  CNN-M (f=32): ~453K params
  CNN-M-S (f=16): ~114K params
  CNN-M-T (f=24): ~140K params ← Phase 2B-1 target

Dual-head:
  - Projection Head (SSL):  Dense(128)→Dense(64)
  - Classifier Head (sup):  Dense(fc)→Dropout→Dense(2)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
from tensorflow.keras import layers, Model

from config import TRAIN_CONFIG, CLASS_NAMES
from losses.focal_loss import FocalLoss

THREE_BEAT_WINDOW = 750


def build_ecg_cnn_m_encoder(
    input_shape=None,
    base_filters=24,
    fc_units=96,
    dropout_rate=0.45,
    name_prefix="cnn_m"
):
    """
    ECG-CNN-M encoder with configurable base_filters.

    Channel plan (base_filters=f):
      Stem   f → Block1 2f → Block2 4f → Block3 6f → Block4 6f

    Args:
        base_filters:  Stem output channels (16=S/114K, 24=T/140K, 32=M/453K)
        fc_units:      Classifier head Dense units
        dropout_rate:  Dropout rate for classifier
        name_prefix:   Layer name prefix
    """
    if input_shape is None:
        input_shape = (THREE_BEAT_WINDOW, 1)

    f = base_filters
    inputs = layers.Input(shape=input_shape, name=f"{name_prefix}_input")

    # Stem: Conv1D(f, k15, stride=2)
    x = layers.Conv1D(f, 15, strides=2, padding="same",
                      name=f"{name_prefix}_stem_conv")(inputs)
    x = layers.BatchNormalization(name=f"{name_prefix}_stem_bn")(x)
    x = layers.ReLU(name=f"{name_prefix}_stem_relu")(x)

    # Block 1: Conv1D(2f, k7) → BN → ReLU → MaxPool(2)
    x = layers.Conv1D(2 * f, 7, padding="same",
                      name=f"{name_prefix}_b1_conv")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_b1_bn")(x)
    x = layers.ReLU(name=f"{name_prefix}_b1_relu")(x)
    x = layers.MaxPooling1D(2, name=f"{name_prefix}_b1_pool")(x)

    # Block 2: Conv1D(4f, k5) → BN → ReLU → MaxPool(2)
    x = layers.Conv1D(4 * f, 5, padding="same",
                      name=f"{name_prefix}_b2_conv")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_b2_bn")(x)
    x = layers.ReLU(name=f"{name_prefix}_b2_relu")(x)
    x = layers.MaxPooling1D(2, name=f"{name_prefix}_b2_pool")(x)

    # Block 3: Conv1D(6f, k5) → BN → ReLU → MaxPool(2)
    x = layers.Conv1D(6 * f, 5, padding="same",
                      name=f"{name_prefix}_b3_conv")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_b3_bn")(x)
    x = layers.ReLU(name=f"{name_prefix}_b3_relu")(x)
    x = layers.MaxPooling1D(2, name=f"{name_prefix}_b3_pool")(x)

    # Block 4: Conv1D(6f, k3) → BN → ReLU → GAP
    x = layers.Conv1D(6 * f, 3, padding="same",
                      name=f"{name_prefix}_b4_conv")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_b4_bn")(x)
    x = layers.ReLU(name=f"{name_prefix}_b4_relu")(x)
    features = layers.GlobalAveragePooling1D(name=f"{name_prefix}_gap")(x)

    # Classifier head
    h = layers.Dense(fc_units, activation="relu",
                     name=f"{name_prefix}_fc1")(features)
    h = layers.Dropout(dropout_rate, name=f"{name_prefix}_do")(h)

    return inputs, h, features


def _build_classifier(h, fc_units, n_classes, dropout_rate, name):
    """Full classifier from already-built head (returns final output)."""
    outputs = layers.Dense(n_classes, activation="softmax",
                           name=f"{name}_out")(h)
    return outputs


def build_projection_head(features, proj_dim=64, name="projection"):
    """Projection head for SimCLR: Dense(128)→Dense(64)."""
    h = layers.Dense(128, activation="relu", name=f"{name}_fc1")(features)
    outputs = layers.Dense(proj_dim, name=f"{name}_out")(h)
    return outputs


# ---- Public factory functions ----

def build_ecg_cnn_m(input_shape=None, n_classes=2, dropout_rate=0.4):
    """CNN-M (base_filters=24, ~140K params). Phase 2B-1 target."""
    inputs, h, features = build_ecg_cnn_m_encoder(
        input_shape=input_shape, base_filters=24,
        fc_units=80, dropout_rate=dropout_rate, name_prefix="ecg_cnn_m"
    )
    outputs = _build_classifier(h, 80, n_classes, dropout_rate, "ecg_cnn_m")
    return Model(inputs=inputs, outputs=outputs, name="ecg_cnn_m")


def build_ecg_cnn_m_classifier(input_shape=None, n_classes=2, dropout_rate=0.4):
    """CNN-M (base_filters=24, ~140K) — alias for build_ecg_cnn_m."""
    return build_ecg_cnn_m(input_shape=input_shape, n_classes=n_classes,
                           dropout_rate=dropout_rate)


def build_ecg_cnn_m_small(input_shape=None, n_classes=2, dropout_rate=0.45):
    """CNN-M-Small (base_filters=16, ~114K). Lightweight variant."""
    inputs, h, features = build_ecg_cnn_m_encoder(
        input_shape=input_shape, base_filters=16,
        fc_units=64, dropout_rate=dropout_rate, name_prefix="ecg_cnn_m_s"
    )
    outputs = _build_classifier(h, 64, n_classes, dropout_rate, "ecg_cnn_m_s")
    return Model(inputs=inputs, outputs=outputs, name="ecg_cnn_m_small")


def build_ecg_cnn_m_large(input_shape=None, n_classes=2, dropout_rate=0.4):
    """CNN-M-Large (base_filters=32, ~453K). For SSL experiments."""
    inputs, h, features = build_ecg_cnn_m_encoder(
        input_shape=input_shape, base_filters=32,
        fc_units=128, dropout_rate=dropout_rate, name_prefix="ecg_cnn_m_l"
    )
    outputs = _build_classifier(h, 128, n_classes, dropout_rate, "ecg_cnn_m_l")
    return Model(inputs=inputs, outputs=outputs, name="ecg_cnn_m_large")


def compile_model(
    model: Model,
    learning_rate: float = None
) -> Model:
    """Compile ECG-CNN-M with FocalLoss + Adam + metrics."""
    if learning_rate is None:
        learning_rate = TRAIN_CONFIG.get("learning_rate", 0.0005)

    fl_cfg = TRAIN_CONFIG.get("focal_loss", {})
    if fl_cfg.get("enabled", True):
        loss = FocalLoss(
            gamma=fl_cfg.get("gamma", 1.0),
            alpha=fl_cfg.get("alpha", 0.75),
            label_smoothing=fl_cfg.get("label_smoothing", 0.0),
            from_logits=False,
        )
        print(f"[ECG-CNN-M] FocalLoss (gamma={loss.gamma}, alpha={loss.alpha})")
    else:
        loss = "categorical_crossentropy"
        print("[ECG-CNN-M] CategoricalCrossentropy")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )
    return model


def get_callbacks(model_name="best_cnn_m.h5"):
    """Training callbacks for ECG-CNN-M."""
    from config import MODELS_DIR
    from config import TRAIN_CONFIG as CFG

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CFG.get("early_stopping_patience", 20),
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=CFG.get("reduce_lr_factor", 0.5),
            patience=CFG.get("reduce_lr_patience", 8),
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / model_name),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
    ]


def model_summary_table(model: Model):
    """Print parameter summary."""
    total = model.count_params()
    print(f"\n[ECG-CNN-M] 总参数: {total:,}")
    print(f"  FP32: {total * 4 / 1024:.1f} KB")
    print(f"  INT8: {total / 1024:.1f} KB")
    trainable = sum(w.shape.num_elements() for w in model.trainable_weights)
    print(f"  Trainable: {trainable:,}")


if __name__ == "__main__":
    model = build_ecg_cnn_m_classifier()
    model.summary()
    model_summary_table(model)
