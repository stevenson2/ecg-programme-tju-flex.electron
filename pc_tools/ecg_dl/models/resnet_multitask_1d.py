#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG-ResNet-MultiTask — Route F: Multi-task Learning

Shared ResNet encoder with task-specific heads:
  1. Classification:  Normal/Abnormal (2-class softmax)
  2. BPM regression:  Heart rate estimate (linear)
  3. SQI regression:  Signal quality 0–1 (sigmoid)

Design follows PHASE3_PLAN.md Route F:
  - Shared ResNet encoder (depthwise sep conv + SE)
  - Multi-head output → weighted loss sum
  - Target: equivalent data volume ×3, Recall 82%→86-88%
"""

import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import INFERENCE_CONFIG


def se_block(x, reduction=4, name="se"):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D(name=f"{name}_gap")(x)
    s = layers.Dense(max(1, ch // reduction), activation='relu',
                     name=f"{name}_d1")(s)
    s = layers.Dense(ch, activation='sigmoid', name=f"{name}_d2")(s)
    s = layers.Reshape((1, ch), name=f"{name}_rs")(s)
    return layers.Multiply(name=f"{name}_mul")([x, s])


def res_block(x, filters, kernel_size, stride=1, block_id=0, pre_act=True):
    """
    Residual block with SE attention.

    pre_act=True:  BN→ReLU→DepthwiseConv→BN→ReLU→PointwiseConv→BN→SE (He 2016)
    pre_act=False: DepthwiseConv→BN→ReLU→PointwiseConv→BN→SE (original)
    """
    pfx = f"b{block_id}"
    shortcut = x
    in_ch = x.shape[-1]

    if pre_act:
        # Pre-activation: BN→ReLU then conv
        x = layers.BatchNormalization(name=f"{pfx}_bn0")(x)
        x = layers.ReLU(name=f"{pfx}_rl0")(x)
        x = layers.DepthwiseConv1D(kernel_size, strides=stride, padding='same',
                                   use_bias=False, name=f"{pfx}_dw")(x)
        x = layers.BatchNormalization(name=f"{pfx}_bn1")(x)
        x = layers.ReLU(name=f"{pfx}_rl1")(x)
        x = layers.Conv1D(filters, 1, padding='same', use_bias=False,
                          name=f"{pfx}_pw")(x)
        x = layers.BatchNormalization(name=f"{pfx}_bn2")(x)
    else:
        x = layers.DepthwiseConv1D(kernel_size, strides=stride, padding='same',
                                   use_bias=False, name=f"{pfx}_dw")(x)
        x = layers.BatchNormalization(name=f"{pfx}_dwbn")(x)
        x = layers.ReLU(name=f"{pfx}_dwrl")(x)
        x = layers.Conv1D(filters, 1, padding='same', use_bias=False,
                          name=f"{pfx}_pw")(x)
        x = layers.BatchNormalization(name=f"{pfx}_pwbn")(x)

    x = se_block(x, reduction=4, name=f"{pfx}_se")

    if stride != 1 or in_ch != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same',
                                 use_bias=False, name=f"{pfx}_sk")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{pfx}_skbn")(shortcut)

    x = layers.Add(name=f"{pfx}_add")([x, shortcut])
    x = layers.ReLU(name=f"{pfx}_out")(x)
    return x


def build_shared_encoder(input_shape=None,
                         filters=(16, 32, 64, 128),
                         blocks_per_stage=(2, 2, 2, 1),
                         kernel_sizes=(5, 5, 5, 5),
                         strides=(1, 2, 2, 1),
                         pre_act=True,
                         concat_pool=True):
    """
    Build shared ResNet encoder (Route H optimizations).

    pre_act:    Use pre-activation ResBlock (BN→ReLU→Conv)
    concat_pool: GAP + GlobalMaxPool concatenation

    Returns:
        inputs: Keras Input tensor.
        features: Output tensor after pooling (GAP only or concat pooling).
    """
    if input_shape is None:
        input_shape = (INFERENCE_CONFIG['window_size'], 1)

    inputs = layers.Input(shape=input_shape, name="ecg_input")

    x = layers.Conv1D(16, 7, strides=2, padding='same', use_bias=False,
                      name="stem")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_rl")(x)

    bid = 0
    for si in range(len(filters)):
        f, k = filters[si], kernel_sizes[si]
        for blk in range(blocks_per_stage[si]):
            s = strides[si] if blk == 0 else 1
            x = res_block(x, filters=f, kernel_size=k, stride=s,
                          block_id=bid, pre_act=pre_act)
            bid += 1

    if concat_pool:
        gap = layers.GlobalAveragePooling1D(name="shared_gap")(x)
        gmp = layers.GlobalMaxPooling1D(name="shared_gmp")(x)
        x = layers.Concatenate(name="shared_concat")([gap, gmp])
        features = layers.Dense(64, activation='relu', name="shared_proj")(x)
    else:
        features = layers.GlobalAveragePooling1D(name="shared_gap")(x)

    return inputs, features


def build_classifier_head(features, n_classes=2, dropout_rate=0.3):
    x = layers.Dense(32, activation='relu', name="cls_fc1")(features)
    x = layers.BatchNormalization(name="cls_bn")(x)
    x = layers.Dropout(dropout_rate, name="cls_do")(x)
    x = layers.Dense(16, activation='relu', name="cls_fc2")(x)
    x = layers.Dropout(0.3, name="cls_do2")(x)
    return layers.Dense(n_classes, activation='softmax', name="cls_out")(x)


def build_bpm_head(features):
    x = layers.Dense(16, activation='relu', name="bpm_fc1")(features)
    return layers.Dense(1, activation='sigmoid', name="bpm_out")(x)


def build_sqi_head(features):
    x = layers.Dense(16, activation='relu', name="sqi_fc1")(features)
    return layers.Dense(1, activation='sigmoid', name="sqi_out")(x)


def build_ecg_resnet_multitask(
    input_shape=None,
    n_classes=2,
    filters=(16, 32, 64, 128),
    blocks_per_stage=(2, 2, 2, 1),
    kernel_sizes=(5, 5, 3, 3),
    strides=(1, 2, 2, 1),
    dropout_rate=0.3
):
    """
    Build multi-task ResNet model for Route F.

    Outputs: [cls_out, bpm_out, sqi_out] as 3 separate tensors.
    - cls_out: (batch, 2) softmax [Normal, Abnormal]
    - bpm_out: (batch, 1) linear BPM estimate
    - sqi_out: (batch, 1) sigmoid SQI [0, 1]

    Returns:
        tf.keras.Model with 3 named outputs.
    """
    inputs, features = build_shared_encoder(
        input_shape=input_shape, filters=filters,
        blocks_per_stage=blocks_per_stage, kernel_sizes=kernel_sizes,
        strides=strides)

    cls_out = build_classifier_head(features, n_classes=n_classes,
                                    dropout_rate=dropout_rate)
    bpm_out = build_bpm_head(features)
    sqi_out = build_sqi_head(features)

    return Model(inputs=inputs, outputs=[cls_out, bpm_out, sqi_out],
                 name="ecg_resnet_multitask")


def compile_multitask_model(model, learning_rate=0.0005,
                             gamma=1.0, alpha=0.75,
                             w_cls=1.0, w_bpm=0.3, w_sqi=0.2):
    """Compile multi-task model with per-head losses and loss_weights."""
    from losses.focal_loss import FocalLoss
    opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate,
                                     weight_decay=1e-4)
    model.compile(
        optimizer=opt,
        loss={
            "cls_out": FocalLoss(gamma=gamma, alpha=alpha),
            "bpm_out": "mse",
            "sqi_out": "mse",
        },
        loss_weights={
            "cls_out": w_cls,
            "bpm_out": w_bpm,
            "sqi_out": w_sqi,
        },
        metrics={
            "cls_out": ["accuracy",
                         tf.keras.metrics.Precision(name='precision'),
                         tf.keras.metrics.Recall(name='recall'),
                         tf.keras.metrics.AUC(name='auc')],
            "bpm_out": ["mae"],
            "sqi_out": ["mae"],
        })
    return model


def model_summary_table(model):
    """Print per-layer parameter table with FP32/INT8 estimates."""
    total = model.count_params()
    trainable = sum(tf.keras.backend.count_params(w)
                    for w in model.trainable_weights)
    print(f"\n{'='*55}")
    print(f"  Model: {model.name} (Multi-Task)")
    print(f"{'='*55}")
    print(f"  {'Layer':<30} {'Output':<18} {'Params':<10}")
    print(f"  {'-'*55}")
    for layer in model.layers:
        sh = str(layer.output_shape) if hasattr(layer, 'output_shape') else '-'
        p = layer.count_params() if hasattr(layer, 'count_params') else 0
        print(f"  {layer.name:<30} {sh:<18} {p:<10,}")
    print(f"  {'-'*55}")
    print(f"  Total params:   {total:>8,}")
    print(f"  Trainable:      {trainable:>8,}")
    print(f"  FP32 estimate:  {total * 4 / 1024:>8.1f} KB")
    print(f"  INT8 estimate:  {total * 1 / 1024:>8.1f} KB")
    print(f"{'='*55}\n")


def get_multitask_callbacks(model_name="best_resnet_multitask.h5",
                             csv_name="multitask_history.csv",
                             early_patience=20, lr_patience=8):
    """Training callbacks for multi-task model."""
    from config import MODELS_DIR
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=early_patience,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=lr_patience,
            min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / model_name),
            monitor='val_loss', mode='min', save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(
            str(MODELS_DIR / csv_name)),
    ]


if __name__ == "__main__":
    import numpy as np
    print("[ECG-ResNet-MultiTask] Model build test...\n")
    m = build_ecg_resnet_multitask()
    model_summary_table(m)
    x = np.random.randn(4, 250, 1).astype(np.float32)
    cls, bpm, sqi = m.predict(x, verbose=0)
    print(f"[Test] {x.shape} -> cls={cls.shape}, bpm={bpm.shape}, sqi={sqi.shape}")
    print(f"  cls: {cls[0]}, bpm: {bpm[0,0]:.1f}, sqi: {sqi[0,0]:.3f}")
    print("[ECG-ResNet-MultiTask] OK")
