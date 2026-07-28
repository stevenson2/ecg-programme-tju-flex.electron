#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG-ResNet-Lite — Lightweight 1D Residual CNN with SE Attention
(ModelPlan Phase 1 Architecture)

Design:
  Depthwise Separable Conv + SE Attention + Residual Connections
  Target: ~55K params, INT8 ~55KB, accuracy ≥94%

Architecture:
  Stem:    Conv1D(16,k7,s2) → BN → ReLU                           (125,16)
  Stage A: 2x ResBlock(16,k5)                                      (125,16)
  Stage B: 2x ResBlock(32,k5,s2)                                   (63,32)
  Stage C: 2x ResBlock(64,k3,s2)                                   (32,64)
  Stage D: 1x ResBlock(128,k3)                                     (32,128)
  Head:    GAP → Dense(64) → Dropout(0.3) → Dense(2,Softmax)
"""

import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import INFERENCE_CONFIG


def se_block(x, reduction=4, name="se"):
    """Squeeze-and-Excitation: GAP → Dense(C/r) → ReLU → Dense(C) → Sigmoid."""
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D(name=f"{name}_gap")(x)
    s = layers.Dense(max(1, ch // reduction), activation='relu',
                     name=f"{name}_d1")(s)
    s = layers.Dense(ch, activation='sigmoid', name=f"{name}_d2")(s)
    s = layers.Reshape((1, ch), name=f"{name}_rs")(s)
    return layers.Multiply(name=f"{name}_mul")([x, s])


def res_block(x, filters, kernel_size, stride=1, block_id=0):
    """
    Residual block: DepthwiseConv → BN → ReLU → PointwiseConv → BN → SE → +skip.

    Args:
        x:           Input tensor (B, T, C).
        filters:     Output channels.
        kernel_size: Depthwise kernel size.
        stride:      Stride (2 for spatial reduction).
        block_id:    Unique identifier.

    Returns:
        Output tensor (B, T', filters).
    """
    pfx = f"b{block_id}"
    shortcut = x
    in_ch = x.shape[-1]

    # Depthwise → BN → ReLU
    x = layers.DepthwiseConv1D(kernel_size, strides=stride, padding='same',
                               use_bias=False, name=f"{pfx}_dw")(x)
    x = layers.BatchNormalization(name=f"{pfx}_dwbn")(x)
    x = layers.ReLU(name=f"{pfx}_dwrl")(x)

    # Pointwise (1x1 Conv) → BN
    x = layers.Conv1D(filters, 1, padding='same', use_bias=False,
                      name=f"{pfx}_pw")(x)
    x = layers.BatchNormalization(name=f"{pfx}_pwbn")(x)

    # SE Attention
    x = se_block(x, reduction=4, name=f"{pfx}_se")

    # Shortcut: 1x1 Conv if shape mismatch
    if stride != 1 or in_ch != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same',
                                 use_bias=False, name=f"{pfx}_sk")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{pfx}_skbn")(shortcut)

    x = layers.Add(name=f"{pfx}_add")([x, shortcut])
    x = layers.ReLU(name=f"{pfx}_out")(x)
    return x
# ===========================================================================
# Model Builder
# ===========================================================================

def build_ecg_resnet_lite(
    input_shape=None,
    n_classes=2,
    filters=(16, 32, 64, 128),
    blocks_per_stage=(2, 2, 2, 1),
    kernel_sizes=(5, 5, 3, 3),
    strides=(1, 2, 2, 1),
    dropout_rate=0.3
):
    """
    Build ECG-ResNet-Lite.

    Args:
        input_shape:      Default (250,1). Customizable.
        n_classes:        Output classes (2 for Normal/Abnormal).
        filters:          Output channels per residual stage.
        blocks_per_stage: Number of blocks per stage.
        kernel_sizes:     Depthwise kernel per stage.
        strides:          First-block stride per stage.
        dropout_rate:     Dropout in classifier head.

    Returns:
        tf.keras.Model
    """
    if input_shape is None:
        input_shape = (INFERENCE_CONFIG['window_size'], 1)

    inputs = layers.Input(shape=input_shape, name="ecg_input")

    # Stem
    x = layers.Conv1D(16, 7, strides=2, padding='same', use_bias=False,
                      name="stem")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_rl")(x)

    # Residual Stages
    bid = 0
    for si in range(len(filters)):
        f, k = filters[si], kernel_sizes[si]
        for blk in range(blocks_per_stage[si]):
            s = strides[si] if blk == 0 else 1
            x = res_block(x, filters=f, kernel_size=k, stride=s, block_id=bid)
            bid += 1

    # Head
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(64, activation='relu', name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="do")(x)
    outputs = layers.Dense(n_classes, activation='softmax', name="out")(x)

    return Model(inputs=inputs, outputs=outputs, name="ecg_resnet_lite")


# ===========================================================================
# Presets
# ===========================================================================

def build_ecg_resnet_lite_small(input_shape=None):
    """Small (~25K params): 3 stages for distillation student."""
    return build_ecg_resnet_lite(
        input_shape=input_shape,
        filters=(16, 32, 64),
        blocks_per_stage=(2, 2, 1),
        kernel_sizes=(5, 5, 3),
        strides=(1, 2, 2),
        dropout_rate=0.5
    )


def build_ecg_resnet_lite_medium(input_shape=None):
    """Medium (~55K params): 4 stages, recommended default."""
    return build_ecg_resnet_lite(
        input_shape, filters=(16, 32, 64, 128),
        blocks_per_stage=(2, 2, 2, 1), kernel_sizes=(5, 5, 3, 3),
        strides=(1, 2, 2, 1), dropout_rate=0.3)


def build_ecg_resnet_lite_large(input_shape=None):
    """Large (~80K params): deeper stages."""
    return build_ecg_resnet_lite(
        input_shape, filters=(16, 32, 64, 128),
        blocks_per_stage=(2, 3, 3, 1), kernel_sizes=(7, 5, 3, 3),
        strides=(1, 2, 2, 1), dropout_rate=0.4)


# ===========================================================================
# Compile & Callbacks (aligned with cnn_1d.py API)
# ===========================================================================

def compile_model(model, learning_rate=0.001, loss=None):
    """Compile with AdamW. Default: categorical_crossentropy (stable)."""
    if loss is None:
        loss = 'categorical_crossentropy'
    opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate,
                                    weight_decay=1e-4)
    model.compile(optimizer=opt, loss=loss, metrics=[
        'accuracy', tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc'),
    ])
    return model


def get_callbacks(model_name="best_resnet_lite.h5",
                  early_patience=20, lr_patience=8):
    """Training callbacks: EarlyStop + ReduceLR + Checkpoint."""
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
            monitor='val_accuracy', save_best_only=True, verbose=1),
    ]


def model_summary_table(model):
    """Per-layer param table + FP32/INT8 estimate."""
    total = model.count_params()
    trainable = sum(tf.keras.backend.count_params(w)
                    for w in model.trainable_weights)
    print(f"\n{'='*55}")
    print(f"  模型: {model.name}")
    print(f"{'='*55}")
    print(f"  {'Layer':<28} {'Output':<18} {'Params':<10}")
    print(f"  {'-'*55}")
    for layer in model.layers:
        sh = str(layer.output_shape) if hasattr(layer, 'output_shape') else '-'
        p = layer.count_params() if hasattr(layer, 'count_params') else 0
        print(f"  {layer.name:<28} {sh:<18} {p:<10,}")
    print(f"  {'-'*55}")
    print(f"  总参数:     {total:>8,}")
    print(f"  可训练:     {trainable:>8,}")
    print(f"  FP32 预估:  {total * 4 / 1024:>8.1f} KB")
    print(f"  INT8 预估:  {total * 1 / 1024:>8.1f} KB")
    print(f"{'='*55}\n")


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import numpy as np
    print("[ECG-ResNet-Lite] 模型构建测试...\n")
    for nm, bld in [("Small  ~25K", build_ecg_resnet_lite_small),
                    ("Medium ~55K", build_ecg_resnet_lite_medium),
                    ("Large  ~80K", build_ecg_resnet_lite_large)]:
        m = bld()
        print(f"--- {nm} ---")
        model_summary_table(m)
    m = build_ecg_resnet_lite_medium()
    x = np.random.randn(4, 250, 1).astype(np.float32)
    y = m.predict(x, verbose=0)
    print(f"[测试] {x.shape} → {y.shape}, 输出: {y[0]}")
    print("[ECG-ResNet-Lite] ✅ 全部通过!")
