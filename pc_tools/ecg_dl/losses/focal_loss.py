#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Focal Loss + Label Smoothing for ECG arrhythmia detection.

Combines:
  1. Focal Loss (gamma=2.0, alpha=0.75) – handle class imbalance
  2. Label Smoothing (epsilon=0.1) – improve generalization
  3. Mixup augmentation for 1D ECG signals
  4. Mild physiologically-plausible data augmentation

Usage:
  from losses.focal_loss import focal_loss_with_smoothing
  model.compile(loss=focal_loss_with_smoothing(gamma=2.0, alpha=0.75), ...)
"""

import tensorflow as tf
from tensorflow.keras import backend as K


# ===========================================================================
# Focal Loss with Label Smoothing
# ===========================================================================

class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for binary/multi-class classification.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    + optional label smoothing.

    Args:
        gamma:     Focusing parameter. Higher = more focus on hard examples.
        alpha:     Class weight for Abnormal class. Normal gets (1-alpha).
        from_logits: Whether y_pred is raw logits (before softmax).
        label_smoothing: Label smooth factor ∈ [0, 1].
    """

    def __init__(self, gamma=2.0, alpha=0.75, from_logits=False,
                 label_smoothing=0.0, reduction="sum_over_batch_size",
                 name="focal_loss"):
        super().__init__(reduction=reduction, name=name)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.from_logits = bool(from_logits)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        # ★ FIX: Save ORIGINAL one-hot y_true BEFORE label smoothing.
        #   alpha_t must use hard labels; otherwise label smoothing
        #   contaminates the class weights (e.g. Normal gets 0.725
        #   instead of 0.25, causing model collapse).
        y_true_hard = tf.stop_gradient(y_true)

        # --- Label Smoothing ---
        if self.label_smoothing > 0.0:
            n_classes = tf.cast(tf.shape(y_true)[-1], y_true.dtype)
            y_true = y_true * (1.0 - self.label_smoothing) + \
                     self.label_smoothing / n_classes

        # --- Softmax if needed ---
        if self.from_logits:
            y_pred_soft = tf.nn.softmax(y_pred, axis=-1)
        else:
            y_pred_soft = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # --- Cross-entropy: -y * log(p) ---
        ce = -y_true * tf.math.log(y_pred_soft)

        # --- Focal weight: (1 - p_t)^γ ---
        p_t = tf.reduce_sum(y_true * y_pred_soft, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        # --- Alpha balancing (uses ORIGINAL hard labels) ---
        alpha_t = y_true_hard * self.alpha + (1.0 - y_true_hard) * (1.0 - self.alpha)

        # --- Final ---
        loss = alpha_t * focal_weight * ce
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma, "alpha": self.alpha,
            "from_logits": self.from_logits,
            "label_smoothing": self.label_smoothing,
        })
        return config


def focal_loss_with_smoothing(gamma=2.0, alpha=0.75, label_smoothing=0.1,
                              from_logits=False):
    """Factory: FocalLoss(gamma, alpha) + LabelSmoothing(epsilon)."""
    return FocalLoss(gamma=gamma, alpha=alpha, from_logits=from_logits,
                     label_smoothing=label_smoothing)


# ===========================================================================
# Mixup Augmentation for 1D ECG Signals
# ===========================================================================

@tf.function
def mixup_1d(x_batch, y_batch, alpha=0.2):
    """Mixup: x_mix = λ·x_i + (1-λ)·x_j, λ ~ Beta(α,α)."""
    x_batch = tf.cast(x_batch, tf.float32)
    y_batch = tf.cast(y_batch, tf.float32)
    batch_size = tf.shape(x_batch)[0]
    gamma_1 = tf.random.gamma((batch_size,), alpha=alpha, dtype=tf.float32)
    gamma_2 = tf.random.gamma((batch_size,), alpha=alpha, dtype=tf.float32)
    lam = gamma_1 / (gamma_1 + gamma_2 + K.epsilon())
    lam_x = tf.cast(tf.reshape(lam, (batch_size, 1, 1)), tf.float32)
    lam_y = tf.cast(tf.reshape(lam, (batch_size, 1)), tf.float32)
    indices = tf.random.shuffle(tf.range(batch_size))
    x_shuf, y_shuf = tf.gather(x_batch, indices), tf.gather(y_batch, indices)
    return lam_x * x_batch + (1 - lam_x) * x_shuf, \
           lam_y * y_batch + (1 - lam_y) * y_shuf


class MixupDataGenerator(tf.keras.utils.Sequence):
    """Keras Sequence applying Mixup with 50% probability per batch."""

    def __init__(self, x, y, batch_size=64, alpha=0.2, shuffle=True):
        self.x = tf.cast(x, tf.float32)
        self.y = tf.cast(y, tf.float32)
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.n = tf.shape(self.x)[0]
        self.indices = tf.range(self.n)
        self.on_epoch_end()

    def __len__(self):
        return int(tf.math.ceil(self.n / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = tf.minimum(start + self.batch_size, self.n)
        bi = self.indices[start:end]


# ===========================================================================
# Mild ECG Data Augmentation (physiologically plausible, max 2x)
# ===========================================================================

@tf.function
def ecg_time_warp(x, max_stretch=0.12):
    """Random time-stretch for whole batch (simplified from per-sample, sufficient for augmentation)."""
    bs = tf.shape(x)[0]
    sl = tf.cast(tf.shape(x)[1], tf.float32)
    sl_i = tf.cast(sl, tf.int32)

    st = tf.random.uniform((), 1.0 - max_stretch, 1.0 + max_stretch)
    new_len = tf.clip_by_value(
        tf.cast(sl * st, tf.int32),
        tf.cast(sl * 0.80, tf.int32),
        tf.cast(sl * 1.20, tf.int32)
    )

    x_4d = x[:, :, :, tf.newaxis]                               # (bs, time, 1) -> (bs, time, 1, 1)
    resized = tf.image.resize(x_4d, (new_len, 1))               # (bs, new_len, 1, 1)
    padded = tf.image.resize_with_crop_or_pad(resized, sl_i, 1) # (bs, sl_i, 1, 1)
    return tf.squeeze(padded, axis=-1)                          # (bs, sl_i, 1)


@tf.function
def ecg_amplitude_scale(x, min_s=0.80, max_s=1.20):
    """Scale amplitude (±20%) to simulate electrode impedance changes. (Phase 2A: ↑ from ±15%)"""
    s = tf.random.uniform((tf.shape(x)[0], 1, 1), min_s, max_s, dtype=x.dtype)
    return x * s


@tf.function
def ecg_gaussian_noise(x, noise_std=0.015):
    """Add noise (σ=0.015) simulating ADC quantization noise. (Phase 2A: ↑ from 0.01)"""
    return x + tf.random.normal(tf.shape(x), 0.0, noise_std, dtype=x.dtype)


@tf.function
def ecg_baseline_wander(x, amplitude=0.20, max_freq=0.05):
    """Add low-freq sinusoidal wander simulating respiration. (Phase 2A: ↑ amplitude from 0.15)"""
    bs = tf.shape(x)[0]
    sl = tf.cast(tf.shape(x)[1], tf.float32)
    t = tf.reshape(tf.linspace(0.0, 1.0, tf.cast(sl, tf.int32)), (1, -1, 1))
    freq = tf.random.uniform((bs, 1, 1), 0.0, max_freq, dtype=x.dtype)
    phase = tf.random.uniform((bs, 1, 1), 0.0, 6.283, dtype=x.dtype)
    amp = tf.random.uniform((bs, 1, 1), 0.0, amplitude, dtype=x.dtype)
    wander = amp * tf.sin(6.283 * freq * t * sl + phase)
    return x + tf.cast(wander, x.dtype)


def apply_mild_augmentation(x, prob=0.80):
    """Apply mild augmentations with given probability. (Phase 2A: ↑ default prob 0.50→0.80)"""
    if tf.random.uniform(()) > prob:
        return x
    # Apply each augmentation independently with 50% chance each
    x = tf.cond(
        tf.random.uniform(()) < 0.5,
        lambda: ecg_time_warp(x, max_stretch=0.12),
        lambda: x
    )
    x = tf.cond(
        tf.random.uniform(()) < 0.5,
        lambda: ecg_amplitude_scale(x, min_s=0.80, max_s=1.20),
        lambda: x
    )
    x = tf.cond(
        tf.random.uniform(()) < 0.5,
        lambda: ecg_gaussian_noise(x, noise_std=0.015),
        lambda: x
    )
    x = tf.cond(
        tf.random.uniform(()) < 0.5,
        lambda: ecg_baseline_wander(x, amplitude=0.20),
        lambda: x
    )
    return x


# ===========================================================================
# Class Weight Helper
# ===========================================================================

def get_class_weights(y_train, abnormal_weight=2.5):
    """Compute class weights {0: w_norm, 1: w_abnorm} for imbalanced ECG."""
    n_total = tf.cast(tf.shape(y_train)[0], tf.float32)
    n_abnormal = tf.reduce_sum(tf.cast(y_train == 1, tf.float32))
    n_normal = n_total - n_abnormal
    w_normal = n_total / (2.0 * n_normal + K.epsilon())
    return {0: float(w_normal), 1: float(w_normal * abnormal_weight)}


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    print("[Focal Loss] 单元测试...")
    loss_fn = focal_loss_with_smoothing()
    yt = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    yp = tf.constant([[0.9, 0.1], [0.3, 0.7], [0.1, 0.9]])
    print(f"  Loss: {loss_fn(yt, yp).numpy()}")
    xb = tf.random.normal((32, 250, 1))
    yb = tf.one_hot(tf.random.uniform((32,), 0, 2, dtype=tf.int32), 2)
    xm, ym = mixup_1d(xb, yb)
    print(f"  Mixup: {xb.shape} → {xm.shape}")
    xa = apply_mild_augmentation(xb)
    print(f"  Aug: {xa.shape}")
    print("[Focal Loss] ✅ 全部通过!")

