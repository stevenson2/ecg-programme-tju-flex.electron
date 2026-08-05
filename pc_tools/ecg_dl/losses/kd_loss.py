#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Distillation Loss for ECG anomaly detection.

Implements the soft-target distillation from Hinton et al. 2015:
  L = (1-alpha) * CE(y_true, y_pred) + alpha * T^2 * KL(t_soft || s_soft)

Inputs (packed into y_true by train_kd.py):
  y_true: (B, 4) = [onehot(2) | teacher_logits(2)]
  y_pred: (B, 2) = student softmax probabilities

Usage:
  from losses.kd_loss import make_kd_loss, kd_loss, SlicedAUC
  model.compile(loss=make_kd_loss(alpha=0.5, temperature=3), ...)
  # Or use the default-configured kd_loss(alpha=0.5, T=3) directly.
"""

import tensorflow as tf


def _kd_loss_impl(y_true, y_pred, alpha: float, temperature: float):
    """Core KD loss math — shared by make_kd_loss closure and module-level kd_loss.

    Args:
        y_true: (B, 4)  — concat([onehot(2), teacher_logits(2)], axis=-1).
        y_pred: (B, 2)  — student softmax probabilities.
        alpha:  blending weight for the KL (teacher) term.
        temperature: softening temperature.

    Returns:
        (B,) per-sample loss (not reduced to scalar).
    """
    # --- Hard-target cross-entropy (per-sample, shape (B,)) ---
    ce = tf.keras.losses.categorical_crossentropy(y_true[:, :2], y_pred)

    # --- Student softened distribution (power form, avoids log-through-clip) ---
    T = temperature
    s_soft = y_pred ** (1.0 / T)
    s_soft = s_soft / tf.reduce_sum(s_soft, axis=-1, keepdims=True)

    # --- Teacher softened distribution ---
    t_soft = tf.nn.softmax(y_true[:, 2:] / T)

    # --- KL divergence: KL(t || s) = sum_i t_i * (log t_i - log s_i) ---
    t_log = tf.math.log(tf.clip_by_value(t_soft, 1e-7, 1.0))
    s_log = tf.math.log(tf.clip_by_value(s_soft, 1e-7, 1.0))
    kl = tf.reduce_sum(t_soft * (t_log - s_log), axis=-1)  # (B,)

    # --- Combined loss (Hinton 2015: scale KL by T^2) ---
    return (1.0 - alpha) * ce + alpha * kl * (T * T)


def make_kd_loss(alpha: float, temperature: float):
    """Return a KD loss callable bound to *alpha* and *temperature*.

    The returned function has signature ``kd_loss(y_true, y_pred) -> (B,)``
    so Keras ``sample_weight`` broadcasts correctly.
    """
    def loss(y_true, y_pred):
        return _kd_loss_impl(y_true, y_pred, alpha, temperature)
    return loss


# Module-level kd_loss with default alpha=0.5, temperature=3.
# Importable as: from losses.kd_loss import kd_loss
kd_loss = make_kd_loss(alpha=0.5, temperature=3.0)


class SlicedAUC(tf.keras.metrics.AUC):
    """AUC metric that slices ``y_true[:, :2]`` before evaluation.

    ``train_kd.py`` packs teacher logits into columns 2-3 of *y_true*;
    the AUC metric should only see the one-hot class columns 0-1.

    Instantiate as ``SlicedAUC(name='auc')`` so Keras reports ``auc`` /
    ``val_auc``.
    """

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, :2], y_pred, sample_weight)
