#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-task combined loss for Route F.

L_total = w_cls * L_cls(Normal/Abnormal) + w_bpm * L_bpm(BPM regression)
          + w_sqi * L_sqi(SQI regression)

L_cls: FocalLoss  (binary cross-entropy with class reweighting)
L_bpm: MSE loss   (heart rate regression)
L_sqi: MSE loss   (signal quality regression)
"""

import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import backend as K

try:
    from .focal_loss import FocalLoss
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from losses.focal_loss import FocalLoss


class MultiTaskLoss(tf.keras.losses.Loss):
    """
    Weighted sum of classification + regression losses.

    Args:
        gamma, alpha:     FocalLoss parameters for classification head.
        w_cls, w_bpm, w_sqi: Loss weights per task.
        label_smoothing:  Label smooth factor for classification.
    """

    def __init__(self, gamma=1.0, alpha=0.75,
                 w_cls=1.0, w_bpm=0.3, w_sqi=0.2,
                 label_smoothing=0.0, reduction="sum_over_batch_size",
                 name="multitask_loss"):
        super().__init__(reduction=reduction, name=name)
        self.focal = FocalLoss(gamma=gamma, alpha=alpha,
                               label_smoothing=label_smoothing,
                               reduction="sum_over_batch_size")
        self.w_cls = float(w_cls)
        self.w_bpm = float(w_bpm)
        self.w_sqi = float(w_sqi)
        self.mse = tf.keras.losses.MeanSquaredError(
            reduction="sum_over_batch_size")

    def call(self, y_true, y_pred):
        y_cls_true = y_true[:, :2]
        y_bpm_true = y_true[:, 2:3]
        y_sqi_true = y_true[:, 3:4]

        y_cls_pred = y_pred[:, :2]
        y_bpm_pred = y_pred[:, 2:3]
        y_sqi_pred = y_pred[:, 3:4]

        loss_cls = self.focal(y_cls_true, y_cls_pred)
        loss_bpm = self.mse(y_bpm_true, y_bpm_pred)
        loss_sqi = self.mse(y_sqi_true, y_sqi_pred)

        return self.w_cls * loss_cls + self.w_bpm * loss_bpm + self.w_sqi * loss_sqi

    def get_config(self):
        config = super().get_config()
        config.update({
            "w_cls": self.w_cls, "w_bpm": self.w_bpm, "w_sqi": self.w_sqi,
        })
        return config


def bpm_mae_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from losses.focal_loss import FocalLoss
    import numpy as np
    loss_fn = MultiTaskLoss(gamma=1.0, alpha=0.75)
    yt = np.array([[1.0, 0.0, 72.0, 0.85],
                    [0.0, 1.0, 88.0, 0.60]], dtype=np.float32)
    yp = np.array([[0.9, 0.1, 70.0, 0.80],
                    [0.3, 0.7, 85.0, 0.55]], dtype=np.float32)
    print(f"Multi-task loss: {loss_fn(yt, yp).numpy():.4f}")
    print("[multitask_loss] OK")
