#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最终模型评估: exp2'(CE+滑窗dup2) vs exp3(FocalLoss) vs P2A部署模型"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import load_mit_incart_merged, train_val_test_split, add_channel_dim

MODELS = Path(__file__).resolve().parent / "models"

data = load_mit_incart_merged()
splits = train_val_test_split(data["beats"], data["labels"], record_ids=data["record_ids"])
x_test, y_test = splits["test"]
x_test_in = add_channel_dim(x_test)
print(f"test: {len(x_test)} (N={(y_test==0).sum()}, A={(y_test==1).sum()})\n")


def evaluate(name, path):
    model = tf.keras.models.load_model(str(path), compile=False)
    probs = model.predict(x_test_in, verbose=0)[:, 1]
    auc = tf.keras.metrics.AUC()(y_test, probs).numpy()

    def at(th):
        pred = (probs >= th).astype(int)
        tp = ((pred == 1) & (y_test == 1)).sum()
        fp = ((pred == 1) & (y_test == 0)).sum()
        fn = ((pred == 0) & (y_test == 1)).sum()
        rec = tp / max(1, tp + fn)
        prec = tp / max(1, tp + fp)
        acc = (pred == y_test).mean()
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        return acc, prec, rec, f1

    a05, p05, r05, f05 = at(0.50)
    a035, p035, r035, f035 = at(0.35)
    print(f"{name:<22} | AUC {auc:.4f} | "
          f"@0.5  P {p05:.3f} R {r05:.3f} F1 {f05:.3f} | "
          f"@0.35 P {p035:.3f} R {r035:.3f} F1 {f035:.3f}")


cands = [
    ("exp2' CE+dup2", MODELS / "final_resnet_l_exp2_dual_dup2.h5"),
    ("exp3 FocalLoss", MODELS / "final_resnet_l_exp3_focal_a075.h5"),
    ("P2A 部署模型", MODELS / "archived" / "final_resnet_l_p2a_backup.h5"),
]
for name, p in cands:
    if p.exists():
        evaluate(name, p)
    else:
        print(f"{name}: 不存在 ({p})")
