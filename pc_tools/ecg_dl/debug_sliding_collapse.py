#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
滑窗增强验证集崩溃根因诊断 (Phase 3A)

隔离实验:
  B: 纯滑窗增强 dup=1 (无 batch 增强/mixup)  -> 滑窗本身是否致崩
  C: baseline (无滑窗, 有 batch 增强)          -> 对照组
  D: 滑窗 dup=1 但 max_shift=20 (减半偏移)      -> 剂量效应

观察指标: val_acc/val_auc 曲线 + 预测偏差
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import (
    load_mit_incart_merged, train_val_test_split,
    apply_sliding_window_augmentation, make_tf_dataset, add_channel_dim,
)
from models.resnet_lite_1d import build_ecg_resnet_lite_large, compile_model

data = load_mit_incart_merged()
splits = train_val_test_split(data["beats"], data["labels"], record_ids=data["record_ids"])
x_val, y_val = splits["val"]
x_val_in = add_channel_dim(x_val)


def run(epochs, dup, augment, max_shift, tag):
    if dup > 0:
        x_tr, y_tr = apply_sliding_window_augmentation(
            splits["train"][0], splits["train"][1],
            dup=dup, max_shift=max_shift, seed=42)
    else:
        x_tr, y_tr = splits["train"]
    train_ds = make_tf_dataset(x_tr, y_tr, batch_size=64, shuffle=True, augment=augment)
    val_ds = make_tf_dataset(x_val, y_val, batch_size=64, shuffle=False)
    model = build_ecg_resnet_lite_large((250, 1))
    model = compile_model(model, learning_rate=5e-4)
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)
    p = model.predict(x_val_in, verbose=0)[:, 1]
    print(f"\n=== {tag} (dup={dup}, augment={augment}, max_shift={max_shift}) ===")
    print(f"  val_acc: {[round(v,4) for v in hist.history['val_accuracy']]}")
    print(f"  val_auc: {[round(v,4) for v in hist.history['val_auc']]}")
    print(f"  NaN preds: {np.isnan(p).sum()}")
    print(f"  P(abn|true=Normal)={p[y_val==0].mean():.4f}  "
          f"P(abn|true=Abnormal)={p[y_val==1].mean():.4f}")


# B: 纯滑窗, 无 batch 增强
run(3, 1, False, 40, "B")
# C: baseline 对照 (无滑窗, 有增强)
run(3, 0, True, 40, "C")
# D: 滑窗 max_shift=20 (偏移减半)
run(3, 1, False, 20, "D")
print("\n[诊断完成]")
