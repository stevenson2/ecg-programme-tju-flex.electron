#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
滑窗增强崩溃机制锁定 (Phase 3A 诊断 3)

关键对照:
  E: dup=1 但 max_shift=0 (纯复制, 不移位) -> 若是类别比例问题, E 也会崩
  F: dup=1 max_shift=40 (复现崩溃) + 手动 sklearn AUC 验证 fit 期 val_auc
     并分别评估 训练集原始拍/变体/验证集, 观察模型对"居中拍"的行为
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

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
n_orig = len(splits["train"][0])


def build_and_train(dup, max_shift, tag):
    x_tr, y_tr = apply_sliding_window_augmentation(
        splits["train"][0], splits["train"][1], dup=dup, max_shift=max_shift, seed=42)
    train_ds = make_tf_dataset(x_tr, y_tr, batch_size=64, shuffle=True, augment=False)
    val_ds = make_tf_dataset(x_val, y_val, batch_size=64, shuffle=False)
    model = build_ecg_resnet_lite_large((250, 1))
    model = compile_model(model, learning_rate=5e-4)
    hist = model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=0)
    print(f"\n=== {tag} (dup={dup}, max_shift={max_shift}) ===")
    print(f"  fit val_acc: {[round(v,4) for v in hist.history['val_accuracy']]}")
    print(f"  fit val_auc: {[round(v,4) for v in hist.history['val_auc']]}")
    # 手动 AUC (独立于 Keras metric)
    p_val = model.predict(x_val_in, verbose=0)[:, 1]
    print(f"  手动 val AUC: {roc_auc_score(y_val, p_val):.4f}")
    print(f"  val: P(abn|正常)={p_val[y_val==0].mean():.4f}  "
          f"P(abn|异常)={p_val[y_val==1].mean():.4f}")
    # 在训练集原始拍上的行为 (与 val 同为居中拍)
    p_tr = model.predict(add_channel_dim(splits["train"][0]), verbose=0)[:, 1]
    y_tr_orig = splits["train"][1]
    print(f"  train原始拍: P(abn|正常)={p_tr[y_tr_orig==0].mean():.4f}  "
          f"P(abn|异常)={p_tr[y_tr_orig==1].mean():.4f}")
    return model


# E: 纯复制 (不移动 R 峰) — 隔离"类别比例" vs "移位结构"
build_and_train(1, 0, "E")
# F: 移位 40 (复现崩溃) + 手动 AUC 对照
build_and_train(1, 40, "F")
print("\n[诊断3完成]")
