#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
滑窗增强修复验证 (Phase 3A 诊断 4)

修复假设: 只对异常类移位 -> 相位成为类别捷径 (模型退化)
修复方案: 对正常类+异常类同等移位, 相位不再携带类别信息

  G: 双类移位 dup=1 shift=40 (无 batch 增强)   -> 验证修复
  H: 双类移位 dup=1 shift=40 (完整管线+增强)   -> 生产配置验证
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


def dual_shift(x, y, dup=1, max_shift=40, seed=42):
    """对全部样本 (两类) 做滑窗移位, 变体保留原标签."""
    rng = np.random.default_rng(seed)
    n = len(x)
    pad = max_shift
    xp = np.pad(x, ((0, 0), (pad, pad)), mode='reflect')
    rows = np.arange(n)
    cols = np.arange(250)
    variants = []
    for _ in range(dup):
        mag = rng.integers(1, max_shift + 1, size=n)
        sign = rng.choice([-1, 1], size=n)
        start = pad + mag * sign
        variants.append(xp[rows[:, None], start[:, None] + cols[None, :]])
    x_v = np.concatenate(variants, axis=0).astype(np.float32)
    x_v = (x_v - x_v.mean(axis=1, keepdims=True)) / (
        x_v.std(axis=1, keepdims=True) + 1e-8)
    y_v = np.concatenate([y] * dup, axis=0)
    return np.concatenate([x, x_v], axis=0), np.concatenate([y, y_v], axis=0)


data = load_mit_incart_merged()
splits = train_val_test_split(data["beats"], data["labels"], record_ids=data["record_ids"])
x_val, y_val = splits["val"]
x_val_in = add_channel_dim(x_val)


def run(dup, max_shift, augment, tag):
    x_tr, y_tr = dual_shift(splits["train"][0], splits["train"][1],
                            dup=dup, max_shift=max_shift)
    print(f"[{tag}] 训练集: {len(x_tr)} 样本, 异常占比 {np.mean(y_tr==1)*100:.1f}%")
    train_ds = make_tf_dataset(x_tr, y_tr, batch_size=64, shuffle=True, augment=augment)
    val_ds = make_tf_dataset(x_val, y_val, batch_size=64, shuffle=False)
    model = build_ecg_resnet_lite_large((250, 1))
    model = compile_model(model, learning_rate=5e-4)
    hist = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0)
    p = model.predict(x_val_in, verbose=0)[:, 1]
    print(f"=== {tag} (dup={dup}, shift={max_shift}, augment={augment}) ===")
    print(f"  fit val_auc: {[round(v,4) for v in hist.history['val_auc']]}")
    print(f"  手动 val AUC: {roc_auc_score(y_val, p):.4f}")
    print(f"  val: P(abn|正常)={p[y_val==0].mean():.4f}  "
          f"P(abn|异常)={p[y_val==1].mean():.4f}")


# G: 双类移位, 无 batch 增强
run(1, 40, False, "G")
# H: 双类移位, 完整管线 (与生产一致)
run(1, 40, True, "H")
print("\n[诊断4完成]")
