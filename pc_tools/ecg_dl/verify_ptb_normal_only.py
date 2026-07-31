#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3B 验证: 只加 PTB 正常拍 (10.4K) 进训练集, 3-epoch 冒烟
目的: 确认污染源是否为 PTB 异常拍 (MI 形态/记录级标签)
判定: val_auc 正常攀升(0.85+) = PTB 正常拍安全; 仍崩 = PTB 整体不可用
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import (
    load_mit_incart_merged, load_ptb_data, train_val_test_split,
    make_tf_dataset, add_channel_dim,
)
from models.resnet_lite_1d import build_ecg_resnet_lite_large, compile_model

data = load_mit_incart_merged()
splits = train_val_test_split(data["beats"], data["labels"], record_ids=data["record_ids"])
x_val, y_val = splits["val"]
x_val_in = add_channel_dim(x_val)

ptb = load_ptb_data()
mask_n = ptb["labels"] == 0
x_ptb_n = ptb["beats"][mask_n]
print(f"PTB 正常拍: {len(x_ptb_n)}")

x_tr = np.concatenate([splits["train"][0], x_ptb_n], axis=0)
y_tr = np.concatenate([splits["train"][1], np.zeros(len(x_ptb_n), dtype=np.int32)], axis=0)
print(f"训练集: {len(x_tr)} (abn {np.mean(y_tr==1)*100:.1f}%)")

train_ds = make_tf_dataset(x_tr, y_tr, batch_size=64, shuffle=True, augment=True)
val_ds = make_tf_dataset(x_val, y_val, batch_size=64, shuffle=False)

model = build_ecg_resnet_lite_large((250, 1))
model = compile_model(model, learning_rate=5e-4, loss='categorical_crossentropy')
hist = model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=1)

p = model.predict(x_val_in, verbose=0)[:, 1]
print(f"\n手动 val AUC: {roc_auc_score(y_val, p):.4f}")
print(f"P(abn|true=0)={p[y_val==0].mean():.4f}  P(abn|true=1)={p[y_val==1].mean():.4f}")
print(f"fit val_auc: {[round(v,4) for v in hist.history['val_auc']]}")
