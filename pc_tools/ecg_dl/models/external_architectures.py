#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
external_architectures.py — 跨架构失配对照实验的外部模型定义
================================================================
三个不属于本项目 ResNet-Lite 家族的外部架构：
  1. lstm_cnn      : 张异凡式 LSTM+CNN 并行组合（单导联 250 点适配版）
  2. cnn_standard  : 标准 1D-CNN（无残差、无深度可分离）
  3. resnet1d      : 通用 ResNet-1d（标准 BasicBlock + 跳连）

所有模型输入 (250,1)，输出 2 类 softmax。
训练/评估统一使用项目既有 FocalLoss/AdamW 协议。
"""
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_lstm_cnn_parallel(input_shape=(250, 1), n_classes=2, seed=42):
    """张异凡式 LSTM+CNN 并行组合。

    CNN 分支: 2 层 Conv1D + BN + ReLU + MaxPool + GAP
    LSTM 分支: 2 层 LSTM
    最后 concat -> FC -> softmax。
    """
    tf.keras.utils.set_random_seed(seed)
    inputs = layers.Input(shape=input_shape, name="input")

    # ---- CNN 分支 ----
    c = layers.Conv1D(64, 7, padding="same", name="cnn_conv1")(inputs)
    c = layers.BatchNormalization(name="cnn_bn1")(c)
    c = layers.ReLU(name="cnn_relu1")(c)
    c = layers.MaxPooling1D(pool_size=2, name="cnn_pool1")(c)

    c = layers.Conv1D(128, 5, padding="same", name="cnn_conv2")(c)
    c = layers.BatchNormalization(name="cnn_bn2")(c)
    c = layers.ReLU(name="cnn_relu2")(c)
    c = layers.MaxPooling1D(pool_size=2, name="cnn_pool2")(c)
    c = layers.GlobalAveragePooling1D(name="cnn_gap")(c)
    c = layers.Dense(64, activation="relu", name="cnn_fc")(c)

    # ---- LSTM 分支 ----
    l = layers.LSTM(64, return_sequences=True, name="lstm_1")(inputs)
    l = layers.LSTM(32, name="lstm_2")(l)

    # ---- 并行合并 ----
    x = layers.Concatenate(name="concat")([c, l])
    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = Model(inputs, outputs, name="lstm_cnn_parallel")
    return model


def build_cnn_standard(input_shape=(250, 1), n_classes=2, seed=42):
    """标准 1D-CNN：无残差、无深度可分离。

    4 层卷积块 + GAP + FC。
    """
    tf.keras.utils.set_random_seed(seed)
    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Conv1D(32, 7, padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = layers.Conv1D(64, 5, padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    x = layers.Conv1D(128, 3, padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool3")(x)

    x = layers.Conv1D(256, 3, padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = Model(inputs, outputs, name="cnn_standard")
    return model


def _residual_block(x, filters, kernel_size=3, stride=1, name="res"):
    """标准 ResNet-1d BasicBlock：conv1-bn-relu-conv2-bn + skip."""
    shortcut = x
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding="same",
                                 use_bias=False, name=f"{name}_shortcut_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)

    y = layers.Conv1D(filters, kernel_size, strides=stride, padding="same",
                      use_bias=False, name=f"{name}_conv1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.ReLU(name=f"{name}_relu1")(y)

    y = layers.Conv1D(filters, kernel_size, padding="same", use_bias=False,
                      name=f"{name}_conv2")(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)

    y = layers.Add(name=f"{name}_add")([y, shortcut])
    y = layers.ReLU(name=f"{name}_relu2")(y)
    return y


def build_resnet1d(input_shape=(250, 1), n_classes=2, seed=42):
    """通用 ResNet-1d：stem + 3 个 BasicBlock + GAP + FC。"""
    tf.keras.utils.set_random_seed(seed)
    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Conv1D(64, 7, strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding="same", name="stem_pool")(x)

    x = _residual_block(x, 64, stride=1, name="block1")
    x = _residual_block(x, 128, stride=2, name="block2")
    x = _residual_block(x, 256, stride=2, name="block3")

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = Model(inputs, outputs, name="resnet1d")
    return model


ARCHITECTURES = {
    "lstm_cnn": build_lstm_cnn_parallel,
    "cnn_standard": build_cnn_standard,
    "resnet1d": build_resnet1d,
}


def print_summary(model):
    total = model.count_params()
    print(f"\n[模型] {model.name} 总参数: {total:,}")
    model.summary(line_length=110)


if __name__ == "__main__":
    for name, builder in ARCHITECTURES.items():
        m = builder()
        print_summary(m)