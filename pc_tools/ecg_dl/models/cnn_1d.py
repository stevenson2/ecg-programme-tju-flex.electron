#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D-CNN 模型定义 - 轻量级 ECG 异常检测

模型结构:
  Input(250,1) → Conv1D(8,k5) → BN → MaxPool(2)
               → Conv1D(8,k3) → BN → MaxPool(2)
               → Conv1D(16,k3) → GAP
               → Dense(16) → Dropout(0.3) → Dense(2,Softmax)

总参数量: ~5,200 (INT8 量化后 ~5.2KB)
推理时间: ~4-7ms @240MHz (ESP32-S3)
"""

import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import INFERENCE_CONFIG, TRAIN_CONFIG


def build_ecg_cnn_1d(
    input_shape: tuple = None,
    n_classes: int = 2
) -> tf.keras.Model:
    """
    构建超轻量 1D-CNN 模型
    
    Args:
        input_shape: 输入形状 (window_size, 1), 默认 (250, 1)
        n_classes: 分类数, 默认 2 (正常/异常)
        
    Returns:
        tf.keras.Model
    """
    if input_shape is None:
        input_shape = (INFERENCE_CONFIG['window_size'], 1)
    
    inputs = layers.Input(shape=input_shape, name="ecg_input")
    
    # Block 1: Conv1D 8 filters, kernel=5
    x = layers.Conv1D(
        filters=8,
        kernel_size=5,
        padding='same',
        activation=None,
        name='conv1d_1'
    )(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.ReLU(name='relu_1')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_1')(x)
    
    # Block 2: Conv1D 8 filters, kernel=3
    x = layers.Conv1D(
        filters=8,
        kernel_size=3,
        padding='same',
        activation=None,
        name='conv1d_2'
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.ReLU(name='relu_2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_2')(x)
    
    # Block 3: Conv1D 16 filters, kernel=3 + GAP
    x = layers.Conv1D(
        filters=16,
        kernel_size=3,
        padding='same',
        activation=None,
        name='conv1d_3'
    )(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.ReLU(name='relu_3')(x)
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    
    # Classifier
    x = layers.Dense(16, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='dense_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='ecg_cnn_1d')
    
    return model



def build_ecg_cnn_1d_v2(
    input_shape: tuple = None,
    n_classes: int = 2
) -> tf.keras.Model:
    """
    增强版 1D-CNN: 更多滤波器, 更好的泛化
    参数量: ~15K, INT8 ~15KB
    """
    if input_shape is None:
        input_shape = (INFERENCE_CONFIG['window_size'], 1)
    
    inputs = layers.Input(shape=input_shape, name="ecg_input")
    
    # Block 1: Conv1D 16 filters, kernel=7
    x = layers.Conv1D(filters=16, kernel_size=7, padding='same', name='conv1d_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.ReLU(name='relu_1')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_1')(x)
    
    # Block 2: Conv1D 32 filters, kernel=5
    x = layers.Conv1D(filters=32, kernel_size=5, padding='same', name='conv1d_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.ReLU(name='relu_2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_2')(x)
    
    # Block 3: Conv1D 64 filters, kernel=3
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', name='conv1d_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.ReLU(name='relu_3')(x)
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    
    # Classifier
    x = layers.Dense(32, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.4, name='dropout')(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='dense_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='ecg_cnn_1d_v2')
    return model


def build_ecg_cnn_1d_v3(
    input_shape=None,
    n_classes=2,
    dropout_rate=0.4
) -> tf.keras.Model:
    """
    CNN v3: scaled-up standard convolutions (~30K params).
    No depthwise separable — stable on small datasets.
    """
    if input_shape is None:
        input_shape = (INFERENCE_CONFIG['window_size'], 1)

    inputs = layers.Input(shape=input_shape, name="ecg_input")

    # Block 1: Conv1D 32, k7 → BN → ReLU → Pool → Dropout
    x = layers.Conv1D(32, 7, padding='same', name='c1')(inputs)
    x = layers.BatchNormalization(name='b1')(x)
    x = layers.ReLU(name='r1')(x)
    x = layers.MaxPooling1D(2, name='p1')(x)
    x = layers.Dropout(0.15, name='d1')(x)

    # Block 2: Conv1D 64, k5
    x = layers.Conv1D(64, 5, padding='same', name='c2')(x)
    x = layers.BatchNormalization(name='b2')(x)
    x = layers.ReLU(name='r2')(x)
    x = layers.MaxPooling1D(2, name='p2')(x)
    x = layers.Dropout(0.2, name='d2')(x)

    # Block 3: Conv1D 96, k3
    x = layers.Conv1D(96, 3, padding='same', name='c3')(x)
    x = layers.BatchNormalization(name='b3')(x)
    x = layers.ReLU(name='r3')(x)
    x = layers.MaxPooling1D(2, name='p3')(x)
    x = layers.Dropout(0.25, name='d3')(x)

    # Block 4: Conv1D 128, k3 → GAP
    x = layers.Conv1D(128, 3, padding='same', name='c4')(x)
    x = layers.BatchNormalization(name='b4')(x)
    x = layers.ReLU(name='r4')(x)
    x = layers.GlobalAveragePooling1D(name='gap')(x)

    # Classifier
    x = layers.Dense(64, activation='relu', name='fc1')(x)
    x = layers.Dropout(dropout_rate, name='do')(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='out')(x)

    return Model(inputs=inputs, outputs=outputs, name='ecg_cnn_1d_v3')


def build_ecg_cnn_1d_tiny(
    input_shape: tuple = None,
    n_classes: int = 2
) -> tf.keras.Model:
    """
    更轻量的版本（仅2层卷积, 适合极端资源受限）
    
    Args:
        input_shape: 输入形状 (window_size, 1)
        n_classes: 分类数
        
    Returns:
        tf.keras.Model
    """
    if input_shape is None:
        input_shape = (INFERENCE_CONFIG['window_size'], 1)
    
    inputs = layers.Input(shape=input_shape, name="ecg_input")
    
    # Block 1: Conv1D 4 filters, kernel=5
    x = layers.Conv1D(4, 5, padding='same', activation='relu', name='conv1d_1')(inputs)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_1')(x)
    
    # Block 2: Conv1D 8 filters, kernel=3
    x = layers.Conv1D(8, 3, padding='same', activation='relu', name='conv1d_2')(x)
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    
    # Classifier
    x = layers.Dense(8, activation='relu', name='dense_1')(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='dense_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='ecg_cnn_1d_tiny')
    
    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = None,
    loss = None
) -> tf.keras.Model:
    """
    编译模型
    
    Args:
        model: 未编译的模型
        learning_rate: 学习率
        loss: 损失函数 (默认 categorical_crossentropy)
        
    Returns:
        编译后的模型
    """
    if learning_rate is None:
        learning_rate = TRAIN_CONFIG['learning_rate']
    
    if loss is None:
        fl_cfg = TRAIN_CONFIG.get('focal_loss', {})
        if fl_cfg.get('enabled', False):
            from losses.focal_loss import FocalLoss
            loss = FocalLoss(
                gamma=fl_cfg.get('gamma', 1.0),
                alpha=fl_cfg.get('alpha', 0.75),
                label_smoothing=fl_cfg.get('label_smoothing', 0.0),
                from_logits=False,
            )
            print(f"[编译] 使用 FocalLoss (γ={loss.gamma}, α={loss.alpha}, "
                  f"LS={loss.label_smoothing})")
        else:
            loss = 'categorical_crossentropy'
            print("[编译] 使用 CategoricalCrossentropy")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
        ]
    )
    
    return model


def get_callbacks() -> list:
    """
    获取训练回调函数
    
    Returns:
        回调列表
    """
    from config import MODELS_DIR
    
    callbacks = [
        # 早停
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=TRAIN_CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        # 学习率衰减
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=TRAIN_CONFIG['reduce_lr_factor'],
            patience=TRAIN_CONFIG['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        ),
        # 模型检查点 (最佳模型)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / 'best_model.h5'),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
    ]

    return callbacks


def model_summary_table(model: tf.keras.Model):
    """
    打印模型参数量统计
    """
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) 
        for w in model.trainable_weights
    )
    
    print(f"\n{'='*50}")
    print(f"模型: {model.name}")
    print(f"{'='*50}")
    
    # 逐层统计
    print(f"{'Layer':<25} {'Output Shape':<20} {'Params':<10}")
    print(f"{'-'*55}")
    for layer in model.layers:
        if hasattr(layer, 'output_shape'):
            shape = str(layer.output_shape)
        else:
            shape = '-'
        params = layer.count_params() if hasattr(layer, 'count_params') else 0
        print(f"{layer.name:<25} {shape:<20} {params:<10}")
    
    print(f"{'-'*55}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"非训练参数: {total_params - trainable_params:,}")
    
    # 预估模型大小
    fp32_size = total_params * 4 / 1024  # KB
    int8_size = total_params * 1 / 1024  # KB
    print(f"\n预估模型大小:")
    print(f"  FP32: {fp32_size:.1f} KB")
    print(f"  INT8: {int8_size:.1f} KB")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    print("[模型] 构建 1D-CNN 模型...")
    model = build_ecg_cnn_1d()
    model = compile_model(model)
    model_summary_table(model)
    
    # 测试前向传播
    import numpy as np
    dummy_input = np.random.randn(4, 250, 1).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    print(f"[模型] 前向传播测试: {dummy_input.shape} → {output.shape}")
    print(f"[模型] ✅ 模型构建成功")