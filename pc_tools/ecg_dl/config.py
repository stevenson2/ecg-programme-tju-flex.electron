#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG 异常检测 — 全局配置
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"         # MIT-BIH 原始数据
PROCESSED_DIR = DATA_DIR / "processed"  # 预处理后的 NumPy 数据
MODELS_DIR = PROJECT_ROOT / "models"    # 训练好的模型文件

# 自动创建目录
for d in [RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ======================== 数据集配置 ========================
# MIT-BIH 数据集 (PhysioNet 官方源)
MIT_BIH_LOCAL_DIR = RAW_DATA_DIR / "mit-bih-arrhythmia-database"
MIT_BIH_TEST_DIR = RAW_DATA_DIR / "mitdb_test"  # 最小测试集 (3条记录)

# MIT-BIH 测试记录 (用于快速验证)
MIT_BIH_TEST_RECORDS = [100, 105, 200]

# MIT-BIH 记录列表 (标准48条)
MIT_BIH_RECORDS = [
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
    122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
    209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
    222, 223, 228, 230, 231, 232, 233, 234
]

# ======================== 预处理配置 ========================
TARGET_FS = 250          # 目标采样率 (Hz) - 与 ESP32 匹配
ORIGINAL_FS = 360        # MIT-BIH 原始采样率
BEAT_WINDOW_MS = 1000    # 每个心拍的窗口长度 (ms)
BEAT_WINDOW_SAMPLES = TARGET_FS * BEAT_WINDOW_MS // 1000  # 250 点

# ======================== 标签映射 ========================
# MIT-BIH AAMI 标准心拍分类 -> 二分类 (Normal=0, Abnormal=1)
# 参考: AAMI EC57 标准
AAMI_CLASSES = {
    'N': 0,   # Normal beat
    'L': 0,   # Left bundle branch block beat
    'R': 0,   # Right bundle branch block beat
    'e': 0,   # Atrial escape beat
    'j': 0,   # Nodal (junctional) escape beat
    
    'A': 1,   # Atrial premature beat
    'a': 1,   # Aberrated atrial premature beat
    'J': 1,   # Nodal (junctional) premature beat
    'S': 1,   # Supraventricular premature or ectopic beat
    'V': 1,   # Premature ventricular contraction
    'F': 1,   # Fusion of ventricular and normal beat
    '!': 1,   # Ventricular flutter wave
    '/': 1,   # Paced beat
    'f': 1,   # Fusion of paced and normal beat
    '?': 1,   # Beat cannot be classified
}

CLASS_NAMES = ['Normal', 'Abnormal']

# ======================== 训练配置 ========================
TRAIN_CONFIG = {
    'batch_size': 64,
    'epochs': 200,
    'learning_rate': 0.0005,
    'validation_split': 0.2,
    'test_split': 0.2,
    'random_seed': 42,
    
    # ------ Data Augmentation (Phase 2A-3: 增强, ~2x stronger) ------
    'augmentation': {
        'enabled': True,
        'apply_prob': 0.80,                         # ↑ 0.50→0.80, more augmentation
        'noise_std': 0.015,                         # ↑ 0.01→0.015
        'time_scale_range': [0.88, 1.12],           # ↑ ±8%→±12%
        'amplitude_scale_range': [0.80, 1.20],      # ↑ ±15%→±20%
        'baseline_drift_amplitude': 0.20,           # ↑ 0.15→0.20
    },
    
    # ------ Loss (Phase 1: FocalLoss + LabelSmoothing) ------
    'focal_loss': {
        'enabled': True,
        'gamma': 1.0,                       # Reduced from 2.0 (避免梯度消失)
        'alpha': 0.75,                      # 最佳: α=0.75 (Acc 93.98%, A.Recall 0.72)
        'label_smoothing': 0.0,             # 关闭，优先验证纯 FocalLoss
    },
    
    # ------ Mixup ------
    'mixup': {
        'enabled': True,
        'alpha': 0.2,                   # Beta distribution shape
        'prob': 0.5,                    # Application probability
    },
    
    # ------ Class Weights ------
    'class_weight': {
        'enabled': True,
        'abnormal_weight': 2.5,         # Abnormal class weighted 2.5x vs Normal
    },
    
    # ------ AdamW ------
    'optimizer': {
        'type': 'adamw',
        'weight_decay': 1e-4,
    },
    
    # ------ Learning Rate Schedule ------
    'lr_schedule': {
        'type': 'cosine_restart',       # cosine_restart / reduce_on_plateau
        'warmup_epochs': 5,
        'warmup_start_lr': 1e-6,
        'initial_lr': 1e-3,
        'min_lr': 1e-6,
        'restart_period': 20,           # Cosine restart every 20 epochs
    },
    
    # ------ Early Stopping ------
    'early_stopping_patience': 20,
    'reduce_lr_patience': 8,
    'reduce_lr_factor': 0.5,
    
    # ------ Cross-Validation ------
    'cross_validation': {
        'enabled': False,               # Enable for 5-fold CV
        'n_folds': 5,
    },
}

# ======================== 多任务学习配置 (Route F) ========================
MULTITASK_CONFIG = {
    'enabled': True,
    'tasks': {
        'classification': {'enabled': True, 'loss_weight': 1.0},
        'bpm_regression': {'enabled': True, 'loss_weight': 0.3},
        'sqi_regression': {'enabled': True, 'loss_weight': 0.2},
    },
    'focal_loss': {
        'gamma': 1.0,
        'alpha': 0.75,
    },
    'bpm_range': [30, 180],
    'sqi_range': [0.0, 1.0],
}

# ======================== TTA 推理配置 (Route G) ========================
TTA_CONFIG = {
    'enabled': True,
    'sliding_window': {
        'enabled': False,           # OFF for isolated beats; ON for streaming ECG
        'stride_samples': 62,       # 0.25s at 250Hz
        'n_views': 3,
        'aggregation': 'mean',
    },
    'augmentation': {
        'enabled': True,            # Primary TTA for beat-level inference
        'n_aug': 5,                 # 5 augmented + 1 original = 6 views
        'noise_std': 0.01,
        'amp_range': 0.05,
        'aggregation': 'mean',
    },
    'multi_beat_confirm': {
        'n_confirm': 3,
    },
    'threshold': 0.35,
}

# ======================== TFLite 导出配置 ========================
TFLITE_CONFIG = {
    'representative_dataset_size': 1000,  # 量化校准集大小
    'quantization': 'int8',               # int8 / float16
    'output_filename': 'ecg_model.tflite',
    'c_array_header': 'ecg_model_data.h',
}

# ======================== 推理配置 ========================
INFERENCE_CONFIG = {
    'window_size': BEAT_WINDOW_SAMPLES,  # 250
    'stride': BEAT_WINDOW_SAMPLES // 2,  # 125 (50% 重叠滑动窗口)
    'threshold': 0.35,                   # 异常判定阈值 (P0优化: 0.50→0.35, 牺牲少量精确率换取召回率)
    'multi_beat_confirm': 2,             # 多拍确认: 连续N拍异常才报警 (减少假阳性)
}