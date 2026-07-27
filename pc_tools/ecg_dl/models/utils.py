#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型工具函数
- 模型加载/保存
- 学习率调度器可视化
- 混淆矩阵绘制
- 心拍可视化
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, CLASS_NAMES, INFERENCE_CONFIG


def plot_training_history(history, save_path: str = None):
    """
    绘制训练历史曲线
    
    Args:
        history: model.fit() 返回的 History 对象
        save_path: 保存路径, 默认保存到 models/training_history.png
    """
    if save_path is None:
        save_path = str(MODELS_DIR / 'training_history.png')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[工具] 训练历史已保存到: {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None
):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签 (n,)
        y_pred: 预测类别 (n,)
        save_path: 保存路径
    """
    if save_path is None:
        save_path = str(MODELS_DIR / 'confusion_matrix.png')
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel='Predicted Label',
        ylabel='True Label',
        title='Confusion Matrix'
    )
    
    # 显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 打印分类报告
    print(f"\n[工具] 分类报告:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print(f"[工具] 混淆矩阵已保存到: {save_path}")


def plot_sample_beats(
    beats: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray = None,
    n_samples: int = 6,
    save_path: str = None
):
    """
    绘制 Sample ECG 心拍可视化
    
    Args:
        beats: 心拍数据 (n, 250)
        labels: 真实标签 (n,)
        predictions: 预测标签 (n,), 可选
        n_samples: 显示样本数
        save_path: 保存路径
    """
    if save_path is None:
        save_path = str(MODELS_DIR / 'sample_beats.png')
    
    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i in range(n_samples):
        idx = np.random.randint(0, len(beats))
        beat = beats[idx]
        true_label = CLASS_NAMES[labels[idx]]
        
        ax = axes[i]
        t = np.arange(len(beat)) / 250.0  # 时间轴 (秒)
        ax.plot(t, beat, linewidth=1.5, color='steelblue')
        
        title = f'True: {true_label}'
        if predictions is not None:
            pred_label = CLASS_NAMES[predictions[idx]]
            color = 'green' if labels[idx] == predictions[idx] else 'red'
            title += f' | Pred: {pred_label}'
            ax.set_title(title, color=color, fontweight='bold')
        else:
            ax.set_title(title)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余子图
    for j in range(n_samples, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[工具] 样本心拍图已保存到: {save_path}")


def save_model_summary(model: tf.keras.Model, save_path: str = None):
    """
    保存模型结构的文本摘要
    
    Args:
        model: Keras 模型
        save_path: 保存路径
    """
    if save_path is None:
        save_path = str(MODELS_DIR / 'model_summary.txt')
    
    with open(save_path, 'w') as f:
        # 重定向 stdout
        original_stdout = sys.stdout
        sys.stdout = f
        
        model.summary()
        
        # 计算预估大小
        total_params = model.count_params()
        print(f"\n{'='*50}")
        print(f"总参数量: {total_params:,}")
        print(f"FP32 模型大小: {total_params * 4 / 1024:.2f} KB")
        print(f"INT8 模型大小: {total_params * 1 / 1024:.2f} KB")
        
        sys.stdout = original_stdout
    
    print(f"[工具] 模型摘要已保存到: {save_path}")


if __name__ == "__main__":
    print("[工具] utils.py 测试...")
    # 生成随机测试数据
    np.random.seed(42)
    dummy_beats = np.random.randn(100, 250)
    dummy_labels = np.random.randint(0, 2, 100)
    dummy_preds = np.random.randint(0, 2, 100)
    
    plot_confusion_matrix(dummy_labels, dummy_preds, str(MODELS_DIR / 'test_cm.png'))
    plot_sample_beats(dummy_beats, dummy_labels, dummy_preds, n_samples=3)
    
    print("[工具] ✅ 测试通过")