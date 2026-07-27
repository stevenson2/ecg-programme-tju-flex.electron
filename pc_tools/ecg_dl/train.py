#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练入口
一站式完成: 数据加载 -> 模型构建 -> 训练 -> 评估 -> 导出
"""

import sys
import os
from pathlib import Path
import numpy as np
import tensorflow as tf

# 设置随机种子
from config import TRAIN_CONFIG
tf.random.set_seed(TRAIN_CONFIG['random_seed'])
np.random.seed(TRAIN_CONFIG['random_seed'])

# 导入自定义模块
from data.dataset import prepare_datasets
from models.cnn_1d import (
    build_ecg_cnn_1d, build_ecg_cnn_1d_v2, build_ecg_cnn_1d_tiny,
    compile_model, get_callbacks, model_summary_table
)
from models.utils import (
    plot_training_history, plot_confusion_matrix,
    plot_sample_beats, save_model_summary
)
from config import MODELS_DIR, CLASS_NAMES


def train(
    use_tiny: bool = False,
    use_v2: bool = True,
    epochs: int = None,
    batch_size: int = None,
    skip_evaluate: bool = False
) -> tf.keras.Model:
    """
    完整训练流程
    
    Args:
        use_tiny: 使用更轻量的 tiny 版本
        epochs: 训练轮数
        batch_size: 批大小
        skip_evaluate: 跳过评估
        
    Returns:
        训练好的模型
    """
    print(f"\n{'='*60}")
    print(" ECG 异常检测模型训练")
    print(f"{'='*60}\n")
    
    # Step 1: 数据准备
    print("[1/5] 准备数据集...")
    datasets = prepare_datasets(
        augment=True,
        batch_size=batch_size or TRAIN_CONFIG['batch_size']
    )
    
    # Step 2: 模型构建
    print("\n[2/5] 构建模型...")
    if use_tiny:
        model = build_ecg_cnn_1d_tiny(
            input_shape=datasets['input_shape'],
            n_classes=len(CLASS_NAMES)
        )
    elif use_v2:
        model = build_ecg_cnn_1d_v2(
            input_shape=datasets['input_shape'],
            n_classes=len(CLASS_NAMES)
        )
    else:
        model = build_ecg_cnn_1d(
            input_shape=datasets['input_shape'],
            n_classes=len(CLASS_NAMES)
        )
    
    model = compile_model(
        model,
        learning_rate=TRAIN_CONFIG['learning_rate']
    )
    model_summary_table(model)
    save_model_summary(model)
    
    # Step 3: 训练
    print("\n[3/5] 开始训练...")
    callbacks = get_callbacks()
    
    history = model.fit(
        datasets['train_ds'],
        validation_data=datasets['val_ds'],
        epochs=epochs or TRAIN_CONFIG['epochs'],
        callbacks=callbacks,
        verbose=2
    )
    
    # 训练曲线
    plot_training_history(history)
    
    # 保存最终模型
    model.save(str(MODELS_DIR / 'final_model.h5'))
    print(f"[训练] 模型已保存到: {MODELS_DIR}")
    
    # Step 4: 评估
    if not skip_evaluate:
        print("\n[4/5] 模型评估...")
        
        # 获取测试数据
        x_test, y_test = datasets['data']['test']
        
        # 预测
        x_test_input = x_test[..., np.newaxis]
        y_pred_probs = model.predict(x_test_input, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # 评估指标
        loss, acc, prec, recall, auc = model.evaluate(
            x_test_input,
            tf.keras.utils.to_categorical(y_test, num_classes=2),
            verbose=0
        )
        
        print(f"\n{'='*40}")
        print("  测试集评估结果")
        print(f"{'='*40}")
        print(f"  Loss:    {loss:.4f}")
        print(f"  Acc:     {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Prec:    {prec:.4f}")
        print(f"  Recall:  {recall:.4f}")
        print(f"  AUC:     {auc:.4f}")
        print(f"{'='*40}")
        
        # 混淆矩阵
        plot_confusion_matrix(y_test, y_pred)
        plot_sample_beats(x_test, y_test, y_pred, n_samples=6)
    
    # Step 5: 汇总
    print("\n[5/5] [OK] 训练完成!")
    print(f"  模型文件: {MODELS_DIR}")
    print(f"  包含文件:")
    for f in MODELS_DIR.glob("*"):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"    - {f.name} ({size_kb:.1f} KB)")
    
    return model


def quick_test():
    """
    快速测试: 使用少量数据进行验证
    仅用 3 条记录, 训练 5 个 epoch
    """
    print(f"\n{'='*60}")
    print(" [TEST] Quick Test Mode")
    print(f"{'='*60}\n")
    
    # 直接生成随机测试数据
    n_samples = 500
    input_shape = (250, 1)
    
    print(f"[测试] 生成 {n_samples} 个随机样本")
    x_train = np.random.randn(int(n_samples*0.6), *input_shape).astype(np.float32)
    y_train = tf.keras.utils.to_categorical(
        np.random.randint(0, 2, int(n_samples*0.6)), 2
    )
    x_val = np.random.randn(int(n_samples*0.2), *input_shape).astype(np.float32)
    y_val = tf.keras.utils.to_categorical(
        np.random.randint(0, 2, int(n_samples*0.2)), 2
    )
    x_test = np.random.randn(int(n_samples*0.2), *input_shape).astype(np.float32)
    y_test = tf.keras.utils.to_categorical(
        np.random.randint(0, 2, int(n_samples*0.2)), 2
    )
    
    # 构建模型
    model = build_ecg_cnn_1d(input_shape=input_shape)
    model = compile_model(model)
    model_summary_table(model)
    
    # 训练
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=5,
        batch_size=32,
        verbose=2
    )
    
    # 评估
    eval_results = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[测试] 评估结果:")
    print(f"  Loss: {eval_results[0]:.4f}")
    print(f"  Acc:  {eval_results[1]:.4f}")
    
    print(f"\n[测试] [OK] 快速测试通过!")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ECG 异常检测模型训练")
    parser.add_argument("--tiny", action="store_true", help="使用 tiny 模型")
    parser.add_argument("--v1", action="store_true", help="使用 v1 原版模型")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="批大小")
    parser.add_argument("--quick-test", action="store_true", help="快速测试")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评估")
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test()
    else:
        train(
            use_tiny=args.tiny,
            use_v2=not args.v1,
            epochs=args.epochs,
            batch_size=args.batch_size,
            skip_evaluate=args.skip_eval
        )