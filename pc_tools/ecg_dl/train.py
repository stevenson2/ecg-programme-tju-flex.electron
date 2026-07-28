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
    build_ecg_cnn_1d, build_ecg_cnn_1d_v2, build_ecg_cnn_1d_v3, build_ecg_cnn_1d_tiny,
    compile_model as compile_cnn, get_callbacks as get_cnn_callbacks,
    model_summary_table
)
from models.resnet_lite_1d import (
    build_ecg_resnet_lite, build_ecg_resnet_lite_small,
    build_ecg_resnet_lite_medium, build_ecg_resnet_lite_large,
    compile_model as compile_resnet, get_callbacks as get_resnet_callbacks,
    model_summary_table as resnet_summary
)
from models.utils import (
    plot_training_history, plot_confusion_matrix,
    plot_sample_beats, save_model_summary
)
from config import MODELS_DIR, CLASS_NAMES


def train(
    use_tiny: bool = False,
    use_v2: bool = True,
    use_v3: bool = False,
    use_resnet: bool = False,
    use_ptbxl: bool = False,
    use_merged: bool = False,
    use_incart: bool = False,
    use_ecg1000: bool = False,
    use_no_focal: bool = False,
    epochs: int = None,
    batch_size: int = None,
    skip_evaluate: bool = False
) -> tf.keras.Model:
    """
    完整训练流程 (支持多模型 + 多数据集).

    Args:
        use_tiny:    CNN tiny (<5K).
        use_v2:      CNN v2 (15K, 默认).
        use_resnet:  ECG-ResNet-Lite medium (55K, Phase 1).
        use_ptbxl:   仅用 PTB-XL 数据.
        use_merged:  MIT-BIH + PTB-XL 合并.
        use_incart:  MIT-BIH + INCART 合并 (P0: 当前优先).
        use_no_focal: 禁用 FocalLoss, 使用标准交叉熵.
        epochs:      训练轮数.
        batch_size:  批大小.
        skip_evaluate: 跳过评估.
    """
    print(f"\n{'='*60}")
    if use_incart:
        ds_name = "MIT-BIH+INCART"
    elif use_ecg1000:
        ds_name = "MIT-BIH+ECG1000"
    elif use_merged:
        ds_name = "Merged"
    else:
        ds_name = "PTB-XL" if use_ptbxl else "MIT-BIH"
    model_type = "ECG-ResNet-Lite" if use_resnet else \
                 ("CNN-v3" if use_v3 else ("CNN-v2" if use_v2 else ("CNN-tiny" if use_tiny else "CNN-v1")))
    loss_type = "CE" if use_no_focal else "FocalLoss"
    print(f" ECG [{ds_name}] [{model_type}] [{loss_type}]")
    print(f"{'='*60}\n")
    
    # Step 1: 数据准备
    print("[1/5] 准备数据集...")
    datasets = prepare_datasets(
        batch_size=batch_size or TRAIN_CONFIG['batch_size'],
        use_ptbxl=use_ptbxl,
        use_merged=use_merged,
        use_incart=use_incart,
        use_ecg1000=use_ecg1000,
    )
    
    # Step 2: 模型构建
    print("\n[2/5] 构建模型...")
    if use_resnet:
        model = build_ecg_resnet_lite_small(
            input_shape=datasets['input_shape']
        )
        model = compile_resnet(model, learning_rate=TRAIN_CONFIG['learning_rate'])
        resnet_summary(model)
        callbacks = get_resnet_callbacks()
        save_model_summary(model)
    elif use_tiny:
        model = build_ecg_cnn_1d_tiny(
            input_shape=datasets['input_shape'],
            n_classes=len(CLASS_NAMES)
        )
    elif use_v3:
        model = build_ecg_cnn_1d_v3(
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
    
    if not use_resnet:
        model = compile_cnn(
            model,
            learning_rate=TRAIN_CONFIG['learning_rate'],
            loss='categorical_crossentropy' if use_no_focal else None
        )
        model_summary_table(model)
        save_model_summary(model)
    
    # Step 3: 训练
    print("\n[3/5] 开始训练...")
    if not use_resnet:
        callbacks = get_cnn_callbacks()
    
    # NOTE: class_weight is incompatible with tf.data.Dataset in Keras 3.x.
    # FocalLoss handles class imbalance internally via alpha parameter.
    # See ModelPlan §11.2 for details.
    
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
    model = compile_cnn(model)
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
    parser.add_argument("--v3", action="store_true", help="CNN v3 (30K, scaled-up)")
    parser.add_argument("--resnet", action="store_true", help="ECG-ResNet-Lite (55K)")
    parser.add_argument("--ptbxl", action="store_true", help="仅用 PTB-XL 数据集")
    parser.add_argument("--merged", action="store_true", help="MIT-BIH + PTB-XL 合并")
    parser.add_argument("--incart", action="store_true", help="MIT-BIH + INCART 合并 (P0)")
    parser.add_argument("--ecg1000", action="store_true", help="MIT-BIH + ECG1000 合并 (本地)")
    parser.add_argument("--no-focal", action="store_true", help="禁用 FocalLoss, 用标准交叉熵")
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
            use_v3=args.v3,
            use_resnet=args.resnet,
            use_tiny=args.tiny,
            use_v2=not args.v1,
            use_ptbxl=args.ptbxl,
            use_merged=args.merged,
            use_incart=args.incart,
            use_ecg1000=args.ecg1000,
            use_no_focal=args.no_focal,
            epochs=args.epochs,
            batch_size=args.batch_size,
            skip_evaluate=args.skip_eval
        )