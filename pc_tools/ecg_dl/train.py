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
from typing import Optional

# 显存按需分配 (避免 TF 贪心占满整池, 真实需求 ~1GB; TUNING_HISTORY 十三章)
_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    try:
        tf.config.experimental.set_memory_growth(_gpus[0], True)
    except Exception:
        pass

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
from models.cnn_m import (
    build_ecg_cnn_m_classifier, build_ecg_cnn_m_small, build_ecg_cnn_m_large,
    compile_model as compile_cnn_m, get_callbacks as get_cnn_m_callbacks,
    model_summary_table as cnn_m_summary
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
    use_resnet_medium: bool = False,
    use_resnet_large: bool = False,
    use_cnn_m: bool = False,
    use_cnn_m_small: bool = False,
    use_cnn_m_large: bool = False,
    use_3beat: bool = False,
    use_ptbxl: bool = False,
    use_merged: bool = False,
    use_incart: bool = False,
    use_ptbxl_rhythm: bool = False,
    use_ecg1000: bool = False,
    use_ptb_beat: bool = False,
    ptb_abn_max: int = 10000,
    domain_balanced: bool = False,
    ptb_batch_frac: float = 0.20,
    ptb_loss_weight: float = 0.5,
    use_no_focal: bool = False,
    use_balanced: bool = False,
    sliding_dup: int = 0,
    sliding_max_shift: int = 40,
    focal_gamma: Optional[float] = None,
    focal_alpha: Optional[float] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    skip_evaluate: bool = False,
    patient_split: bool = False,     # 4.4-4: 患者级划分训练 (消除记录级泄漏, 发表级严谨)
    early_patience: int = 20,        # EarlyStopping patience (val_loss), 部署链试点用 40
    optimizer: str = "adamw",        # adamw (默认) 或 sgd (Nesterov, 泛化更强, Wilson 2017)
    lr: Optional[float] = None       # 覆盖 TRAIN_CONFIG learning_rate (sgd 需 ~1e-2)
) -> tf.keras.Model:
    """
    完整训练流程 (支持多模型 + 多数据集).

    Args:
        use_tiny:    CNN tiny (<5K).
        use_v2:      CNN v2 (15K, 默认).
        use_v3:      CNN v3 (30K, scaled-up).
        use_resnet:  ECG-ResNet-Lite small (25K).
        use_resnet_medium:  ECG-ResNet-Lite medium (55K, P0推荐).
        use_resnet_large:   ECG-ResNet-Lite large (80K).
        use_cnn_m:  ECG-CNN-M (600K, Phase 2B 三拍输入).
        use_3beat: 使用 3-beat 序列数据 (750 点, 仅配合 --cnn-m).
        use_ptbxl:   仅用 PTB-XL 数据.
        use_merged:  MIT-BIH + PTB-XL 合并.
        use_incart:  MIT-BIH + INCART 合并 (P0: 当前优先).
        use_no_focal: 禁用 FocalLoss, 使用标准交叉熵.
        use_balanced: 训练集 50/50 类别均衡采样.
        focal_gamma:  覆盖 config 中的 FocalLoss gamma.
        focal_alpha:  覆盖 config 中的 FocalLoss alpha.
        epochs:      训练轮数.
        batch_size:  批大小.
        skip_evaluate: 跳过评估.
    """
    print(f"\n{'='*60}")
    if use_ptb_beat:
        ds_name = "MIT+INCART+PTB"
    elif use_ptbxl_rhythm:
        ds_name = "MIT+INCART+PTBXL"
    elif use_incart:
        ds_name = "MIT-BIH+INCART"
    elif use_ecg1000:
        ds_name = "MIT-BIH+ECG1000"
    elif use_merged:
        ds_name = "Merged"
    else:
        ds_name = "PTB-XL" if use_ptbxl else "MIT-BIH"
    
    if use_cnn_m_small:
        model_type = "ECG-CNN-M-Small (114K, 3-beat)"
    elif use_cnn_m_large:
        model_type = "ECG-CNN-M-Large (453K, 3-beat)"
    elif use_cnn_m:
        model_type = "ECG-CNN-M (140K, 3-beat)"
    elif use_resnet_large:
        model_type = "ECG-ResNet-Lite-Large (80K)"
    elif use_resnet_medium:
        model_type = "ECG-ResNet-Lite-Medium (55K)"
    elif use_resnet:
        model_type = "ECG-ResNet-Lite-Small (25K)"
    else:
        model_type = "CNN-v3" if use_v3 else ("CNN-v2" if use_v2 else ("CNN-tiny" if use_tiny else "CNN-v1"))
    
    loss_type = "CE" if use_no_focal else "FocalLoss"
    bal_tag = "+Bal" if use_balanced else ""
    sliding_tag = (f"+Sliding(dup={sliding_dup},shift={sliding_max_shift})"
                   if sliding_dup > 0 else "")
    print(f" ECG [{ds_name}] [{model_type}] [{loss_type}]{bal_tag}{sliding_tag}")
    print(f"{'='*60}\n")

    # Override FocalLoss params from CLI
    if focal_gamma is not None:
        TRAIN_CONFIG['focal_loss']['gamma'] = focal_gamma
        print(f"[参数覆盖] FocalLoss gamma = {focal_gamma}")
    if focal_alpha is not None:
        TRAIN_CONFIG['focal_loss']['alpha'] = focal_alpha
        print(f"[参数覆盖] FocalLoss alpha = {focal_alpha}")

    # Step 1: 数据准备
    print("[1/5] 准备数据集...")
    datasets = prepare_datasets(
        batch_size=batch_size or TRAIN_CONFIG['batch_size'],
        use_ptbxl=use_ptbxl,
        use_merged=use_merged,
        use_incart=use_incart,
        use_ecg1000=use_ecg1000,
        use_ptb_beat=use_ptb_beat,
        ptb_abn_max=ptb_abn_max,
        domain_balanced=domain_balanced,
        ptb_batch_frac=ptb_batch_frac,
        ptb_loss_weight=ptb_loss_weight,
        use_ptbxl_rhythm=use_ptbxl_rhythm,
        use_balanced=use_balanced,
        use_3beat=use_3beat,
        sliding_dup=sliding_dup,
        sliding_max_shift=sliding_max_shift,
        patient_split=patient_split,
    )
    
    # Step 2: 模型构建
    print("\n[2/5] 构建模型...")
    is_cnn_m = False
    callbacks: list = []
    if use_cnn_m_small:
        model = build_ecg_cnn_m_small(
            input_shape=datasets['input_shape'],
            n_classes=len(CLASS_NAMES)
        )
        model = compile_cnn_m(model, learning_rate=TRAIN_CONFIG['learning_rate'])
        cnn_m_summary(model)
        callbacks = get_cnn_m_callbacks(model_name="best_cnn_m_small.h5")
        is_resnet = False
        is_cnn_m = True
    elif use_cnn_m_large:
        model = build_ecg_cnn_m_large(
            input_shape=datasets['input_shape'],
            n_classes=len(CLASS_NAMES)
        )
        model = compile_cnn_m(
            model,
            learning_rate=(lr if lr is not None else TRAIN_CONFIG['learning_rate']),
            optimizer=optimizer)
        cnn_m_summary(model)
        callbacks = get_cnn_m_callbacks(model_name="best_cnn_m_large.h5",
                                        early_patience=early_patience)
        is_resnet = False
        is_cnn_m = True
    elif use_cnn_m:
        model = build_ecg_cnn_m_classifier(
            input_shape=datasets['input_shape'],
            n_classes=len(CLASS_NAMES)
        )
        model = compile_cnn_m(model, learning_rate=TRAIN_CONFIG['learning_rate'])
        cnn_m_summary(model)
        callbacks = get_cnn_m_callbacks()
        is_resnet = False
        is_cnn_m = True
    elif use_resnet_large:
        model = build_ecg_resnet_lite_large(
            input_shape=datasets['input_shape']
        )
        model = compile_resnet(
            model, learning_rate=(lr if lr is not None else TRAIN_CONFIG['learning_rate']),
            loss='categorical_crossentropy' if use_no_focal else None,
            optimizer=optimizer)
        resnet_summary(model)
        callbacks = get_resnet_callbacks(model_name="best_resnet_large.h5",
                                         early_patience=early_patience)
        save_model_summary(model)
    elif use_resnet_medium:
        model = build_ecg_resnet_lite_medium(
            input_shape=datasets['input_shape']
        )
        model = compile_resnet(
            model, learning_rate=(lr if lr is not None else TRAIN_CONFIG['learning_rate']),
            loss='categorical_crossentropy' if use_no_focal else None,
            optimizer=optimizer)
        resnet_summary(model)
        callbacks = get_resnet_callbacks(model_name="best_resnet_medium.h5",
                                         early_patience=early_patience)
        save_model_summary(model)
    elif use_resnet:
        model = build_ecg_resnet_lite_small(
            input_shape=datasets['input_shape']
        )
        model = compile_resnet(
            model, learning_rate=(lr if lr is not None else TRAIN_CONFIG['learning_rate']),
            loss='categorical_crossentropy' if use_no_focal else None,
            optimizer=optimizer)
        resnet_summary(model)
        callbacks = get_resnet_callbacks(early_patience=early_patience)
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
    
    is_resnet = use_resnet or use_resnet_medium or use_resnet_large
    is_cnn_m = use_cnn_m

    if not is_resnet and not is_cnn_m:
        model = compile_cnn(
            model,
            learning_rate=TRAIN_CONFIG['learning_rate'],
            loss='categorical_crossentropy' if use_no_focal else None
        )
        model_summary_table(model)
        save_model_summary(model)

    is_resnet = use_resnet or use_resnet_medium or use_resnet_large
    # Step 3: 训练
    print("\n[3/5] 开始训练...")
    if not is_resnet and not is_cnn_m:
        callbacks = get_cnn_callbacks()
    
    # NOTE: class_weight is incompatible with tf.data.Dataset in Keras 3.x.
    # FocalLoss handles class imbalance internally via alpha parameter.
    # See ModelPlan §11.2 for details.

    # Loss 可视化支持: 追加 CSVLogger (配合 plot_history.py --watch)
    from tensorflow.keras.callbacks import CSVLogger
    history_csv_path = str(MODELS_DIR / "train_history.csv")
    callbacks = callbacks + [CSVLogger(history_csv_path, append=False)]
    print(f"[训练] Loss 历史: {history_csv_path}")
    print(f"[训练] 实时可视化: python3 plot_history.py --csv {history_csv_path} --watch 30 --show")

    history = model.fit(
        datasets['train_ds'],
        validation_data=datasets['val_ds'],
        epochs=epochs or TRAIN_CONFIG['epochs'],
        callbacks=callbacks,
        verbose=2
    )
    
    # 训练曲线
    plot_training_history(history)
    
    # 保存最终模型 (模型特异性命名，不再覆盖)
    final_name = {
        "tiny": "final_cnn_tiny.h5", "v2": "final_cnn_v2.h5",
        "v3": "final_cnn_v3.h5", "v1": "final_cnn_v1.h5",
        "resnet_s": "final_resnet_s.h5", "resnet_m": "final_resnet_m.h5",
        "resnet_l": "final_resnet_l.h5",
        "cnn_m": "final_cnn_m.h5",
        "cnn_m_s": "final_cnn_m_small.h5",
        "cnn_m_l": "final_cnn_m_large.h5",
    }
    if use_cnn_m_small:      _key = "cnn_m_s"
    elif use_cnn_m_large:    _key = "cnn_m_l"
    elif use_cnn_m:          _key = "cnn_m"
    elif use_resnet_large:   _key = "resnet_l"
    elif use_resnet_medium:  _key = "resnet_m"
    elif use_resnet:         _key = "resnet_s"
    elif use_v3:             _key = "v3"
    elif use_tiny:           _key = "tiny"
    elif use_v2:             _key = "v2"
    else:                    _key = "v1"
    model.save(str(MODELS_DIR / final_name[_key]))
    print(f"[训练] 模型已保存到: {MODELS_DIR / final_name[_key]}")
    
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
    parser.add_argument("--resnet", action="store_true", help="ECG-ResNet-Lite Small (25K)")
    parser.add_argument("--resnet-medium", action="store_true", help="ECG-ResNet-Lite Medium (55K, P0推荐)")
    parser.add_argument("--resnet-large", action="store_true", help="ECG-ResNet-Lite Large (80K)")
    parser.add_argument("--cnn-m", action="store_true", help="ECG-CNN-M (140K, Phase 2B)")
    parser.add_argument("--cnn-m-small", action="store_true", help="ECG-CNN-M-Small (114K)")
    parser.add_argument("--cnn-m-large", action="store_true", help="ECG-CNN-M-Large (453K)")
    parser.add_argument("--3beat", dest="use_3beat", action="store_true",
                        help="使用 3-beat 序列数据 (配合 --cnn-m)")
    parser.add_argument("--ptbxl", action="store_true", help="仅用 PTB-XL 数据集")
    parser.add_argument("--merged", action="store_true", help="MIT-BIH + PTB-XL 合并")
    parser.add_argument("--incart", action="store_true", help="MIT-BIH + INCART 合并 (P0)")
    parser.add_argument("--ptbxl-r", action="store_true", help="MIT-BIH+INCART+PTBXL节律合并")
    parser.add_argument("--ptb-beat", action="store_true",
                        help="MIT+INCART+PTB原始库(beat级)合并 (Phase 3B)")
    parser.add_argument("--ptb-abn-max", type=int, default=10000,
                        help="PTB 异常拍限量 (默认10000, 防MI形态主导)")
    parser.add_argument("--domain-balanced", action="store_true",
                        help="域平衡采样: 每batch固定比例PTB拍 (配合--ptb-beat)")
    parser.add_argument("--ptb-frac", type=float, default=0.20,
                        help="每batch中PTB拍占比 (默认0.20)")
    parser.add_argument("--ptb-weight", type=float, default=0.5,
                        help="PTB拍loss权重 (默认0.5, 记录级标签降权)")
    parser.add_argument("--ecg1000", action="store_true", help="MIT-BIH + ECG1000 合并 (本地)")
    parser.add_argument("--no-focal", action="store_true", help="禁用 FocalLoss, 用标准交叉熵")
    parser.add_argument("--balanced", action="store_true", help="训练集 50/50 类别均衡采样 (Phase 2A-4)")
    parser.add_argument("--sliding-dup", type=int, default=0,
                        help="异常类滑窗采样增强: 每个异常心拍生成的移位视图数 (0=关闭, 张异凡2019方法)")
    parser.add_argument("--sliding-shift", type=int, default=40,
                        help="滑窗最大平移量 (采样点, 默认40=160ms @250Hz)")
    parser.add_argument("--focal-gamma", type=float, default=None, help="覆盖 FocalLoss gamma")
    parser.add_argument("--focal-alpha", type=float, default=None, help="覆盖 FocalLoss alpha")
    parser.add_argument("--tiny", action="store_true", help="使用 tiny 模型")
    parser.add_argument("--v1", action="store_true", help="使用 v1 原版模型")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="批大小")
    parser.add_argument("--quick-test", action="store_true", help="快速测试")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评估")
    parser.add_argument("--patient-split", action="store_true",
                        help="4.4-4 患者级划分训练 (消除记录级泄漏, 发表级严谨)")
    parser.add_argument("--deploy-chain", action="store_true",
                        help="阶段1.5: 使用部署链重建数据 (*_deploy.npz) 训练 (TUNING_HISTORY 十三章)")
    parser.add_argument("--patience", type=int, default=20,
                        help="EarlyStopping patience (val_loss), 部署链延长跑用 40")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sgd"],
                        help="adamw (默认) 或 sgd (Nesterov, 泛化更强, Wilson 2017)")
    parser.add_argument("--lr", type=float, default=None,
                        help="覆盖学习率 (sgd 建议 1e-2 量级)")
    parser.add_argument("--phase-shift", type=int, default=0,
                        help="T2-5: 全类相位扰动最大样本数 (batch 统一 roll, 两类一起; "
                             "0=关闭; 10=±10 样本 @250Hz)")

    args = parser.parse_args()
    
    if args.deploy_chain:
        import data.dataset as _ds
        _ds.set_npz_suffix("_deploy")

    if args.phase_shift and args.phase_shift > 0:
        from config import TRAIN_CONFIG
        TRAIN_CONFIG['augmentation']['phase_shift'] = args.phase_shift
        print(f"[T2-5] 全类相位扰动增强: ±{args.phase_shift} 样本 (batch 统一, 两类一起)")
    
    if args.quick_test:
        quick_test()
    else:
        train(
            use_v3=args.v3,
            use_resnet=args.resnet,
            use_resnet_medium=args.resnet_medium,
            use_resnet_large=args.resnet_large,
            use_cnn_m=args.cnn_m,
            use_cnn_m_small=args.cnn_m_small,
            use_cnn_m_large=args.cnn_m_large,
            use_3beat=args.use_3beat,
            use_tiny=args.tiny,
            use_v2=not args.v1,
            use_ptbxl=args.ptbxl,
            use_merged=args.merged,
            use_incart=args.incart,
            use_ecg1000=args.ecg1000,
            use_ptb_beat=args.ptb_beat,
            ptb_abn_max=args.ptb_abn_max,
            domain_balanced=args.domain_balanced,
            ptb_batch_frac=args.ptb_frac,
            ptb_loss_weight=args.ptb_weight,
            use_ptbxl_rhythm=args.ptbxl_r,
            use_no_focal=args.no_focal,
            use_balanced=args.balanced,
            sliding_dup=args.sliding_dup,
            sliding_max_shift=args.sliding_shift,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
            epochs=args.epochs,
            batch_size=args.batch_size,
            skip_evaluate=args.skip_eval,
            patient_split=args.patient_split,
            early_patience=args.patience,
            optimizer=args.optimizer,
            lr=args.lr
        )