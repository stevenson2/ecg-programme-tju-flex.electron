#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估
- 加载训练好的模型
- 在测试集上评估
- 混淆矩阵、ROC曲线、分类报告
- 计算 TFLite 量化后精度
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, CLASS_NAMES
from data.dataset import load_processed_data, train_val_test_split, add_channel_dim
from models.utils import plot_confusion_matrix, plot_sample_beats


def evaluate_h5_model(model_path: str = None):
    """
    评估 H5 模型
    
    Args:
        model_path: 模型文件路径, 默认使用 best_model.h5
    """
    if model_path is None:
        model_path = str(MODELS_DIR / 'best_model.h5')
    
    print(f"[评估] 加载模型: {model_path}")
    model = tf.keras.models.load_model(model_path)
    model.summary()
    
    # 加载测试数据
    data = load_processed_data()
    splits = train_val_test_split(data["beats"], data["labels"], record_ids=data.get("record_ids"))
    x_test, y_test = splits['test']
    
    # 添加通道维度
    x_test_input = add_channel_dim(x_test)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=2)
    
    # 评估
    print("\n[评估] 测试集评估...")
    results = model.evaluate(x_test_input, y_test_onehot, verbose=1)
    
    metric_names = ['Loss', 'Accuracy', 'Precision', 'Recall', 'AUC']
    print(f"\n{'='*40}")
    print("  H5 模型评估结果")
    print(f"{'='*40}")
    for name, val in zip(metric_names, results):
        print(f"  {name:<12}: {val:.4f}")
    print(f"{'='*40}")
    
    # 预测
    y_pred_probs = model.predict(x_test_input, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 混淆矩阵
    plot_confusion_matrix(y_test, y_pred)
    plot_sample_beats(x_test, y_test, y_pred, n_samples=6)
    
    return model, results


def evaluate_tflite_model(tflite_path: str = None):
    """
    评估 TFLite 模型（量化后精度）
    
    Args:
        tflite_path: TFLite 模型路径
    """
    if tflite_path is None:
        tflite_path = str(MODELS_DIR / 'ecg_model.tflite')
    
    print(f"[评估] 加载 TFLite 模型: {tflite_path}")
    
    # 加载 TFLite 模型
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    output_dtype = output_details[0]['dtype']
    
    print(f"[评估] 输入形状: {input_shape}, 类型: {input_dtype}")
    print(f"[评估] 输出形状: {output_details[0]['shape']}, 类型: {output_dtype}")
    
    # 加载测试数据
    data = load_processed_data()
    splits = train_val_test_split(data["beats"], data["labels"], record_ids=data.get("record_ids"))
    x_test, y_test = splits['test']
    
    # 添加通道维度
    x_test_input = add_channel_dim(x_test)
    
    # 量化参数
    input_scale = float(np.array(input_details[0]['quantization_parameters']['scales']).flatten()[0]) if input_dtype in (np.int8, np.uint8) else 1.0
    input_zp = int(np.array(input_details[0]['quantization_parameters']['zero_points']).flatten()[0]) if input_dtype in (np.int8, np.uint8) else 0
    output_scale = float(np.array(output_details[0]['quantization_parameters']['scales']).flatten()[0]) if output_dtype in (np.int8, np.uint8) else 1.0
    output_zp = int(np.array(output_details[0]['quantization_parameters']['zero_points']).flatten()[0]) if output_dtype in (np.int8, np.uint8) else 0

    # TFLite 推理 (正确量化)
    y_pred = []
    
    for i in range(len(x_test_input)):
        if input_dtype in (np.int8, np.uint8):
            sample = (x_test_input[i:i+1] / input_scale + input_zp).astype(input_dtype)
        else:
            sample = x_test_input[i:i+1].astype(input_dtype)
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        if output_dtype in (np.int8, np.uint8):
            output_f = (output[0].astype(np.float32) - output_zp) * output_scale
        else:
            output_f = output[0]
        y_pred.append(np.argmax(output_f))
    
    y_pred = np.array(y_pred)
    
    # 计算精度
    accuracy = np.mean(y_pred == y_test)
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n{'='*40}")
    print("  TFLite 模型评估结果 (INT8 量化)")
    print(f"{'='*40}")
    print(f"  Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"{'='*40}")
    
    # 混淆矩阵
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=str(MODELS_DIR / 'tflite_confusion_matrix.png')
    )
    
    return accuracy, y_pred


def compare_models(h5_path: str = None, tflite_path: str = None):
    """
    比较 H5 模型和 TFLite 模型的精度差异
    
    Args:
        h5_path: H5 模型路径
        tflite_path: TFLite 模型路径
    """
    print(f"\n{'='*60}")
    print(" H5 vs TFLite 精度对比")
    print(f"{'='*60}\n")
    
    if h5_path is None:
        h5_path = str(MODELS_DIR / 'best_model.h5')
    if tflite_path is None:
        tflite_path = str(MODELS_DIR / 'ecg_model.tflite')
    
    # 评估 H5
    model = tf.keras.models.load_model(h5_path)
    
    data = load_processed_data()
    splits = train_val_test_split(data["beats"], data["labels"], record_ids=data.get("record_ids"))
    x_test, y_test = splits['test']
    x_test_input = add_channel_dim(x_test)
    
    # H5 预测
    y_pred_h5 = np.argmax(model.predict(x_test_input, verbose=0), axis=1)
    acc_h5 = np.mean(y_pred_h5 == y_test)
    
    # TFLite 预测
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_scale = float(np.array(input_details[0]['quantization_parameters']['scales']).flatten()[0]) if input_details[0]['dtype'] in (np.int8, np.uint8) else 1.0
    input_zp = int(np.array(input_details[0]['quantization_parameters']['zero_points']).flatten()[0]) if input_details[0]['dtype'] in (np.int8, np.uint8) else 0
    output_scale = float(np.array(output_details[0]['quantization_parameters']['scales']).flatten()[0]) if output_details[0]['dtype'] in (np.int8, np.uint8) else 1.0
    output_zp = int(np.array(output_details[0]['quantization_parameters']['zero_points']).flatten()[0]) if output_details[0]['dtype'] in (np.int8, np.uint8) else 0

    y_pred_tflite = []
    for i in range(len(x_test_input)):
        if input_details[0]['dtype'] in (np.int8, np.uint8):
            sample = (x_test_input[i:i+1] / input_scale + input_zp).astype(input_details[0]['dtype'])
        else:
            sample = x_test_input[i:i+1].astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        if output_details[0]['dtype'] in (np.int8, np.uint8):
            output_f = (output[0].astype(np.float32) - output_zp) * output_scale
        else:
            output_f = output[0]
        y_pred_tflite.append(np.argmax(output_f))
    
    y_pred_tflite = np.array(y_pred_tflite)
    acc_tflite = np.mean(y_pred_tflite == y_test)
    
    # 对比
    print(f"{'Model':<20} {'Accuracy':<12} {'Size':<12}")
    print(f"{'-'*44}")
    
    h5_size = Path(h5_path).stat().st_size / 1024
    tflite_size = Path(tflite_path).stat().st_size / 1024
    
    print(f"{'H5 (FP32)':<20} {acc_h5:.4f}        {h5_size:.1f} KB")
    print(f"{'TFLite (INT8)':<20} {acc_tflite:.4f}        {tflite_size:.1f} KB")
    
    diff = acc_h5 - acc_tflite
    print(f"\n精度差异: {diff:.4f} ({diff*100:.2f}%)")
    
    if diff < 0.02:
        print("结论: ✅ 量化精度损失可接受 (< 2%)")
    else:
        print(f"结论: ⚠️ 量化精度损失 {diff*100:.1f}%, 建议使用 QAT")
    
    return acc_h5, acc_tflite


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ECG 模型评估")
    parser.add_argument("--h5", type=str, default=None, help="H5 模型路径")
    parser.add_argument("--tflite", type=str, default=None, help="TFLite 模型路径")
    parser.add_argument("--compare", action="store_true", help="对比 H5 和 TFLite")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.h5, args.tflite)
    elif args.tflite:
        evaluate_tflite_model(args.tflite)
    else:
        evaluate_h5_model(args.h5)