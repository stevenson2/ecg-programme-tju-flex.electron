#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型导出
- 加载训练好的 Keras 模型
- INT8 量化 → TFLite
- 导出为 C 头文件 (用于 TFLite Micro / ESP32)
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MODELS_DIR, TFLITE_CONFIG, INFERENCE_CONFIG,
    CLASS_NAMES
)
from data.dataset import load_processed_data, train_val_test_split, add_channel_dim


def representative_dataset(num_samples: int = None) -> tf.data.Dataset:
    """
    生成代表性数据集 (用于 INT8 量化校准)
    
    Args:
        num_samples: 校准样本数
        
    Yields:
        输入样本
    """
    if num_samples is None:
        num_samples = TFLITE_CONFIG['representative_dataset_size']
    
    data = load_processed_data()
    splits = train_val_test_split(data["beats"], data["labels"], record_ids=data.get("record_ids"))
    
    x_train = splits['train'][0]
    x_train_input = add_channel_dim(x_train)
    
    # 随机选择校准样本
    indices = np.random.choice(len(x_train_input), 
                                min(num_samples, len(x_train_input)),
                                replace=False)
    
    for idx in indices:
        sample = x_train_input[idx:idx+1].astype(np.float32)
        yield [sample]


def convert_to_tflite(
    h5_path: str = None,
    output_path: str = None,
    quantization: str = None
) -> str:
    """
    将 Keras H5 模型转换为 TFLite
    
    Args:
        h5_path: H5 模型路径
        output_path: 输出路径
        quantization: 量化类型 ('int8', 'float16', None)
        
    Returns:
        TFLite 文件路径
    """
    if h5_path is None:
        h5_path = str(MODELS_DIR / 'best_model.h5')
    
    if output_path is None:
        output_path = str(MODELS_DIR / TFLITE_CONFIG['output_filename'])
    
    if quantization is None:
        quantization = TFLITE_CONFIG['quantization']
    
    print(f"\n{'='*60}")
    print(" TFLite 模型导出")
    print(f"{'='*60}\n")
    
    # 加载模型
    print(f"[导出] 加载 Keras 模型: {h5_path}")
    model = tf.keras.models.load_model(h5_path)
    
    # 转换为 TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 配置量化
    if quantization == 'int8':
        print(f"[导出] 量化类型: INT8 (全整数量化)")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif quantization == 'float16':
        print(f"[导出] 量化类型: FP16")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    else:
        print(f"[导出] 量化类型: 无 (FP32)")
    
    # 转换
    print("[导出] 转换中...")
    tflite_model = converter.convert()
    
    # 保存
    output_path = Path(output_path)
    output_path.write_bytes(tflite_model)
    
    size_kb = output_path.stat().st_size / 1024
    print(f"[导出] ✅ TFLite 模型已保存: {output_path}")
    print(f"[导出]    大小: {size_kb:.1f} KB")
    
    return str(output_path)


def convert_all_variants(h5_path: str = None):
    """
    导出所有版本的 TFLite 模型
    
    Args:
        h5_path: H5 模型路径
    """
    if h5_path is None:
        h5_path = str(MODELS_DIR / 'best_model.h5')
    
    variants = {
        'int8': MODELS_DIR / 'ecg_model_int8.tflite',
        'float16': MODELS_DIR / 'ecg_model_fp16.tflite',
        'float32': MODELS_DIR / 'ecg_model_fp32.tflite',
    }
    
    for quant, path in variants.items():
        print(f"\n{'='*40}")
        convert_to_tflite(h5_path, str(path), quantization=quant)
    
    print(f"\n{'='*60}")
    print(" 所有变体导出完成")
    print(f"{'='*60}")
    for name, path in variants.items():
        size = path.stat().st_size / 1024 if path.exists() else 0
        print(f"  {name:<10}: {path.name:<25} {size:.1f} KB")
    print(f"{'='*60}")


def tflite_to_c_array(
    tflite_path: str = None,
    output_path: str = None,
    variable_name: str = "ecg_model_data",
    guard_name: str = "ECG_MODEL_DATA_H",
) -> str:
    """
    将 TFLite 模型转换为 C 语言头文件 (用于 TFLite Micro)
    
    Args:
        tflite_path: TFLite 文件路径
        output_path: 输出 .h 文件路径
        variable_name: C 变量名
        guard_name: 头文件宏守卫 (双模型部署时需不同值)
        
    Returns:
        输出文件路径
    """
    if tflite_path is None:
        tflite_path = str(MODELS_DIR / TFLITE_CONFIG['output_filename'])
    if output_path is None:
        output_path = str(MODELS_DIR / TFLITE_CONFIG['c_array_header'])
    
    print(f"\n{'='*60}")
    print(" TFLite → C 数组导出")
    print(f"{'='*60}\n")
    
    # 读取 TFLite 模型
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    # 获取模型信息
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print(f"[导出] 模型大小: {len(model_data)} bytes ({len(model_data)/1024:.1f} KB)")
    print(f"[导出] 输入形状: {input_details['shape']}")
    print(f"[导出] 输入类型: {input_details['dtype']}")
    print(f"[导出] 输出形状: {output_details['shape']}")
    print(f"[导出] 输出类型: {output_details['dtype']}")
    
    # 生成 C 数组
    lines = []
    lines.append("// Auto-generated by export.py from ECG DL Project")
    lines.append(f"// Model: {Path(tflite_path).name}")
    lines.append(f"// Size: {len(model_data)} bytes ({len(model_data)/1024:.1f} KB)")
    lines.append(f"// Input: {input_details['shape']}, Type: {input_details['dtype']}")
    lines.append(f"// Output: {output_details['shape']}, Type: {output_details['dtype']}")
    lines.append("//")
    lines.append(f"// Inference: TFLite Micro compatible")
    lines.append("")
    lines.append(f"#ifndef {guard_name}")
    lines.append(f"#define {guard_name}")
    lines.append("")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append(f"// Model input size: {INFERENCE_CONFIG['window_size']} samples")
    lines.append(f"// Model classes: {len(CLASS_NAMES)} ({', '.join(CLASS_NAMES)})")
    lines.append(f"const int ECG_MODEL_INPUT_SIZE = {INFERENCE_CONFIG['window_size']};")
    lines.append(f"const int ECG_MODEL_NUM_CLASSES = {len(CLASS_NAMES)};")
    lines.append("")
    lines.append(f"// TFLite model binary ({len(model_data)} bytes)")
    lines.append(f"const unsigned char {variable_name}[] = {{")
    
    # 格式化输出（每行16字节）
    for i in range(0, len(model_data), 16):
        chunk = model_data[i:i+16]
        hex_bytes = ', '.join(f'0x{b:02x}' for b in chunk)
        if i + 16 < len(model_data):
            lines.append(f"    {hex_bytes},")
        else:
            lines.append(f"    {hex_bytes}")
    
    lines.append("};")
    lines.append("")
    
    # 模型长度变量
    lines.append(f"const int {variable_name}_len = {len(model_data)};")
    lines.append("")
    lines.append(f"#endif // {guard_name}")
    
    # 写入文件
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    
    print(f"[导出] ✅ C 头文件已生成: {output_path}")
    print(f"[导出]    行数: {len(lines)}")
    
    return output_path


def verify_tflite_on_esp32_data(tflite_path: str = None):
    """
    在 PC 上模拟 TFLite Micro 推理, 
    输出测试向量用于 ESP32 验证
    """
    if tflite_path is None:
        tflite_path = str(MODELS_DIR / TFLITE_CONFIG['output_filename'])
    
    print(f"\n{'='*60}")
    print(" 生成 ESP32 测试向量")
    print(f"{'='*60}\n")
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # 生成测试输入
    test_input = np.sin(np.linspace(0, 4*np.pi, INFERENCE_CONFIG['window_size']))
    test_input = test_input.astype(input_details['dtype']).reshape(1, -1, 1)
    
    # TFLite 推理
    interpreter.set_tensor(input_details['index'], test_input)
    interpreter.invoke()
    test_output = interpreter.get_tensor(output_details['index'])[0]
    
    # 输出测试向量
    print("[导出] 测试输入 (前20个样本):")
    print(", ".join(f"{v}" for v in test_input[0, :20, 0]))
    print(f"\n[导出] 测试输出: {test_output}")
    print(f"[导出] 预测类别: {CLASS_NAMES[np.argmax(test_output)]}")
    
    # 保存测试向量到文件
    output_path = MODELS_DIR / 'esp32_test_vector.h'
    with open(output_path, 'w') as f:
        f.write("// ESP32 TFLite Micro 测试向量\n")
        f.write("// 用于验证 ESP32 推理结果与 PC 端一致\n\n")
        f.write(f"#define TEST_INPUT_SIZE {INFERENCE_CONFIG['window_size']}\n")
        f.write("const int8_t test_input[TEST_INPUT_SIZE] = {\n")
        for i in range(0, len(test_input[0]), 16):
            chunk = test_input[0, i:i+16, 0]
            f.write("    " + ", ".join(f"{v}" for v in chunk) + ",\n")
        f.write("};\n")
        f.write(f"\n// 预期输出: {test_output}\n")
        f.write(f"// 预期类别: {np.argmax(test_output)} ({CLASS_NAMES[np.argmax(test_output)]})\n")
    
    print(f"[导出] ✅ 测试向量已保存: {output_path}")
    
    return test_input, test_output


def export_pipeline(h5_path: str = None):
    """
    完整导出流水线
    H5 → INT8 TFLite → C 数组 → 测试向量
    """
    print(f"\n{'='*60}")
    print(" 完整导出流水线")
    print(f"{'='*60}\n")
    
    # 1. 转换到 TFLite
    tflite_path = convert_to_tflite(h5_path, quantization='int8')
    
    # 2. 生成 C 头文件
    c_header = tflite_to_c_array(tflite_path)
    
    # 3. 生成测试向量
    verify_tflite_on_esp32_data(tflite_path)
    
    print(f"\n{'='*60}")
    print(" ✅ 导出完成!")
    print(f"{'='*60}")
    print(f"  请将以下文件添加到 ESP32 项目中:")
    print(f"    1. {c_header}")
    print(f"    2. 实现 TFLite Micro 推理")
    
    return tflite_path, c_header


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ECG 模型导出")
    parser.add_argument("--h5", type=str, default=None, help="H5 模型路径")
    parser.add_argument("--all", action="store_true", help="导出所有量化变体")
    parser.add_argument("--pipeline", action="store_true", help="完整导出流水线")
    parser.add_argument("--to-c", action="store_true", help="仅生成 C 数组")
    parser.add_argument("--verify", action="store_true", help="生成验证测试向量")
    
    args = parser.parse_args()
    
    if args.pipeline:
        export_pipeline(args.h5)
    elif args.all:
        convert_all_variants(args.h5)
    elif args.to_c:
        tflite_to_c_array()
    elif args.verify:
        verify_tflite_on_esp32_data()
    else:
        convert_to_tflite(args.h5)
        tflite_to_c_array()