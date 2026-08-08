#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_pc_inference.py — PC 端实时 ECG 异常推理

功能:
  - 加载 TFLite 模型
  - 从 串口/BLE/文件 读取 ECG 数据
  - 滑动窗口推理 (步进 125 样本, 50% 重叠)
  - 实时显示异常标签和置信度

用法:
  python 07_pc_inference.py --source serial --port COM3
  python 07_pc_inference.py --source file --input test_ecg.npy
  python 07_pc_inference.py --benchmark
"""

import sys
import os
import io
import time
import argparse
import threading
from pathlib import Path
from collections import deque
import numpy as np

# 修复 Windows 控制台编码问题
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, OSError):
        pass

# 导入项目配置
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MODELS_DIR, TFLITE_CONFIG, INFERENCE_CONFIG,
    CLASS_NAMES, TARGET_FS, BEAT_WINDOW_SAMPLES
)


# ======================== TFLite 推理器 ========================

class ECGInferenceEngine:
    """基于 TFLite 的 ECG 异常检测推理引擎"""

    def __init__(self, model_path: str = None):
        """初始化推理引擎"""
        import tensorflow as tf

        if model_path is None:
            model_path = str(MODELS_DIR / TFLITE_CONFIG['output_filename'])

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please run: python export.py --pipeline"
            )

        print(f"[Inference] Loading TFLite model: {model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        self.input_shape = self.input_details['shape']
        self.input_dtype = self.input_details['dtype']
        self.output_dtype = self.output_details['dtype']

        self.window_size = self.input_shape[1]  # 250
        self.stride = self.window_size // 2      # 125

        size_kb = Path(model_path).stat().st_size / 1024
        print(f"[Inference] Input shape: {self.input_shape}, dtype: {self.input_dtype}")
        print(f"[Inference] Output shape: {self.output_details['shape']}, dtype: {self.output_dtype}")
        print(f"[Inference] Model size: {size_kb:.1f} KB")
        print(f"[Inference] Window: {self.window_size}, Stride: {self.stride}")

        self.inference_count = 0
        self.abnormal_count = 0
        self.total_time = 0.0
        self.last_result = {"label": "Normal", "confidence": 0.0, "abnormal": False}

    def preprocess(self, signal: np.ndarray) -> np.ndarray:
        """Z-score normalize + add batch/channel dims"""
        if np.std(signal) > 1e-6:
            signal = (signal - np.mean(signal)) / np.std(signal)
        else:
            signal = signal - np.mean(signal)
        # INT8 量化: x_int8 = round(x_fp32 / scale) + zero_point
        if self.input_dtype in (np.int8, np.uint8):
            input_scale = float(np.array(self.input_details.get('quantization_parameters', {}).get('scales', [1.0])).flatten()[0])
            input_zp = int(np.array(self.input_details.get('quantization_parameters', {}).get('zero_points', [0])).flatten()[0])
            signal = (signal / input_scale + input_zp).astype(self.input_dtype).reshape(1, -1, 1)
        else:
            signal = signal.astype(self.input_dtype).reshape(1, -1, 1)
        return signal

    def predict(self, signal: np.ndarray) -> dict:
        """Run inference on a signal window, return label/confidence"""
        t_start = time.perf_counter()
        input_tensor = self.preprocess(signal)

        self.interpreter.set_tensor(self.input_details['index'], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details['index'])[0]

        t_elapsed = (time.perf_counter() - t_start) * 1000

        pred_class = np.argmax(output)
        confidence = float(output[pred_class])

        # Dequantize INT8 output + Softmax
        if self.output_dtype in (np.int8, np.uint8):
            output_scale = float(np.array(self.output_details.get('quantization_parameters', {}).get('scales', [1.0])).flatten()[0])
            output_zp = int(np.array(self.output_details.get('quantization_parameters', {}).get('zero_points', [0])).flatten()[0])
            output_f = (output.astype(np.float32) - output_zp) * output_scale
            output_f = output_f - np.max(output_f)
            exp_out = np.exp(output_f)
            probs = exp_out / np.sum(exp_out)
            pred_class = np.argmax(probs)
            confidence = float(probs[pred_class])

        result = {
            "label": CLASS_NAMES[pred_class],
            "confidence": confidence,
            "abnormal": pred_class == 1,
            "probabilities": output,
            "latency_ms": t_elapsed
        }

        self.inference_count += 1
        if result["abnormal"]:
            self.abnormal_count += 1
        self.total_time += t_elapsed
        self.last_result = result

        return result

    def get_stats(self) -> dict:
        """Get inference statistics"""
        if self.inference_count == 0:
            return {"count": 0, "abnormal_rate": 0, "avg_latency_ms": 0}
        return {
            "count": self.inference_count,
            "abnormal_rate": self.abnormal_count / self.inference_count * 100,
            "avg_latency_ms": self.total_time / self.inference_count
        }


# ======================== Sliding Window Buffer ========================

class SlidingWindowBuffer:
    """Sliding window buffer for real-time streaming inference"""

    def __init__(self, window_size: int = 250, stride: int = 125):
        self.window_size = window_size
        self.stride = stride
        self.buffer = deque(maxlen=window_size * 4)
        self.sample_count = 0
        self.last_inference_sample = -stride

    def add_sample(self, value: float):
        self.buffer.append(value)
        self.sample_count += 1

    def add_samples(self, values: np.ndarray):
        for v in values:
            self.buffer.append(float(v))
        self.sample_count += len(values)

    def is_ready(self) -> bool:
        if len(self.buffer) < self.window_size:
            return False
        return self.sample_count - self.last_inference_sample >= self.stride

    def get_window(self) -> np.ndarray:
        buf_list = list(self.buffer)
        window = np.array(buf_list[-self.window_size:], dtype=np.float32)
        self.last_inference_sample = self.sample_count
        return window

    def reset(self):
        self.buffer.clear()
        self.sample_count = 0
        self.last_inference_sample = -self.stride


# ======================== Data Sources ========================

def serial_source(port: str, baud: int = 460800, timeout: float = 1.0):
    """Serial data source: parse ESP32 ECG data"""
    import serial
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        print(f"[Source] Serial connected: {port} @ {baud}")
        time.sleep(1.0)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"[Source] Serial connection failed: {e}")
        return

    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                filtered_val = float(parts[2].strip())
                yield filtered_val, line
            except ValueError:
                continue
        except serial.SerialException:
            print("[Source] Serial disconnected")
            break
        except Exception as e:
            print(f"[Source] Error: {e}")
            continue


def file_source(filepath: str, channel: int = 0):
    """File data source: supports .npy and .csv"""
    filepath = Path(filepath)
    if filepath.suffix == '.npy':
        data = np.load(filepath)
        if data.ndim > 1:
            data = data[:, channel] if data.shape[1] > channel else data.flatten()
        print(f"[Source] File loaded: {filepath} ({len(data)} samples)")
        for val in data:
            yield float(val), None
    elif filepath.suffix == '.csv':
        import csv
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    try:
                        yield float(row[0]), None
                    except ValueError:
                        continue
    else:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield float(line.split(",")[0]), line
                except ValueError:
                    continue


# ======================== Main ========================

def run_inference(engine, buffer, source_generator,
                  verbose=True, max_samples=None, alert_threshold=3):
    """Run real-time inference loop"""
    sample_idx = 0
    consecutive_abnormal = 0
    last_alert_time = 0.0

    print(f"\n{'='*60}")
    print(" ECG Real-time Anomaly Detection")
    print(f"{'='*60}")
    print(" Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    for value, raw_line in source_generator:
        if max_samples and sample_idx >= max_samples:
            break
        buffer.add_sample(value)
        sample_idx += 1
        if not buffer.is_ready():
            continue
        window = buffer.get_window()
        result = engine.predict(window)
        if result["abnormal"]:
            consecutive_abnormal += 1
        else:
            consecutive_abnormal = 0
        now = time.time()
        if consecutive_abnormal >= alert_threshold and (now - last_alert_time) > 2.0:
            print(f"\n  !! ALERT: {consecutive_abnormal} consecutive abnormal beats !!")
            last_alert_time = now
        if verbose:
            marker = "!!" if result["abnormal"] else "OK"
            print(f"  [{sample_idx:6d}] {marker} {result['label']:<10s}"
                  f"  conf={result['confidence']*100:5.1f}%"
                  f"  latency={result['latency_ms']:5.1f}ms"
                  f"  consec_abn={consecutive_abnormal}")
    stats = engine.get_stats()
    print(f"\n{'='*60}")
    print(" Inference Statistics")
    print(f"{'='*60}")
    print(f"  Total inferences: {stats['count']}")
    print(f"  Abnormal rate:    {stats['abnormal_rate']:.1f}%")
    print(f"  Avg latency:      {stats['avg_latency_ms']:.1f} ms")
    print(f"  Total samples:    {sample_idx}")
    print(f"{'='*60}")


def run_benchmark(engine, n_iterations=1000):
    """Inference performance benchmark"""
    print(f"\n[Benchmark] Running {n_iterations} inferences...")
    np.random.seed(42)
    latencies = []
    for i in range(n_iterations):
        t = np.linspace(0, 4 * np.pi, engine.window_size)
        signal = np.sin(t) + np.random.randn(engine.window_size) * 0.05
        signal = signal.astype(np.float32)
        result = engine.predict(signal)
        latencies.append(result['latency_ms'])
    latencies = np.array(latencies)
    print(f"\n[Benchmark] Results:")
    print(f"  Mean:     {np.mean(latencies):.3f} ms")
    print(f"  Median:   {np.median(latencies):.3f} ms")
    print(f"  P95:      {np.percentile(latencies, 95):.3f} ms")
    print(f"  P99:      {np.percentile(latencies, 99):.3f} ms")
    print(f"  Throughput: {1000/np.mean(latencies):.0f} infer/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Real-time Inference")
    parser.add_argument("--source", choices=["serial", "file"], default="serial")
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--baud", type=int, default=460800)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--alert-threshold", type=int, default=3)
    args = parser.parse_args()

    try:
        engine = ECGInferenceEngine(model_path=args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.benchmark:
        run_benchmark(engine)
        sys.exit(0)

    if args.source == "serial":
        import serial.tools.list_ports
        if args.port is None:
            ports = serial.tools.list_ports.comports()
            if ports:
                port = ports[0].device
            else:
                print("No serial port found.")
                sys.exit(1)
        else:
            port = args.port
        source_gen = serial_source(port, baud=args.baud)
    elif args.source == "file":
        if args.input is None:
            print("Error: --input required for file mode")
            sys.exit(1)
        source_gen = file_source(args.input)
    else:
        sys.exit(1)

    buffer = SlidingWindowBuffer(
        window_size=INFERENCE_CONFIG['window_size'],
        stride=INFERENCE_CONFIG['stride']
    )

    try:
        run_inference(engine=engine, buffer=buffer,
                      source_generator=source_gen,
                      verbose=not args.quiet,
                      max_samples=args.max_samples,
                      alert_threshold=args.alert_threshold)
    except KeyboardInterrupt:
        print("\nInference stopped.")
