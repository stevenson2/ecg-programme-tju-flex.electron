# ESP32-ECG 项目指南

## 项目概述
基于 ESP32-S3 的便携式心电采集系统，集成深度学习异常检测。
- 芯片: ESP32-S3-SUPERMINI (ESP32S3FH4R2)
- 框架: PlatformIO + Arduino
- 采样率: 250Hz 三通道 (clean/noisy/filtered)
- AI: TFLite Micro 1D-CNN INT8 推理
- BLE: Nordic UART Service (NUS)

## 项目结构
- `src/` — 固件源码
  - `main.cpp` — 主程序入口
  - `adc_afe/` — ADC/AFE 采集
  - `ai_inference/` — TFLite Micro 推理
  - `bluetooth/` — BLE 通信
  - `filter/` — 数字滤波 (HP+LP)
  - `heartrate/` — Pan-Tompkins 心率检测
  - `signal_generator/` — 模拟信号发生器
  - `thermal/` — 温度管理
- `pc_tools/` — Python PC 工具 (训练/绘图/调试)
- `ecg_app/` — Flutter 手机 App
- `test/` — 测试代码
- `include/` — 头文件
- `lib/` — 库文件

## 构建命令
- `pio run` — 编译固件
- `pio run -t upload` — 编译并上传
- `pio device monitor -b 115200` — 串口监视器
- `pio test` — 运行测试

## 编码规范
- C++: Arduino 风格，使用 `.cpp`/`.h` 扩展名
- Python: TensorFlow 训练脚本，使用 `argparse` 参数化
- 串口/BLE CSV 格式: `<clean>,<noisy>,<filtered>,<bpm>,<true_bpm>,<sqi>,<motion>,<abnormal_flag>,<confidence>`

## Agent 行为规范

### 1. 修改文件后编译检查
- 修改 C++ 文件后，自动运行 `pio run` 检查编译是否通过。
- 修改 Python 文件后，自动运行 `python -m py_compile <file>` 检查语法错误。
- 若编译/语法检查失败，必须修复后再标记任务完成。

### 2. 给用户的终端操作命令
- 要求用户执行终端操作时，必须给出**完整的命令行**。
- 必须指明在**何种终端**执行（如：WSL2 Ubuntu、Windows PowerShell、PlatformIO 终端等）。
- 示例格式：
  ```
  在 WSL2 Ubuntu 中运行：
  $ cd /mnt/c/Users/cai/OneDrive/Desktop/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl
  $ python3 train.py --incart --epochs 200
  ```
