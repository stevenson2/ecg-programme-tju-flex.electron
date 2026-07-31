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

### 3. 长时训练任务: 交给用户终端 + Loss 可视化
- 需要长时间执行的训练任务，**不得**由 Agent 在后台悄悄运行；
  必须给出完整终端命令，让用户在自己的终端运行。
- 训练的同时必须启动 Loss 可视化：给出两个终端的命令
  （终端 A 训练 + 终端 B `plot_history.py --watch --show` 实时监控）。
- 示例格式：
  ```
  在 WSL2 Ubuntu 终端 A（训练）中运行：
  $ cd /mnt/c/Users/cai/OneDrive/Desktop/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl
  $ python3 train.py --resnet-large --incart --sliding-dup 1

  在 WSL2 Ubuntu 终端 B（Loss 可视化，每 30s 刷新实时曲线）中运行：
  $ python3 plot_history.py --csv models/train_history.csv --watch 30 --show
  ```
- Agent 负责：训练前完成代码正确性验证（冒烟测试/编译检查），给出命令，
  在用户训练完成后评估与汇总结果。

### 4. Git 状态检查
- 每次做出**大更改之前**，先运行 `git status` 检查当前工作区状态。
- 每次修改**成功生效之后**（编译通过/实验完成/归档完成），再运行 `git status`
  确认变更范围符合预期。
- 只提交用户明确要求的文件；大文件（模型权重 .h5/.tflite 等）不得擅自提交。

### 5. 回答可信度要求
- 回答技术问题/做项目决策时，在必要的情形使用**联网搜索**（webfetch）与
  **文献阅读**来支撑结论，保证内容可信度。
- 发现用户陈述中的错误时，必须**直接指出**，不得掩盖或附和。
- 不确定的信息必须明确标注"不确定"，不得编造数据、引用或指标。
