# ESP32-ECG 心电采集与 AI 异常检测系统

> **ESP32-S3-SUPERMINI (ESP32S3FH4R2) | PlatformIO + Arduino | 250Hz 采样 | BLE NUS | TFLite Micro AI 推理**

## 目录

1. [项目简介](#项目简介)
2. [系统架构](#系统架构)
3. [AI 异常检测概述](#ai-异常检测概述)
4. [快速开始](#快速开始)
5. [项目结构](#项目结构)
6. [开发板说明](#开发板说明)
7. [PC 端工具](#pc-端工具)
8. [手机端 App](#手机端-app)
9. [温度与功耗诊断](#温度与功耗诊断)
10. [核心技术要点](#核心技术要点)
11. [核心配置](#核心配置)

---

## 项目简介

基于 ESP32-S3 的便携式心电采集系统，集成 **深度学习异常检测** 功能：
- **实时采集**: 250Hz 三通道心电信号（clean / noisy / filtered）
- **双模输入**: 软件模拟发生器 或 真实 AFE 模拟前端采集
- **AI 异常检测**: 1D-CNN 模型 (TFLite Micro)，ESP32-S3 Core 0 独立推理
- **BLE 透传**: Nordic UART Service，手机 App 实时波形显示
- **PC 训练工具链**: 完整的 MIT-BIH 数据下载、预处理、训练、评估、导出流水线

**硬件平台**：ESP32-S3-SUPERMINI (ESP32S3FH4R2)  
**开发框架**：PlatformIO + Arduino  
**采样率**：250Hz (每 4ms 一个样本)  
**AI 框架**：TensorFlow 2.x (训练) + TFLite Micro (边缘推理)  
**PC 工具**：Python (TensorFlow + matplotlib + pyserial)  
**手机 App**：Flutter (flutter_blue_plus + provider)

---

## 系统架构

```
+----------------------------------------------------------+
|                    ESP32-S3 (双核)                         |
|                                                          |
|  Core 1 (数据采集, 原主循环)          Core 0 (AI 推理)     |
|  +-----------------------------+    +----------------+   |
|  | 模拟器/ADC  ->  梳状滤波    |    | 环形缓冲 250点  |   |
|  |  HP 0.5Hz  ->  LP 40Hz    |--->| Z-score 归一化  |   |
|  |  Pan-Tompkins 心率检测     |    | TFLite Micro    |   |
|  |  BLE 批量发送 (4帧合包)    |    | (INT8 1D-CNN)   |   |
|  +-----------------------------+    +----------------+   |
|                |                          |               |
|                v                          v               |
|  串口 CSV (9列) + abnormal_flag      BLE NUS TX Notify   |
+-----------------------+----------------------------------+
                        |
           +------------+------------+
           v                         v
     pc_tools/ecg_plotter.py    ecg_app (Flutter)
     (Python 实时绘图)          (手机端心电监测)
     + AI 异常标签显示          + AI 异常高亮
```

**串口/BLE CSV 数据格式（每行）：**
```
<clean>,<noisy>,<filtered>,<bpm>,<true_bpm>,<sqi>,<motion>,<abnormal_flag>,<confidence>
```
示例：`0.253,-0.187,0.241,75,75,0.87,0,0,0.012`

| 列 | 字段 | 说明 |
|----|------|------|
| 1 | clean | 纯净心电 / 去偏置 ADC 信号 (V) |
| 2 | noisy | 含噪声原始信号 (V) |
| 3 | filtered | 数字滤波后信号 (V) |
| 4 | bpm | ESP32 板上心率检测值 |
| 5 | true_bpm | 真实心率 (仅模拟器模式) |
| 6 | sqi | 信号质量指数 (0-1) |
| 7 | motion | 运动标志 (0/1) |
| 8 | abnormal_flag | AI 异常标志 (0=正常, 1=异常) |
| 9 | confidence | AI 异常置信度 (0-1) |


---

## AI 异常检测概述

### 整体流程

```
PC (离线训练)                            ESP32-S3 (边缘推理)
+--------------------------+            +---------------------------+
| 1. MIT-BIH 数据集下载     |            | 采样: 250Hz 滤波后 ECG     |
| 2. 预处理: 重采样+心拍切割|            | 环形缓冲: 250 样本累积     |
| 3. 标签: 15类 -> 二分类   |  模型     | Z-score 归一化              |
| 4. 原始心拍 (无增强)      |  导出     | INT8 量化输入               |
| 5. 1D-CNN 训练 (30 epoch) | =======>  | TFLite Micro 推理 (~5ms)    |
| 6. INT8 量化 -> TFLite    |  .tflite  | 输出: Normal/Abnormal + conf |
| 7. 模型权重 -> C 头文件   |  .h       | 串口/BLE 追加异常标签       |
+--------------------------+            +---------------------------+
```

### 数据集

使用 **MIT-BIH Arrhythmia Database** (PhysioNet 官方源)：

| 项目 | 详情 |
|------|------|
| 记录数 | 48 条 x 30 分钟 |
| 原始采样率 | 360 Hz |
| 标注心拍 | ~110,000 个 |
| 标注类别 | 15 种心律类型 |
| 映射方式 | AAMI EC57 标准 -> 二分类 (Normal / Abnormal) |
| 预处理 | 360 -> 250Hz 重采样 -> R-peak 心拍切割 (250 点窗口) |

**标签映射规则 (AAMI 标准):**

| 二分类 | MIT-BIH 原始标注 |
|--------|-----------------|
| Normal (0) | N (正常), L (左束支阻滞), R (右束支阻滞), e (房性逸搏), j (结性逸搏) |
| Abnormal (1) | A (房性早搏), a (异常房早), J (结性早搏), S (室上性早搏), V (室性早搏), F (融合), ! (室扑), / (起搏), f (融合起搏) |

### 模型架构

**1D-CNN v2**，~15K 参数 (INT8 量化后 24.8 KB)：

```
Input (250, 1)
  |
  +-- Conv1D: filters=16, kernel_size=7, padding=same
  +-- BatchNormalization + ReLU
  +-- MaxPooling1D: pool_size=2              -> (125, 16)
  |
  +-- Conv1D: filters=32, kernel_size=5, padding=same
  +-- BatchNormalization + ReLU
  +-- MaxPooling1D: pool_size=2              -> (62, 32)
  |
  +-- Conv1D: filters=64, kernel_size=3, padding=same
  +-- BatchNormalization + ReLU
  +-- GlobalAveragePooling1D                 -> (64)
  |
  +-- Dense: 32 units, ReLU
  +-- Dropout: 0.4 (训练时, 推理时移除)
  +-- Dense: 2 units, Softmax
  |
  v
Output: [P(Normal), P(Abnormal)]
```

**模型特性：**

| 指标 | 值 |
|------|-----|
| 总参数量 | ~15,000 |
| FP32 模型大小 | 207 KB |
| INT8 模型大小 | **24.8 KB** |
| C 数组头文件 | 160 KB |
| 推理时间 @240MHz (ESP32-S3) | 预计 5-10ms |
| PC 推理时间 (XNNPACK) | <0.1ms |
| 输入窗口 | 250 样本 (1 秒 @250Hz) |
| 滑动步进 | 125 样本 (50% 重叠) |

### 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 学习率 | 0.001 (ReduceLROnPlateau: factor=0.5, patience=5) |
| 损失函数 | CategoricalCrossentropy |
| 评估指标 | Accuracy, Precision, Recall, AUC |
| 批大小 | 32 |
| 训练轮数 | 50 (EarlyStopping: patience=10, restore_best_weights) |
| 数据划分 | 训练 23 / 验证 7 / 测试 7 条记录 (按病人分组, 防数据泄露) |
| 随机种子 | 42 |

### 数据增强

> **重要发现：数据增强对跨病人泛化有严重负面影响。**
> 10x 增强后的模型精度仅 75.7%，去除增强后提升至 85.6%。
> 原因：增强产生的噪声模式失真，模型学到了增强伪影而非真实心律特征。
> 最终模型 **不使用数据增强**，仅用 87K 原始心拍训练。

### 评估结果

**测试集：7 条未见过病人记录，16,664 个心拍，按记录号分组（无数据泄露）**

| 指标 | FP32 (H5) | INT8 (TFLite) | 说明 |
|------|-----------|---------------|------|
| Accuracy | **85.63%** | **88.11%** | INT8 反而更高 (量化正则化效应) |
| AUC | 0.9423 | - | 优秀 |
| Normal Recall | 84.0% | 86.5% | 误报率仅 16% |
| Abnormal Recall | 88.0% | 89.5% | 漏检率仅 12% |
| Abnormal Precision | 77.0% | 79.0% | 告警中 ~21% 为误报 |
| 模型文件大小 | 207 KB | **24.8 KB** | 压缩 88% |

> INT8 量化不仅未损失精度，反而因量化噪声起到正则化作用，精度提升 2.5%。

### 模型进化史

| 阶段 | 数据 | 划分方式 | 精度 | 结论 |
|------|------|---------|------|------|
| 阶段 1 | 3 条记录, 10x 增强 | **样本随机划分** | 98.4% | 数据泄露假象 |
| 阶段 2 | 37 条记录, 10x 增强 | 按病人分组 | 75.7% | 增强毒化泛化 |
| 阶段 3 | 37 条记录, **无增强** | 按病人分组 | **85.6%** | 真实泛化能力 |

![模型进化](models/fig_evolution.png)

![模型总览](models/fig_model_summary.png)

### INT8 量化原理

```
FP32 推理:                       INT8 推理:
x_fp32 -> Conv1D -> ReLU -> ...    x_int8 = round(x_fp32 / Sx + Zx)
                                   y_int8 = Conv1D_int8(x_int8)
                                   y_fp32 = (y_int8 - Zy) * Sy

量化公式:
  Sx = (x_max - x_min) / 255       (scale)
  Zx = round(-x_min / Sx)          (zero_point)
  x_int8 = round(x_fp32 / Sx + Zx)
```

模型使用 TensorFlow Lite Converter 的 `tf.lite.Optimize.DEFAULT` 进行全整数量化：
- 权重: FP32 -> INT8 (离线量化)
- 激活值: 校准数据集统计 min/max -> INT8
- 输入/输出: INT8 (需手动量化/反量化)


---

## 快速开始

### 1. AI 模型训练 (PC 端)

```bash
cd pc_tools/ecg_dl

# 安装依赖
pip install -r requirements.txt

# 下载数据集 (最小测试集, 3 条记录)
python data/download.py --test-only

# 预处理数据
python data/preprocess.py --test

# 训练模型 (30 epochs, 约 3 分钟)
python train.py --epochs 30

# 评估 + H5/TFLite 精度对比
python evaluate.py --compare

# INT8 量化导出 + C 头文件生成
python export.py --pipeline
```

> 训练完成后，`models/` 目录包含 `ecg_model.tflite` (TFLite 模型) 和 `ecg_model_data.h` (ESP32 固件用 C 数组)。

### 2. PC 端推理验证

```bash
# 基准测试 (1000 次推理, 测量延迟)
python 07_pc_inference.py --benchmark

# 文件模式推理
python 07_pc_inference.py --source file --input test_ecg.npy

# 串口模式实时推理 (需 ESP32 连接)
python 07_pc_inference.py --source serial --port COM3
```

### 3. 编译烧录 ESP32 固件

```bash
# 1. 将生成的模型 C 头文件复制到固件目录
cp pc_tools/ecg_dl/models/ecg_model_data.h include/ai_inference/

# 2. 编译
pio run

# 3. 烧录到 SUPERMINI
pio run -t upload

# 4. 打开串口监视器
pio device monitor -p COM9 -b 115200
```

### 4. 使用 PC 绘图仪

```bash
python pc_tools/ecg_plotter.py
```

### 5. 使用手机 App

1. 进入 `ecg_app/` 目录
2. 执行 `flutter pub get`
3. 执行 `flutter run`
4. 扫描并连接 "ESP32-ECG" 设备

### 6. 使用工具箱

双击 `ESP32-ECG toolbox.bat` 启动菜单界面，可选择：
- [1] PC Plotter — 三通道波形绘图
- [2] Serial Monitor — 串口数据监视
- [3] Compile & Upload — 编译烧录固件

### 7. 串口指令

| 指令 | 功能 |
|------|------|
| `r` / `R` | 重置数字滤波器 |
| `s` / `S` | 重置信号发生器 |
| `m` / `M` | 切换 模拟/真实AFE 输入模式 |
| `t` / `T` | 打印温度状态详情 |
| `c` / `C` | 打印 CPU 当前频率 |
| `a` / `A` | 打印 AI 推理统计 (总次数/异常率/平均延迟) |


---

## 项目结构

```
ecg-programme-tju-flex.electron/
├── platformio.ini              # PlatformIO 构建配置 + TFLite Micro 依赖
├── README.md                   # 本文件
├── PLAN.md                     # 完整实施方案 (AI 异常检测)
├── LICENSE                     # 项目许可证
│
├── src/                        # ESP32 固件源码
│   ├── main.cpp                # 主程序入口 (含 AI 推理集成)
│   ├── ai_inference/
│   │   ├── ai_inference.cpp    # TFLite Micro 推理实现 (Core 0 FreeRTOS)
│   │   └── ecg_model_data.cpp  # 模型权重占位 (训练后替换)
│   ├── bluetooth/
│   │   └── ble.cpp             # BLE NUS UART 透传模块
│   ├── filter/
│   │   └── filter.cpp          # 三级数字滤波器 (HP/LP/Notch)
│   ├── heartrate/
│   │   └── heartrate.cpp       # 心率检测 (简化 Pan-Tompkins)
│   ├── signal_generator/
│   │   └── ecg_simulator.cpp   # 临床级心电信号生成器
│   ├── adc_afe/
│   │   └── afe_hal.cpp         # ADC 采集硬件抽象层
│   └── thermal/
│       ├── thermal.h           # 温度监测接口
│       └── thermal.cpp         # 温度监测实现
│
├── include/                    # 头文件 (对应 src/ 各模块)
│   └── ai_inference/
│       ├── ai_inference.h      # AI 推理模块 API
│       ├── tflite_settings.h   # TFLite Micro 配置 (Arena/核心/队列)
│       └── ecg_model_data.h    # 模型权重 C 数组 (由 export.py 生成)
│
├── pc_tools/                   # PC 端 Python 工具
│   ├── ecg_plotter.py          # 实时三通道波形绘图 + AI 异常标签
│   ├── ecg_dl/                 # ★ 深度学习训练/推理工具包
│   │   ├── requirements.txt    # Python 依赖清单
│   │   ├── config.py           # 全局配置 (数据集/训练/TFLite)
│   │   ├── train.py            # 一站式训练脚本
│   │   ├── evaluate.py         # 模型评估 (H5/TFLite 对比)
│   │   ├── export.py           # INT8 量化 + TFLite 导出 + C 数组生成
│   │   ├── 07_pc_inference.py  # PC 端实时推理 (串口/文件/基准)
│   │   ├── data/
│   │   │   ├── download.py     # MIT-BIH 数据集下载
│   │   │   ├── preprocess.py   # 心拍分割 + 重采样 + 数据增强
│   │   │   ├── dataset.py      # TensorFlow Dataset 构建
│   │   │   ├── raw/            # MIT-BIH 原始数据 (.dat/.hea/.atr)
│   │   │   └── processed/      # 预处理后的 NumPy 数组
│   │   └── models/
│   │       ├── cnn_1d.py       # 1D-CNN 模型定义
│   │       ├── utils.py        # 可视化工具 (混淆矩阵/ROC)
│   │       ├── *.h5            # 训练好的 Keras 模型
│   │       ├── *.tflite        # INT8 量化 TFLite 模型
│   │       └── *.h             # C 数组模型权重
│   ├── capture_debug.py
│   ├── hr_sim_verify.py
│   └── verify_filter_coeffs.py
│
├── ecg_app/                    # Flutter 手机端 App
│   ├── pubspec.yaml
│   └── lib/
│       ├── main.dart           # App 入口
│       ├── models/ecg_data.dart
│       ├── providers/ecg_provider.dart
│       ├── services/ble_service.dart
│       └── widgets/
│           ├── ecg_waveform.dart  # 实时波形 (CustomPainter)
│           └── info_panel.dart    # 信息面板
│
├── test/                       # 测试固件
│   └── adc_test.cpp
│
├── lib/                        # 第三方库
├── ecg_toolbox.ps1             # PowerShell 工具箱
└── ESP32-ECG toolbox.bat       # 工具箱入口
```


## 开发板说明

### ESP32-S3-SUPERMINI vs ESP32-S3-DevKitM-1

| 参数 | DevKitM-1 (原) | SUPERMINI (现) |
|------|---------------|----------------|
| **芯片** | ESP32-S3 | **ESP32S3FH4R2** |
| **Flash** | 8MB | **4MB** |
| **PSRAM** | 8MB Octal | **2MB Octal** |
| **USB 接口** | 外置 USB-UART | **内置 USB-Serial-JTAG** |
| **LED** | GPIO48 单色 | **GPIO48 RGB 共阳极** |
| **尺寸** | 54x28mm | **22.5x18mm** |

### 引脚映射

| 功能 | 引脚 | 说明 |
|------|------|------|
| BOOT 按键 | GPIO0 | 内部上拉，按下切换输入源 |
| AFE ADC 采集 | GPIO4 | ADC1_CH3 |
| RGB LED | GPIO48 | 共阳极，LOW=亮 |
| UART TX/RX | GPIO1/2 | 串口 115200 |

### AI 推理资源预算

| 资源 | 占用 | 说明 |
|------|------|------|
| Flash | ~100 KB | TFLite Micro Runtime + 模型 |
| SRAM | ~37 KB | Tensor Arena (32KB) + 代码/栈 |
| Core 0 | 推理任务 | FreeRTOS 任务，优先级 1 |
| Core 1 | 采样+滤波+BLE | 无变化 |

### 烧录说明

**自动烧录**：连接 USB-C -> `pio run -t upload`

**手动烧录**（自动失败时）：
1. 按住 **BOOT** 按钮
2. 短按 **RESET** 按钮
3. 松开 **BOOT** 按钮
4. 执行 `pio run -t upload`

---

## PC 端工具

### ecg_plotter.py

实时三通道波形显示（绿=纯净, 红=带噪, 蓝=滤波后），支持 AI 异常标签。

- **异常指示灯**：BPM 文字红色=异常检测中，绿色=正常
- **数据格式**：解析 9 列 CSV (含 abnormal_flag + confidence)

**快捷键**：
| 按键 | 功能 |
|------|------|
| 1/2/3 | 切换通道可见性 |
| 空格 | 暂停/恢复 |
| R | 复位视图 |
| Q | 退出 |

### ecg_dl/ 深度学习工具包

核心训练和推理工具：

| 脚本 | 功能 | 关键参数 |
|------|------|---------|
| `train.py` | 完整训练流程 | `--epochs` `--batch-size` `--quick-test` |
| `evaluate.py` | H5/TFLite 精度对比 | `--compare` `--h5` `--tflite` |
| `export.py` | INT8 量化 + C 数组 | `--pipeline` `--all` `--to-c` |
| `07_pc_inference.py` | 实时推理 + 基准 | `--source` `--benchmark` |

---

## 手机端 App

心电监测 App (Flutter)，通过 BLE NUS 连接 ESP32-ECG 设备。

**功能**：
- BLE 扫描连接 "ESP32-ECG"
- 三通道实时波形绘制 (CustomPainter)
- BPM 心率显示 + 信号质量指示
- 速度/幅度控制 (1s / 2s / 4s / 6s 窗口)

**BLE 服务 UUID**：
| 服务/特征值 | UUID |
|-------------|------|
| NUS Service | `6E400001-B5A3-F393-E0A9-E50E24DCCA9E` |
| TX (Notify) | `6E400002-B5A3-F393-E0A9-E50E24DCCA9E` |
| RX (Write) | `6E400003-B5A3-F393-E0A9-E50E24DCCA9E` |

---

## 温度与功耗诊断

系统内置芯片温度监测模块（`src/thermal/`），利用 ESP32-S3 内置温度传感器：

- 每秒采样一次，8 点滑动平均
- 过热保护：>65C 自动降频至 60MHz，<55C 恢复 240MHz

| 热源 | 贡献 |
|------|------|
| **BLE 协议栈** | 主要 |
| CPU 240MHz | 次要 |
| USB LDO | 次要 |


## 核心技术要点

### 数字滤波器 (`src/filter/filter.cpp`)

三级级联 IIR Biquad (直接 II 型转置结构)：
- **第1级**：二阶 Butterworth 高通 0.5Hz（去基线漂移）
- **第2级**：二阶 Butterworth 低通 40Hz（去肌电干扰）
- **第3级**：二阶 50Hz 陷波器 Q=20（工频陷零）
- 所有系数在 MATLAB 中设计，硬编码避免运行时计算

### 50Hz 梳状滤波器 (`main.cpp`)

双级级联滑动平均，利用 500Hz/50Hz = 10 的精确比：
- 第1级：10 抽头滑动平均 (50Hz 衰减 -59.6dB)
- 第2级：对第1级输出再做 10 抽头 (总计 -119.2dB)
- 同时抑制 100Hz 谐波
- 群延迟：20ms (< RR 间期 800ms)

### 心电信号模拟 (`src/signal_generator/ecg_simulator.cpp`)

- P-QRS-T 波用 5 个高斯函数叠加模拟
- 直流偏置 1.65V，心电峰峰值 ~1.3V
- 7 种噪声按 IEC60601-2-51 场景三分布

### 心率检测 (`src/heartrate/heartrate.cpp`)

- 简化 Pan-Tompkins QRS 检测算法
- 固定阈值 0.3V + 最小间隔 200ms (50点)
- 心率滑动平均 alpha=0.7
- 信号质量指数 (SQI) + 运动检测

### AI 推理模块 (`src/ai_inference/ai_inference.cpp`)

**架构**：FreeRTOS 任务绑定到 Core 0，与 Core 1 主循环通过环形缓冲 + 信号量解耦。

**数据流**：
```
Core 1 (main.cpp loop)              Core 0 (inference_task)
  |                                       |
  | ai_inference_push(sample)             | xSemaphoreTake()
  | -> 写入环形缓冲                        | -> 获取窗口数据
  | -> 每 125 样本触发                     | -> Z-score 归一化
  |    xSemaphoreGive()                   | -> 量化 INT8
  |                                       | -> TFLite Micro Invoke()
  |                                       | -> 反量化 -> Softmax
  |                                       | -> xQueueSend(result)
  |                                       |
  | ai_inference_pop_result()             |
  | <- 非阻塞读取结果                      |
  v                                       v
  串口/BLE 输出 (含 abnormal_flag)
```

**推理步骤**：
1. **归一化**：Z-score (x - mu) / sigma
2. **INT8 量化**：x_int8 = round(x_fp32 / scale + zero_point)
3. **TFLite Micro 推理**：1D-CNN 前向传播
4. **反量化 + Softmax**：y_fp32 = (y_int8 - zero_point) * scale -> Softmax
5. **判定**：P(Abnormal) > 0.5 -> 异常

**FreeRTOS 同步**：
- `g_data_ready_sem`：二进制信号量，推理任务等待数据就绪
- `g_mutex`：保护环形缓冲的互斥锁
- `g_result_queue`：结果队列 (深度 8)，主循环非阻塞读取

---

## 核心配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 250 Hz | 每样本间隔 4ms (原 500Hz 降为 250Hz) |
| CPU 频率 | 240 MHz | 性能模式 |
| BLE TX 功率 | +9 dBm | 原始最高功率 |
| 串口波特率 | 115200 | 数据输出频率 25Hz |
| BLE 设备名 | ESP32-ECG | - |
| 滤波器类型 | IIR Biquad + 梳状 | HP 0.5 -> LP 40 -> Notch 50 + Comb 50/100 |
| 心率算法 | Pan-Tompkins | 阈值 0.3V, 间隔 200ms |
| 缓冲区大小 | 1500 点 | 6 秒数据 |
| AI 模型 | 1D-CNN (1,082 参数) | TFLite Micro INT8 量化 |
| AI 输入窗口 | 250 样本 | 1 秒 @250Hz |
| AI 推理间隔 | 125 样本 | 0.5 秒一次 |
| AI 模型大小 | 11.8 KB | .tflite 文件 |
| AI Tensor Arena | 32 KB | SRAM 分配 |
| AI 核心 | Core 0 | 与采样/滤波/BLE (Core 1) 解耦 |
| 温度过热阈值 | 65C | 自动降频至 60MHz |
| 温度恢复阈值 | 55C | 自动恢复 240MHz |
