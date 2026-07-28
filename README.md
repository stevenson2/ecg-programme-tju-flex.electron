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

## AI 异常检测

### Phase 1 最佳模型

| 指标 | 数值 |
|------|------|
| 数据 | MIT-BIH (87K) + INCART (176K) = 263K beat级心拍 |
| 模型 | CNN v2, 15K 参数, INT8 ~15KB |
| Loss | FocalLoss (γ=1.0, α=0.75, bug已修复) |
| Acc | **93.98%** |
| AUC | **0.9716** |
| Abnormal Recall | 72% |
| 部署 | `models/ecg_model.tflite` (24.8 KB) |

Phase 2 计划: 自监督预训练 + 600K 模型, 目标 Recall ≥88%.
详见 [ROADMAP.md](ROADMAP.md)

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
| `train.py` | 完整训练流程 | `--epochs` `--batch-size` `--v3` `--resnet` `--merged` |
| `evaluate.py` | H5/TFLite 精度对比 | `--compare` `--h5` `--tflite` |
| `export.py` | INT8 量化 + C 数组 | `--pipeline` `--all` `--to-c` |
| `07_pc_inference.py` | 实时推理 + 基准 | `--source` `--benchmark` |

新增模块：

| 模块 | 功能 |
|------|------|
| `losses/focal_loss.py` | Focal Loss + Label Smoothing + Mixup + ECG 增强 |
| `models/resnet_lite_1d.py` | ECG-ResNet-Lite (Small 25K / Medium 55K / Large 80K) |
| `models/cnn_1d.py` | CNN v1/v2/v3 (含 v3: 30K, 4 层, Dropout 正则化) |
| `data/preprocess.py` | MIT-BIH 预处理 (含 ESP32 滤波器匹配) |
| `data/preprocess_ptbxl.py` | PTB-XL 预处理 |
| `data/preprocess_svdb.py` | SVDB 预处理 |

### GPU 训练 (WSL2)

Windows TensorFlow ≥2.11 原生不支持 GPU，需通过 WSL2：

```bash
# PowerShell (管理员)
wsl --install -d Ubuntu-24.04
wsl

# WSL 内
sudo apt update && sudo apt install python3-pip -y
pip install --break-system-packages tensorflow[and-cuda] -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --break-system-packages numpy scipy pandas wfdb matplotlib scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple

# 设置 CUDA 库路径
echo 'export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 训练
cd /mnt/c/Users/cai/OneDrive/Desktop/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl
python3 train.py --epochs 200 --batch-size 128
```

> ⚠️ RTX 5070 (compute capability 12.0a) 首次运行需 JIT 编译 CUDA 内核 (~30 min)，后续缓存后 ~7ms/step。

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
