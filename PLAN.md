# 基于深度学习的 ECG 异常检测 — 完整实施方案

## 项目概述

在现有 **ESP32-S3 ECG 采集系统** 基础上，新增基于深度学习的 ECG 异常检测功能，采用 **PC 端训练 + ESP32-S3 边缘推理** 的混合架构。

---

## 一、整体架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          ESP32-S3 (双核边缘推理)                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Core 1 (数据采集, 原主循环)              Core 0 (AI 推理协处理器)            │
│  ┌─────────────────────────────┐        ┌────────────────────────┐          │
│  │ ADC采样 (500Hz)             │        │ 等待环形缓冲满250样本   │          │
│  │ 双级梳状滤波(50/100Hz)      │ Free-  │ 预处理 (归一化)         │          │
│  │ HPF 0.5Hz + LP 40Hz        │ RTOS   │ TFLite Micro 推理       │          │
│  │ Pan-Tompkins 心率检测       │ 环形    │ (INT8 1D-CNN, ~5ms)    │          │
│  │ BLE 批量发送 (4帧合包)      │ 缓冲    │ 写回推理结果            │          │
│  │ 温度监控                    │────────→│ (正常/异常 + 置信度)    │          │
│  └──────────────┬──────────────┘        └────────────────────────┘          │
│                 │                                                           │
│                 ▼                                                           │
│  BLE 数据包扩展格式:                                                         │
│  "clean,noisy,filtered,bpm,sqi,abnormal_flag,confidence;"                    │
│                                                                              │
│  PC端(串口) 扩展格式:                                                        │
│  "clean,noisy,filtered,bpm,true_bpm,sqi,motion,abnormal_flag,confidence;"    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                             PC 端 (离线训练)                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  pc_tools/ecg_dl/                                                            │
│  ├── 01_download_data.py     # 从 Gitee 下载 MIT-BIH 数据集                  │
│  ├── 02_preprocess.py        # 心拍分割、重采样、归一化、数据增强             │
│  ├── 03_train.py             # 训练 1D-CNN (TensorFlow)                      │
│  ├── 04_evaluate.py          # 评估精度、混淆矩阵、ROC曲线                    │
│  ├── 05_export_tflite.py     # INT8 量化 + 导出 TFLite                       │
│  ├── 06_export_c_array.py    # 模型权重 → C 头文件 (.h)                       │
│  ├── 07_pc_inference.py      # PC端实时推理 (串口/BLE/文件输入)               │
│  └── models/                 # 训练好的模型文件                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                          Flutter App (手机端推理)                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  作为第二优先级，在 ESP32-S3 边缘推理稳定后实施                                │
│  - 使用 tflite_flutter 插件加载同一模型                                        │
│  - 接收 BLE 数据后滑动窗口推理                                                │
│  - 在波形图上高亮异常段                                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、模型设计

### 超轻量 1D-CNN 结构

```
输入层: 1 × 250 (1秒 ECG 片段 @ 250Hz)
│
├─ Conv1D: filters=8, kernel_size=5, activation=ReLU           → 246 × 8
├─ BatchNormalization (推理时折叠到卷积权重中)                   → 246 × 8
├─ MaxPooling1D: pool_size=2                                    → 123 × 8
│
├─ Conv1D: filters=8, kernel_size=3, activation=ReLU           → 121 × 8
├─ BatchNormalization                                          → 121 × 8
├─ MaxPooling1D: pool_size=2                                    → 60 × 8
│
├─ Conv1D: filters=16, kernel_size=3, activation=ReLU          → 58 × 16
├─ GlobalAveragePooling1D                                       → 16
│
├─ Dense: units=16, activation=ReLU                             → 16
├─ Dropout(0.3) (训练时使用，推理时移除)                         → 16
│
└─ Dense: units=2, activation=Softmax                           → 2
     └─ [Normal, Abnormal]
```

| 指标 | 值 |
|------|-----|
| 参数量 (FP32) | ~5,200 (约 20KB) |
| 参数量 (INT8) | ~5,200 (约 5.2KB) |
| TFLite Micro 运行时 RAM | ~16KB |
| 中间特征图 RAM | ~25KB |
| **总计额外 RAM** | **~50KB** |
| 推理时间 @240MHz | ~4-7ms |
| 推理周期 | 每 250 样本一次 (0.5s 滑动窗口, 步进125样本) |
| 预期二分类精度 | 93-95% |

---

## 三、数据集方案

### MIT-BIH Arrhythmia Database

| 项 | 详情 |
|----|------|
| 获取方式 | Gitee 镜像: `git clone https://gitee.com/mirrors/mit-bih-arrhythmia-database.git` |
| 数据量 | 48条记录 × 30分钟, 360Hz |
| 标注类别 | 15种心律类型 → 映射为 **正常(N) / 异常(A)** 二分类 |
| 心拍数 | ~110,000 个标注心拍 |
| 标签映射 | N={NORMAL,LBBB,RBBB,APB...} → 正常; A={PVC,VF,VFL,VT...} → 异常 |
| 预处理流程 | 360Hz→250Hz重采样 → R-peak定位 → 切割心拍(以R峰为中心的250点窗口) |

### 数据增强策略

| 增强方法 | 描述 | 倍数 |
|---------|------|------|
| 加性高斯噪声 | σ=0.01-0.05 | 2× |
| 基线漂移 | 低频正弦波叠加 | 2× |
| 时间缩放 | 0.9-1.1倍拉伸 | 2× |
| 幅度缩放 | 0.8-1.2倍放大 | 2× |
| **合计** | 5类组合 | ~10× |

---

## 四、文件组织

### 新增文件

```
ESP32-S3 固件 (C++):
├── include/ai_inference/
│   ├── ai_inference.h           # 推理模块 API
│   ├── ecg_model_data.h         # TFLite 模型权重 (C数组)
│   └── tflite_settings.h        # TFLite Micro 配置
├── src/ai_inference/
│   ├── ai_inference.cpp          # 推理模块实现
│   └── ring_buffer.cpp           # 环形缓冲(可选,或用FreeRTOS队列)
└── test/
    └── ai_inference_test.cpp    # 单元测试

PC 工具 (Python):
├── pc_tools/ecg_dl/
│   ├── __init__.py
│   ├── config.py                # 全局配置
│   ├── data/
│   │   ├── download.py           # MIT-BIH 下载
│   │   ├── preprocess.py         # 预处理
│   │   └── dataset.py            # TF Dataset 构建
│   ├── models/
│   │   ├── cnn_1d.py             # 1D-CNN 模型定义
│   │   └── utils.py              # 工具函数
│   ├── train.py                  # 训练脚本
│   ├── evaluate.py               # 评估脚本
│   └── export.py                 # 导出 TFLite + C数组

文档:
└── docs/
    └── ai_inference.md          # 推理模块使用文档
```

### 需要修改的文件

| 文件 | 修改内容 |
|------|---------|
| `src/main.cpp` | 新增 #include "ai_inference/ai_inference.h", 在loop中调用推理 | 
| `include/bluetooth/ble.h` | 如需要扩展消息格式 |
| `platformio.ini` | 新增 TFLite Micro 库依赖 |
| `pc_tools/ecg_plotter.py` | 新增异常标签显示 |

---

## 五、分步实施计划

### 阶段 1: PC 端数据准备与模型训练 (1-2周)

| 步骤 | 具体内容 | 产出 |
|------|---------|------|
| 1.1 | 创建 `pc_tools/ecg_dl/` 目录结构 | 项目骨架 |
| 1.2 | 从 Gitee 克隆 MIT-BIH 数据集 | 原始数据 (.dat/.hea/.atr) |
| 1.3 | 数据预处理：重采样(360→250Hz)、R-peak定位、心拍切割(250点/拍) | 预处理后的 NumPy 数组 |
| 1.4 | 标签映射：15类 → 二分类(正常/异常) | 标注文件 |
| 1.5 | 数据增强 + 训练/验证/测试集划分 (60/20/20) | TFRecord 或内存数据集 |
| 1.6 | 实现 1D-CNN 模型并训练 | 训练好的 .h5 模型 |
| 1.7 | 评估：混淆矩阵、ROC曲线、精度/召回率/F1 | 评估报告 |
| 1.8 | INT8 量化 + TFLite 导出 | .tflite 文件 (<10KB) |
| 1.9 | 模型权重 → C 数组 (.h 文件) | `ecg_model_data.h` |

### 阶段 2: ESP32-S3 边缘推理移植 (1周)

| 步骤 | 具体内容 | 产出 |
|------|---------|------|
| 2.1 | 调研 TFLite Micro for ESP32 (或 ESP-DL) | 技术选型报告 |
| 2.2 | 在 `platformio.ini` 中添加依赖 | 编译通过 |
| 2.3 | 实现推理模块 `src/ai_inference/ai_inference.cpp` | 推理 API |
| 2.4 | 模型权重文件 `ai_inference/ecg_model_data.h` 加入固件 | 权重嵌入 |
| 2.5 | 实现环形缓冲或 FreeRTOS 消息队列 (Core1→Core0) | 数据管道 |
| 2.6 | 修改 `main.cpp` 集成推理调用 | 功能集成 |
| 2.7 | 扩展 BLE/串口数据包格式 (追加 abnormal_flag) | 数据结构更新 |

### 阶段 3: PC 端验证与调优 (3-5天)

| 步骤 | 具体内容 | 产出 |
|------|---------|------|
| 3.1 | 实现 `pc_tools/ecg_dl/07_pc_inference.py` | PC端推理工具 |
| 3.2 | 在模拟器模式下对比推理结果与真实标签 | 精度验证 |
| 3.3 | 集成到 `ecg_plotter.py`：可视化异常标签 | 实时可视化 |
| 3.4 | 端到端延迟测试 | 性能报告 |

### 阶段 4: Flutter App 端推理 (后续，可选)

| 步骤 | 具体内容 | 产出 |
|------|---------|------|
| 4.1 | 添加 tflite_flutter 依赖 | pubspec.yaml 更新 |
| 4.2 | 模型嵌入 Flutter 项目 | assets/ 加载 |
| 4.3 | BLE 数据缓冲 + 滑动窗口推理 | 推理服务 |
| 4.4 | UI 更新：异常段高亮 + 告警显示 | UI 更新 |

---

## 六、关键技术选择

### TFLite Micro vs ESP-DL

| 对比维度 | TFLite Micro | ESP-DL (Espressif 官方) |
|---------|-------------|----------------------|
| 维护方 | Google + 社区 | Espressif |
| ESP32-S3 优化 | 通用优化 | ✅ Xtensa LX7 SIMD 深度优化 |
| 大小 | ~16KB RAM | ~20KB RAM |
| 算子支持 | 较全 (Conv/Dense/ReLU/Softmax...) | 有限 (CNN为主) |
| 社区资源 | 丰富 | 较少 |
| 部署难度 | 中等 (需手动导模型) | 低 (ESP-DL 自己量化) |
| **推荐** | ✅ 首选 | ⭐ 备选 (性能更好但灵活性低) |

**推荐：TFLite Micro** — 生态系统更成熟，社区资源多，便于后续扩展和调试。

但如果遇到性能瓶颈，可迁移到 ESP-DL 做进一步优化。

---

## 七、硬件资源预算

| 资源 | 当前使用 | 新增使用 | 总计 | 余量 |
|------|---------|---------|------|------|
| Flash (4MB) | ~1.8MB | ~80KB (TFLite Micro runtime) | ~1.88MB | **~2.1MB** ✅ |
| | | ~5KB (模型权重) | | |
| | | ~15KB (额外代码) | | |
| PSRAM (2MB) | ~500KB | ~50KB (推理缓冲区) | ~550KB | **~1.4MB** ✅ |
| RAM (512KB SRAM) | ~250KB | ~16KB (TFLite arena) | ~266KB | **~246KB** ✅ |
| | | ~25KB (特征图缓冲) | | |
| CPU Core 0 | 空闲 | 推理任务 | 100% | 正常 |
| CPU Core 1 | 100% (采样+滤波+BLE) | 不变 | 100% | 不变 |

**结论：资源充足，无需担心。**

---

## 八、风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| TFLite Micro 与 ESP32-S3 编译兼容性问题 | 中 | 高 | 提前验证 HelloWorld 示例；备用 ESP-DL |
| 量化模型精度下降 >3% | 中 | 中 | 使用量化感知训练(QAT)；或回退FP16推理 |
| Core 0 推理延迟导致 BLE 数据堆积 | 低 | 中 | 推理频率降为每250样本一次；设置超时跳过 |
| 训练数据分布与真实 ECG 不匹配 | 高 | 高 | 使用你的 ECG 模拟器数据进行迁移学习/微调 |
| MIT-BIH 原始下载源失效 | 中 | 中 | 使用 PhysioNet 官方源；或本地备份 |

---

## 九、预期成果

| 里程碑 | 时间 | 完成标准 |
|--------|------|---------|
| ✅ M1: PC端训练完成 | 第1-2周 | 模型精度 > 93%, TFLite 导出成功 |
| ✅ M2: ESP32-S3 推理成功 | 第3周 | 模拟器模式下推理结果与 PC 端一致 |
| ✅ M3: 端到端集成测试 | 第4周 | BLE 数据正确携带异常标签，波形图显示异常段 |
| ⏸ M4: Flutter App 推理 | 可选 | 手机端推理延迟 < 20ms |

---

## 十、下一步行动

**立即开始第一步骤：**
1. 切换到 **ACT MODE**
2. 在 `pc_tools/` 下创建 `ecg_dl/` 项目骨架
3. 从 Gitee 拉取 MIT-BIH 数据集
4. 开始数据预处理脚本编写

**技术栈确认：**
- [x] 硬件平台：ESP32-S3 (240MHz, 2MB PSRAM)
- [x] 边缘推理框架：TFLite Micro
- [x] 训练框架：TensorFlow 2.x
- [x] 训练数据：MIT-BIH Arrhythmia Database (Gitee 镜像)
- [x] 模型：1D-CNN (INT8量化, <10KB)
- [x] 推理频率：每250样本 (0.5-1s间隔)

---

*本计划于 2026-06-22 制订，可根据实际情况调整。*