# ESP32-ECG 心电采集与 AI 异常检测系统

> **ESP32-S3-WROOM-1-N16R8 (16MB Flash / 8MB Octal PSRAM) | 官方固件 = ESP-IDF 迁移工程 | 500Hz 采样 | BLE NUS | TFLite Micro + ESP-NN AI 推理 | Flutter App**

> **ⓘ 固件线状态（2026-09-01）**：旧 **Arduino + PlatformIO** 固件线已归档至 [`legacy_arduino/`](legacy_arduino/)（不再作为正式线）；**官方固件为 ESP-IDF 迁移工程** [`experiments/esp_idf_ecg_migration/`](experiments/esp_idf_ecg_migration/)（2026-08-28 转正，exp7c INT8 + ESP-NN，已于 2026-09-01 补齐录制链与 WiFi 记录 API）。下文涉及固件具体实现处，默认指 **IDF 官方线**，旧 Arduino 线内容仅作历史参考。

## 目录

1. [项目简介](#项目简介)
2. [快速开始](#快速开始)
3. [系统架构](#系统架构)
4. [项目结构](#项目结构)
5. [硬件平台](#硬件平台)
6. [AI 异常检测](#ai-异常检测)
7. [报警与心律分析](#报警与心律分析)
8. [数据记录与回放](#数据记录与回放)
9. [PC 端工具](#pc-端工具)
10. [手机端 App](#手机端-app)
11. [文档导航](#文档导航)
12. [核心配置](#核心配置)
13. [开发状态与路线](#开发状态与路线)

---

## 项目简介

基于 ESP32-S3 的**便携式单导联心电采集系统**：以 500Hz 实时采集心电信号，在板上完成数字滤波、心率检测与**深度学习逐拍异常检测**，异常时锁存报警并推送到手机 App，同时支持 ECG 数据在板存储。

系统做的事情：

- **实时采集**：500Hz（clean / noisy / filtered 三通道），信号源可选软件模拟发生器、真实 AFE 模拟前端或 MIT-BIH 回放
- **数字滤波**：双级梳状（50/100Hz 陷零）→ 高通 → 低通 40Hz 级联
- **心率检测**：能量包络 QRS 检测（x² + 8-25Hz 带通 + MWI + 形态学门 v6；LUDB 金标准验证 F1 0.868）
- **AI 逐拍异常检测**：exp7c 部署模型（ResNet-L INT8，2026-08-14 真实数据微调版），经 `esp-tflite-micro + ESP-NN` 在板上推理
- **心律安全与节律分析**：停搏 / 过缓 / 过速（规则）、房颤（CV + Shannon 熵）、VF/VT 检测
- **报警锁存**：AI 异常触发后 5 秒锁存，防止一闪而过
- **BLE 透传**：Nordic UART Service，手机 App 实时波形显示与报警提示
- **数据记录**：SPIFFS/LittleFS 记录器（ECGR 格式），异常触发自动录制，记录经 WiFi HTTP REST API 下载

**三个角色**（借自 [docs/MODEL_GUIDE.md](docs/MODEL_GUIDE.md)）：

| 角色 | 是什么 | 通俗比喻 |
|------|--------|---------|
| **固件** | ESP32 上跑的 C++ 代码（IDF 官方线 `experiments/esp_idf_ecg_migration/`） | 小盒子的"身体" |
| **心电模型** | TFLite Micro 部署的深度学习网络 | 小盒子的"大脑" |
| **PC 工具链** | Python 训练 / 评估 / 导出脚本（`pc_tools/`） | 训练大脑的"学校" |

**技术栈**：

| 项 | 选型 |
|----|------|
| 硬件平台 | ESP32-S3-WROOM-1-N16R8（16MB Flash / 8MB Octal PSRAM） |
| 开发框架 | **ESP-IDF（官方）**；旧 Arduino + PlatformIO 线见 `legacy_arduino/` |
| 采样率 | 500Hz（串口降频输出 100Hz） |
| AI 框架 | TensorFlow 2.x（训练）+ TFLite Micro / ESP-NN（边缘推理，INT8） |
| PC 工具 | Python（TensorFlow + matplotlib + pyserial） |
| 手机 App | Flutter（flutter_blue_plus + provider） |

---

## 快速开始

### 官方固件（ESP-IDF）

```bash
# 进入 IDF 迁移工程（需要 ESP-IDF v6 环境，cmd/PowerShell 下导出后再执行）
cd experiments/esp_idf_ecg_migration
idf.py build                 # 编译
idf.py -p <PORT> flash       # 烧录（需硬件连接）
idf.py monitor               # 串口监视
```

### 旧 Arduino 线（仅历史，已归档）

```bash
cd legacy_arduino
pio run                      # 编译（需 PlatformIO）
pio run -t upload            # 编译并上传
```

### 其它

```bash
# PC 端实时绘图（可用打包版 ECG-Plotter.exe 替代）
python pc_tools/ecg_plotter.py

# 手机端 App（Flutter）
cd ecg_app && flutter run
```

> ⚠️ 训练 AI 模型（GPU）需通过 WSL2，见下文 [GPU 训练 (WSL2)](#gpu-训练-wsl2)。

---

## 系统架构

### 采集前端（模拟链路）

```
单导联电极 (RA/LA, 贴片 ×2) ──► AD8232/自制AFE 单导联 AFE (差分输入) ──► 单路心电信号 ──► ESP32 ADC (单通道)
   电极A ─┐                                                          (12-bit, 500Hz)
          ├── IN+ / IN- 差分 ──► 放大 + 滤波 ──► 1 路输出 (RL 电极接 RLD)
   电极B ─┘
```

- **导联**是贴在身上的**电极对构成的电学测量向量**；AD8232（单导联 AFE）把两路电极（RA/LA）的差分信号放大合成 **1 路心电信号**，再进 **单个 ADC** 采样。所以是 **3 电极（RA/LA/RL）、单通道、导联 II 接法**（等效标准 ECG 的 Lead II，差分抑制共模干扰；RL 为右腿驱动）。（见 [docs/02_Hardware_Docs/afe_selection_notes.md](docs/02_Hardware_Docs/afe_selection_notes.md)）
- 信号源三选一：真实 AFE 采集（`experiments/esp_idf_ecg_migration/components/ecg_afe/`）、软件模拟发生器（`src/signal_generator/`，IDF 版在 `components/ecg_core/`）、MIT-BIH 回放

### 数字滤波链

- **梳状滤波**：双级级联滑动平均，利用 500Hz/50Hz = 10 的精确比在 50Hz/100Hz 处精确陷零，双级合计 50Hz 衰减 **-119.2dB**，群延迟 20ms（远小于 RR 间期 800ms）
- **高通 + 低通**（`components/ecg_core`）：IIR Biquad 级联，高通去基线漂移、低通 40Hz 去肌电干扰。系数在 PC 侧（Python）设计，**完整 double 精度**硬编码
- ⚠️ **高通截止频率**：固件实际为 **0.05Hz**（依据 TH §十三·8.3.1）；**训练主线仍为 0.5Hz**（TH §十三·8.7：0.05Hz 恢复 PTB 形态但伤 MIT 域，留作论文敏感性分析）

### 双核分工与数据流

```
+------------------------------------------------------------------+
|                    ESP32-S3-WROOM-1-N16R8 (双核)                 |
|                                                                  |
|  Core 1 (主循环: 采集/滤波/通信/存储)    Core 0 (AI 推理任务)    |
|  +--------------------------------+    +--------------------+    |
|  | 模拟器 / ADC / MIT-BIH 回放    |    | 环形缓冲 250 点    |    |
|  |   -> 梳状 -> HP -> LP40        |--->| Z-score 归一化     |    |
|  | 能量包络心率 v6 + SQI         |    | INT8 量化          |    |
|  |  心律安全 / AF / VF 检测       |    | TFLite Micro+NN    |    |
|  |  SPIFFS 记录器 (自动录制)      |    | 推理 (exp7c, INT8)|    |
|  +--------------------------------+    +--------------------+    |
|            |                                  |                  |
|            v                                  v                  |
|  WiFi HTTP 记录 API / 串口                BLE NUS TX (Notify)    |
|  报警锁存 5s -> abnormal 列 + 异常位图(1Hz)                      |
+------------------------------------------------------------------+
             |                                  |
             v                                  v
      pc_tools/ecg_plotter.py            ecg_app (Flutter)
      (PC 实时绘图, ECG-Plotter.exe)      (手机端心电监测)
      + AI 异常标签显示               + AI 异常高亮 + InfoPanel 警告
```

- Core 1 主循环：采样调度（2ms 间隔）、滤波链、心率检测、报警锁存、串口输出、ECG 记录
- Core 0 推理任务：等待信号量 → 取 250 点窗口 → 归一化 → 推理 → 结果队列（深度 8），主循环非阻塞读取
- 推理间隔 **AI_STRIDE=250（1Hz）**（IDF 线用 ESP-NN 后可显著降低单次推理耗时）

**串口 / BLE CSV 数据格式（每行）**：

```
<clean>,<noisy>,<filtered>,<bpm>,<true_bpm>,<sqi>,<motion>,<abnormal_flag>,<confidence>
```

| 列 | 字段 | 说明 |
|----|------|------|
| 1 | clean | 纯净心电 / 去偏置 ADC 信号 (V) |
| 2 | noisy | 含噪声原始信号 (V) |
| 3 | filtered | 数字滤波后信号 (V) |
| 4 | bpm | ESP32 板上心率检测值 |
| 5 | true_bpm | 真实心率（仅模拟器 / 回放模式） |
| 6 | sqi | 信号质量指数 (0-1) |
| 7 | motion | 运动标志 (0/1) |
| 8 | abnormal_flag | AI 异常标志（0=正常, 1=异常, 报警锁存 5s） |
| 9 | confidence | AI 异常置信度 (0-1) |

---

## 项目结构

```
├── experiments/esp_idf_ecg_migration/   官方 IDF 固件线（2026-08-28 转正）
│   ├── main/                主程序（采样/滤波/心率/报警/记录/BLE/WiFi）
│   ├── components/          IDF 组件：ecg_ai / ecg_core / ecg_ble / ecg_wifi /
│   │                        ecg_storage / ecg_afe + esp-tflite-micro + esp-nn
│   └── partitions_ecg.csv   分区表
├── legacy_arduino/          旧 Arduino + PlatformIO 线（2026-09-01 归档，仅历史参考）
│   ├── src/  include/  partitions/  platformio.ini  scripts/inject_build_flags.py
│   └── components/  esp-nn/   （孤儿 vendor 副本，git 忽略，不提交）
├── pc_tools/                Python PC 工具（ecg_plotter + ecg_dl 深度学习工具包）
├── ecg_app/                 Flutter 手机 App（BLE NUS 客户端）
├── web/                     Web 前端（记录下载）
├── docs/                    论文与结果文档（权威数字见 docs/FINAL_RESULTS.md）
├── papers/                  文献 PDF 与阅读笔记
└── test/                    测试代码（ecg_recorder_format_test 等）
```

---

## 硬件平台

**ESP32-S3-WROOM-1-N16R8**：乐鑫官方模块，双核 Xtensa LX7 @240MHz，**16MB Flash + 8MB Octal PSRAM**（芯片官方上限）。

- **双核分工**：Core 1 承担采样 / 滤波 / 心率 / 通信 / 存储，Core 0 独立跑 AI 推理
- **温度保护**：内置温度传感器每秒采样（8 点滑动平均），>65°C 自动降频至 60MHz，<55°C 恢复 240MHz
- **分区表**：IDF 线 `partitions_ecg.csv`（factory app + storage）；旧 Arduino 线 `legacy_arduino/partitions/esp32s3_16m_noota_v2.csv`（ota_0 + ecgdata SPIFFS）

---

## AI 异常检测

**部署模型（2026-08-14 起）**：单拍 250 点 ResNet-L（~80K 参数）+ 因果部署链；INT8 模型实测 **167,376 B**。板上模型演进：CNN-v2 → **exp6-SGD** → **exp7c**（08-14 真实 AFE 数据微调版上板）。论文评估口径最优操作点：**beat θ≈0.35 / patient θ≈0.5**；**固件实际运行 θ=0.60 + 5 拍确认**（保守化调整，见核心配置表）。

### 板上推理管线

```
500Hz 采样 -> 梳状 + HP + LP40 因果滤波 -> 2:1 抽取 (250Hz) -> 250 点窗口
  -> Z-score 归一化 -> INT8 量化 -> TFLite Micro 推理 (Core 0) -> 反量化直取概率
```

1. **归一化**：Z-score (x − mu) / sigma
2. **INT8 量化**：x_int8 = round(x_fp32 / scale + zero_point)
3. **TFLite Micro 推理**：ResNet-L 前向传播（1D-CNN，INT8）
4. **反量化直取概率**：模型输出层自带 softmax，直接取异常类概率，**不再二次 softmax**
5. **群延迟补偿**：因果部署链群延迟 ~6 样本 @250Hz，固件将触发时刻后移（`AI_TRIGGER_OFFSET=6`），等效评估侧 δ=+6

### 关键指标速览

> 权威数字见 [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md)（表2 / 表4），每个数值可溯源至 `patient_split_eval.json` / `retrain_exp6_sgd_eval.json`。**口径标注**：表2 为患者级 60/20/20 划分 + 训练链（filtfilt）评估 + **MIT 测试未增强**；表4 为部署链口径（因果滤波 + 2:1 抽取 + 梳状）。两表口径不同，不可直接比较。

| 口径 | 模型 | MIT-AUC | MIT-R@0.5 | PTB-AUC | PTB-R@0.5 | 说明 |
|------|------|:---:|:---:|:---:|:---:|------|
| 论文主结果（患者级清洁 / 未增强测试） | exp5 | 0.9295 | 0.9264 | 0.7845 | 0.6281 | 训练链（filtfilt）口径 |
| 论文主结果（患者级清洁 / 未增强测试） | exp6 | 0.8942 | 0.9194 | **0.8232** | 0.7019 | PTB 域最强（患者级清洁版） |
| 跨域参考（无 PTB 训练） | P2A (ResNet-L) | **0.9878** | 0.9312 | 0.7502 | 0.2552 | MIT 域最佳；PTB 为跨域泛化 |
| 部署链试点（D3，δ 对齐） | exp6-SGD | 0.9122 | 0.9102 | 0.7697 | 0.7069 | 0.5Hz 因果链 + SGD 重训（表4） |

---

## 报警与心律分析

三个基于规则 / 轻量特征的检测模块，指标来源：**docs/FINAL_RESULTS.md 表9 / 表10**（T4-8 / T4-9）。

### 心律安全（`src/rhythm_safety/`，纯规则，秒级）
| 事件 | 规则 |
|------|------|
| 停搏 asystole | RR 间期 > 4s |
| 重度过缓 bradycardia | 30s 滑窗平均 HR < 40 bpm |
| 过速 tachycardia | 30s 滑窗平均 HR > 180 bpm |

### 房颤检测（`src/af_detect/`，CV + Shannon 熵）
- 特征：RR 不规则度（变异系数 CV + Shannon 熵），零训练
- **AFDB 30s 窗**：组合分数 AUC 0.9353；最优阈值（CV>0.12, 熵>1.5）Se 0.814 / Sp 0.954
- **PTB-XL 10s 短窗**：组合 AUC **0.9717**，最优阈值（CV>0.08, 熵>1.2）Se 0.845 / Sp 0.955

### VF/VT 检测（`src/vf_detect/`，5s 窗 DSP 特征 + LR + 2 窗确认）
| 测试 | 口径 | 指标 | v2 结果（当前固件） |
|------|------|------|:---:|
| VFDB 留出（7 记录 789 VF 窗） | VF 窗 | Se | **0.9848** |
| MIT-BIH 正常对照（3117 窗） | 正常窗 | Sp | **0.8877** |
| CUDB 独立（35 条全 VF，6601 窗） | 窗级 / 2 窗确认 | Se | 0.9212 / 0.9032 |

结论（v2, 2026-08-16）：**Se 0.985 / Sp 0.888**（验收 Se≥95% / Sp≥83% 均达标）。详见 FINAL_RESULTS.md 表10。

### 报警锁存
AI / 规则报警触发后，CSV 的 `abnormal_flag` 保持 1 共 **5 秒**（`s_alarmHold = 500` @100Hz 输出），防止一闪而过；该锁存值同时作为记录器的 1Hz 异常位图来源。

---

## 数据记录与回放

### MIT-BIH 回放模式（`src/signal_generator/ecg_replay.cpp`）
无需硬件即可验证全链路：串口发送 **m / n / e** 切换播放段。

| 命令 | 播放内容 |
|------|---------|
| m | 循环切换播放段 |
| n | 段 0：MIT-BIH 100（窦性心律正常段） |
| e | 段 1：MIT-BIH 106 @90-135s（VEB 室早密集异常段） |

### 记录器（ECGR 格式）
- **ECGR 格式**（`include/storage/ecg_recorder_format.h`，固件 / PC 解码 / 云 mock 三端共用）：32B 头部（magic "ECGR" + version + flags + sampleRate + startUnix + durationSec + totalSamples + abnormalSec，全小端）+ int16 样本流 + **异常位图 1 byte/秒**；idx 行格式 `<startUnix>,<dur>,<samples>,<abnSec>,<sizeBytes>`
- **崩溃安全**：启动即写 totalSamples=0 头部；STOP 时重写最终字段；挂载扫描删除损坏 .ecgr
- **自动录制**：异常上升沿自动开始，连续 5 个正常秒自动停止（IDF 线 2026-09-01 已接线 `ecgRecorderPushSample` / `ecgRecorderSetSecondAbnormal` 并启用 auto-record）
- **下载**：IDF 线 WiFi HTTP REST API —— `GET /api/records`(JSON 列表)、`GET /api/records/{id}/meta`、`GET /api/records/{id}/data`(原始 .ecgr)、`DELETE /api/records/{id}`

> 📌 **遗留**：旧 Arduino 线含完整 BLE/串口 `REC_*` 命令通道；IDF 官方线当前以自动录制为主，BLE/串口命令通道尚未完全对齐（2026-09-01 状态）。

---

## PC 端工具

### ecg_plotter.py
实时三通道波形显示（绿 = 纯净，红 = 带噪，蓝 = 滤波后），支持 AI 异常标签（解析 abnormal_flag + confidence），串口命令按键透传。已打包 **ECG-Plotter.exe**。

**快捷键**：`1/2/3` 切换通道、`空格` 暂停/恢复、`R` 复位、`Q` 退出。

### ecg_dl/ 深度学习工具包
核心训练与推理工具：

| 脚本 | 功能 |
|------|------|
| `train.py` | 主训练流程（FocalLoss + 数据增强） |
| `train_kd.py` | 知识蒸馏训练（a070_t1 心梗筛查器） |
| `evaluate.py` | H5 / TFLite 精度对比 |
| `export_*.py` | INT8 量化导出 + C 数组 |
| `finetune_exp7c_v4.py` | exp7c_v4 多域平衡后训练（患者级无泄漏 + 真实 AFE 留出） |
| `eval_exp7c_v4.py` | exp7c_v4 联合验收 |

数据预处理：`data/preprocess.py`（MIT-BIH）、`preprocess_incart.py`、`preprocess_ptb.py`、`preprocess_ptbxl.py`、`preprocess_svdb.py`。模型定义：`models/resnet_lite_1d.py`、`models/cnn_1d.py`。

### GPU 训练 (WSL2)
Windows TensorFlow ≥2.11 原生不支持 GPU，需通过 WSL2：

```bash
wsl --install -d Ubuntu-24.04
wsl
# WSL 内安装 tensorflow[and-cuda] + 依赖，参见 docs/03_Software_Docs/AGENTS.md
cd /mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl
python3 train.py --epochs 200 --batch-size 128
```

> ⚠️ RTX 5070（compute capability 12.0a）首次运行需 JIT 编译 CUDA 内核（~30 min），后续缓存后 ~7ms/step。

---

## 手机端 App

心电监测 App（Flutter，flutter_blue_plus + provider），通过 BLE NUS 连接 "ESP32-ECG" 设备。

**功能**：BLE 扫描连接、三通道实时波形绘制（CustomPainter）、BPM 心率 + SQI、**AI 异常高亮**（波形变红 + InfoPanel 警告）、报警弹窗/提示音/历史、记录列表与回放 + 云端记录 API 骨架、速度/幅度控制。

**BLE 服务 UUID**：

| 服务 / 特征值 | UUID |
|-------------|------|
| NUS Service | `6E400001-B5A3-F393-E0A9-E50E24DCCA9E` |
| TX (Notify) | `6E400002-B5A3-F393-E0A9-E50E24DCCA9E` |
| RX (Write) | `6E400003-B5A3-F393-E0A9-E50E24DCCA9E` |

---

## 文档导航

**文档层级：README（总览入口）→ ROADMAP（决策视图）→ TUNING_HISTORY（证据日志）→ FINAL_RESULTS（权威数字）→ MODEL_GUIDE（通俗故事）**。

| 文档 | 定位 |
|------|------|
| [docs/01_Project_Overview/ROADMAP.md](docs/01_Project_Overview/ROADMAP.md) | 战略决策视图：D1-D14 决策清单 + 模型演进表 |
| [docs/03_Software_Docs/TUNING_HISTORY.md](docs/03_Software_Docs/TUNING_HISTORY.md) | 实验证据日志（持续追加） |
| [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md) | **论文权威数字源** |
| [docs/MODEL_GUIDE.md](docs/MODEL_GUIDE.md) | 通俗故事版导读 |
| [docs/03_Software_Docs/IDF_MIGRATION_ASSESSMENT.md](docs/03_Software_Docs/IDF_MIGRATION_ASSESSMENT.md) | 固件迁移评估与进度 |
| [pc_tools/ecg_dl/README.md](pc_tools/ecg_dl/README.md) | 深度学习工具包分类目录 |

---

## 核心配置

> 以下数值与源码/最新实验核对（2026-08-25 状态）。AI 常量见 `components/ecg_ai/include/ecg_ai.h`；固件运行参数见 `main/main.cc`。

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 500 Hz | esp_timer 500Hz 硬件节拍；串口每 5 帧输出 1 次 = 100Hz |
| 串口波特率 | 460800 | 板载 USB CDC 下波特率实际无效，保持一致性 |
| CPU 频率 | 240 MHz | 过温（>65°C）降频至 60MHz，<55°C 恢复 |
| BLE 设备名 | ESP32-ECG | - |
| 滤波链 | 双级梳状 → HP → LP 40Hz | 50Hz 衰减 -119.2dB、群延迟 20ms；HP 固件实际 0.05Hz（训练主线 0.5Hz） |
| 心率算法 | 能量包络 v6 | x² 能量包络 + 8-25Hz 带通 + 40 样本 MWI + millis RR + 形态学门；LUDB：Se 96.40% / PPV 78.87% / F1 0.868 / BPM MAE 4.16 |
| AI 模型 | exp7c（ResNet-L ~80K 参数） | TFLite Micro INT8，167,376 B；2026-08-14 上板 |
| AI 输入窗口 | 250 点 @ 250Hz = 1.0s | 固件 2:1 抽取（500→250Hz） |
| AI 推理间隔 | AI_STRIDE=250（1s） | IDF 线用 ESP-NN 可降至 ~49ms/次 |
| AI 判定阈值 | θ = 0.60 | **固件实际值**（TH §40）；论文口径 beat θ≈0.35 / patient θ≈0.5（D13）不可混用 |
| 多拍确认 | 5 拍 | MULTI_BEAT_CONFIRM=5 |
| 报警锁存 | 5 秒 | s_alarmHold=500 @100Hz 输出；同步写入记录器异常位图 |

---

## 开发状态与路线

**当前状态（2026-09-01）**：

- ✅ **正式固件 = ESP-IDF 迁移工程**（2026-08-28 转正，`experiments/esp_idf_ecg_migration`，exp7c INT8 + ESP-NN；2026-09-01 补齐录制链与 WiFi 记录 API）。旧 **Arduino + PlatformIO** 线已归档至 `legacy_arduino/`。
- ✅ **exp7c_v4 后训练完成并通过 PC 侧联合验收**：真实 AFE 留出抑制显著、公共库事件级能力保持、患者级无泄漏；当前为 float32 H5 候选，**尚未导出 INT8、未上板**。
- 🚧 **IDF 迁移仍有收尾**：ADC/AFE 接线细节、BLE/串口 REC_* 命令通道、完整真机联调尚未全部完成。
- 📝 **论文与审计**：跨架构部署链失配、AAMI 矩阵、QAT/后训练、泄漏审计均已留档；权威数字仍以 `docs/FINAL_RESULTS.md` 为准。

**模型侧（ROADMAP Phase 4）**：板上单模型 = exp7c；关卡 + 双专家部署为现役方案（KD a070_t1 INT8 已导出）；exp7c_v4 是当前最接近替换候选的 PC 模型，但还需 INT8 导出、PC↔板端一致性、长时真实 AFE 验证。

**已知遗留**：exp7c_v4 尚未导出 INT8、未做板端验证；IDF 线 ADC/AFE 与完整真机验收未完成；回放 100 段 bpm 偏低、VF/VT 在模拟/回放段误报等旧问题仍待硬件阶段复核。

---

*本文档为项目总览入口。指标溯源规范（项目铁律）：文中所有评估数字均来自 `docs/FINAL_RESULTS.md` 或 `TUNING_HISTORY.md` 原文，未做任何修改或"优化"；疑问数字宁可省略，不可猜测。*
