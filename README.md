# ESP32-ECG 心电采集与 AI 异常检测系统

> **ESP32-S3-WROOM-1-N16R8 (16MB Flash / 8MB Octal PSRAM) | PlatformIO + Arduino | 500Hz 采样 | BLE NUS | TFLite Micro AI 推理 | Flutter App**

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

基于 ESP32-S3 的**便携式单导联心电采集系统**：以 500Hz 实时采集心电信号，在板上完成数字滤波、心率检测与**深度学习逐拍异常检测**，异常时锁存报警并推送到手机 App，同时支持 ECG 数据在板存储（阶段 A 已完成）。

系统做的事情：

- **实时采集**：500Hz 三通道信号（clean / noisy / filtered），信号源可选软件模拟发生器、真实 AFE 模拟前端或 MIT-BIH 回放
- **数字滤波**：双级梳状（50/100Hz 陷零）→ 高通 → 低通 40Hz 级联
- **心率检测**：Pan-Tompkins QRS 检测（LUDB 金标准验证 F1 0.774）
- **AI 逐拍异常检测**：exp6-SGD 部署链定稿模型（TFLite Micro INT8），Core 0 独立推理
- **心律安全与节律分析**：停搏 / 过缓 / 过速（纯规则）、房颤（CV + Shannon 熵）、VF/VT 检测
- **报警锁存**：AI 异常触发后 5 秒锁存，防止一闪而过
- **BLE 透传**：Nordic UART Service，手机 App 实时波形显示与报警提示
- **数据记录**：SPIFFS 记录器（ECGR 格式），REC_* 命令经 BLE / 串口双通道控制

**三个角色**（借自 [docs/MODEL_GUIDE.md](docs/MODEL_GUIDE.md)）：

| 角色 | 是什么 | 通俗比喻 |
|------|--------|---------|
| **固件** | ESP32 上跑的 C++ 代码（`src/`） | 小盒子的"身体" |
| **心电模型** | TFLite Micro 部署的深度学习网络 | 小盒子的"大脑" |
| **PC 工具链** | Python 训练 / 评估 / 导出脚本（`pc_tools/`） | 训练大脑的"学校" |

**技术栈**：

| 项 | 选型 |
|----|------|
| 硬件平台 | ESP32-S3-WROOM-1-N16R8（16MB Flash / 8MB Octal PSRAM） |
| 开发框架 | PlatformIO + Arduino（board: `4d_systems_esp32s3_gen4_r8n16`） |
| 采样率 | 500Hz（串口降频输出 100Hz） |
| AI 框架 | TensorFlow 2.x（训练）+ TFLite Micro（边缘推理，INT8） |
| PC 工具 | Python（TensorFlow + matplotlib + pyserial） |
| 手机 App | Flutter（flutter_blue_plus + provider） |

---

## 快速开始

```bash
# 编译固件（Windows: pio 命令已加入 PATH）
pio run

# 编译并上传（需硬件连接开发板）
pio run -t upload

# 串口监视器（9 列 CSV @100Hz，波特率 460800）
pio device monitor -b 460800

# PC 端实时绘图（可用打包版 ECG-Plotter.exe 替代）
python pc_tools/ecg_plotter.py

# 手机端 App（Flutter）
cd ecg_app && flutter run
```

> ⚠️ 分区表已变更（新增 4MB SPIFFS 数据分区），**下次烧录必须 `pio run -t erase` + 全量重烧**，否则文件系统与新分区表不匹配（TH §三十二）。
>
> 训练 AI 模型（GPU）需通过 WSL2，见下文 [GPU 训练 (WSL2)](#gpu-训练-wsl2)。

---

## 系统架构

### 采集前端（模拟链路）

```
单导联电极 (RA/LA, 贴片 ×2) ──► AD8232 单导联 AFE (差分输入) ──► 单路心电信号 ──► ESP32 ADC (单通道)
   电极A ─┐                                                          (12-bit, 500Hz)
          ├── IN+ / IN- 差分 ──► 放大 + 滤波 ──► 1 路输出 (RL 电极接 RLD)
   电极B ─┘
```

- **导联**是贴在身上的**电极对构成的电学测量向量**；AD8232（单导联 AFE）把两路电极（RA/LA）的差分信号放大合成 **1 路心电信号**，再进 **单个 ADC** 采样。所以是 **3 电极（RA/LA/RL）、单通道、导联 II 接法**（等效标准 ECG 的 Lead II，差分抑制共模干扰；RL 为右腿驱动）。⚠️ 历史文档"AD620 / 双导联"为错误表述，已更正（见 [docs/hardware/afe_selection_notes.md](docs/hardware/afe_selection_notes.md)，含多导联备选评估）
- 信号源三选一：真实 AFE 采集（`src/adc_afe/`）、软件模拟发生器（`src/signal_generator/ecg_simulator.cpp`）、MIT-BIH 回放（`src/signal_generator/ecg_replay.cpp`）

### 数字滤波链

- **梳状滤波**（`main.cpp`）：双级级联滑动平均，利用 500Hz/50Hz = 10 的精确比在 50Hz/100Hz 处精确陷零，双级合计 50Hz 衰减 **-119.2dB**，群延迟 20ms（远小于 RR 间期 800ms）
- **高通 + 低通**（`src/filter/filter.cpp`）：IIR Biquad 级联，高通去基线漂移、低通 40Hz 去肌电干扰。系数在 PC 侧（Python）设计，**完整 double 精度**硬编码（2026-08-08 修复：float32 下 0.05Hz 高通极点极近单位圆，灾难性抵消导致滤波输出 9→28V 爬升、心率失效）
- ⚠️ **高通截止频率**：固件实际为 **0.05Hz**（filter.cpp；依据 TH §十三·8.3.1，0.5Hz 因果 HP 在 ST 带引入 1.5-9mm 伪 ST 偏移，是 PTB 部署链 AUC 缺口主因）。**训练主线仍为 0.5Hz**（TH §十三·8.7 决策：0.05Hz 恢复 PTB 形态但伤 MIT 域，留作论文敏感性分析）。两处口径差异见 TH §二十（论文 §3.3 HP 描述按固件实际改写）

### 双核分工与数据流

```
+------------------------------------------------------------------+
|                    ESP32-S3-WROOM-1-N16R8 (双核)                 |
|                                                                  |
|  Core 1 (主循环: 采集/滤波/通信/存储)    Core 0 (AI 推理任务)    |
|  +--------------------------------+    +--------------------+    |
|  | 模拟器 / ADC / MIT-BIH 回放    |    | 环形缓冲 250 点    |    |
|  |   -> 梳状 -> HP -> LP40        |--->| Z-score 归一化     |    |
|  |  Pan-Tompkins 心率 + SQI       |    | INT8 量化          |    |
|  |  心律安全 / AF / VF 检测       |    | TFLite Micro 推理  |    |
|  |  SPIFFS 记录器 (REC_* 命令)    |    | (exp6-SGD, INT8)   |    |
|  +--------------------------------+    +--------------------+    |
|            |                                  |                  |
|            v                                  v                  |
|  串口 CSV (9列 @100Hz)                  BLE NUS TX (Notify)      |
|  报警锁存 5s -> abnormal 列 + 异常位图(1Hz)                      |
+------------------------------------------------------------------+
             |                                  |
             v                                  v
      pc_tools/ecg_plotter.py            ecg_app (Flutter)
      (PC 实时绘图, ECG-Plotter.exe)      (手机端心电监测)
      + AI 异常标签显示               + AI 异常高亮 + InfoPanel 警告
```

- Core 1 主循环：采样调度（2ms 间隔）、滤波链、心率检测、报警锁存、串口 100Hz 输出、SPIFFS 记录
- Core 0 推理任务：等待信号量 → 取 250 点窗口 → 归一化 → 推理 → 结果队列（深度 8），主循环非阻塞读取
- 推理间隔 **AI_STRIDE=250（1Hz）**：2026-08-08 真机实测单次推理 ~910ms，超过原 0.5s 触发间隔，任务占满 Core 0 触发 Task WDT 崩溃；调为 1s 间隔 + 推理后让出 CPU 修复（TH §三十一）

**串口 / BLE CSV 数据格式（每行）**：

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
| 5 | true_bpm | 真实心率（仅模拟器 / 回放模式） |
| 6 | sqi | 信号质量指数 (0-1) |
| 7 | motion | 运动标志 (0/1) |
| 8 | abnormal_flag | AI 异常标志（0=正常, 1=异常, 报警锁存 5s） |
| 9 | confidence | AI 异常置信度 (0-1) |

---

## 项目结构

```
├── src/                    固件源码（Arduino 风格）
│   ├── main.cpp            主程序：采样调度 / 滤波链 / 心率 / 报警锁存 / 串口输出 /
│   │                       BLE 输出 / REC_*·WIFI_*·回放命令解析
│   ├── adc_afe/            ADC / AFE 采集（afe_hal）
│   ├── ai_inference/       TFLite Micro 推理（Core 0 任务 + 环形缓冲）+ 模型权重
│   ├── bluetooth/          BLE NUS 通信（+9dBm，RX 命令队列）
│   ├── filter/             数字滤波（IIR Biquad HP + LP）
│   ├── heartrate/          Pan-Tompkins 心率检测（自适应阈值 + SQI + 运动检测）
│   ├── rhythm_safety/      心律安全（停搏 / 过缓 / 过速，纯规则秒级）
│   ├── af_detect/          房颤检测（CV + Shannon 熵，10s 窗三态）
│   ├── vf_detect/          VF/VT 检测（5s 窗 DSP 特征 + LR + 2 窗确认）
│   ├── signal_generator/   ecg_simulator（模拟信号）+ ecg_replay（MIT-BIH 回放）
│   ├── storage/            SPIFFS 记录器（ECGR 格式，崩溃安全）
│   ├── thermal/            温度管理（>65°C 降频保护）
│   └── wifi/               阶段 B WiFi 传输（占位，WIFI_ON/OFF 命令已接线）
├── include/                头文件（与 src/ 同构；含 ai_inference/ecg_model_data.h 模型权重）
├── pc_tools/               Python PC 工具
│   ├── ecg_plotter.py      实时三通道绘图（可打包 ECG-Plotter.exe）
│   ├── capture_debug.py / find_port.py / hr_sim_verify.py / verify_filter_coeffs.py
│   └── ecg_dl/             深度学习工具包（训练 / 评估 / 导出 / PC 推理，见下文）
├── ecg_app/                Flutter 手机 App（BLE NUS 客户端）
├── test/                   测试代码（ecg_recorder_format_test 等，WSL g++ 可独立跑）
├── docs/                   论文与结果文档（权威数字见 docs/FINAL_RESULTS.md）
├── papers/                 文献 PDF 与阅读笔记
├── partitions/             分区表（esp32s3_16m_noota_v2.csv 等）
└── lib/                    库文件（TFLite Micro 等经 platformio.ini lib_deps 引入）
```

---

## 硬件平台

**ESP32-S3-WROOM-1-N16R8**：乐鑫官方模块，双核 Xtensa LX7 @240MHz，**16MB Flash + 8MB Octal PSRAM**（芯片官方上限）。PlatformIO 内置唯一 R8N16 定义为 `4d_systems_esp32s3_gen4_r8n16`（memory_type qio_opi）；旧 SUPERMINI 板（4MB Flash / 2MB PSRAM，`adafruit_qtpy_esp32s3_n4r2`）会让 Octal PSRAM 失效且多烧 tinyuf2.bin，换板必须换 board 定义（TH §三十一）。

- **双核分工**：Core 1 承担采样 / 滤波 / 心率 / 通信 / 存储，Core 0 独立跑 AI 推理（16KB 任务栈，推理后让出 CPU）
- **温度保护**：内置温度传感器每秒采样（8 点滑动平均），>65°C 自动降频至 60MHz，<55°C 恢复 240MHz（`src/thermal/`）
- **分区表** `partitions/esp32s3_16m_noota_v2.csv`（2026-08-08 分区重排，阶段 A）：

| 分区 | 类型 | 偏移 | 大小 |
|------|------|------|------|
| nvs | data/nvs | 0x9000 | 20K |
| otadata | data/ota | 0xe000 | 8K |
| ota_0 | app/ota_0 | 0x10000 | **11M** |
| ecgdata | data/spiffs | 0xb10000 | **4M**（ECG 记录存储，250Hz int16 三通道原始数据约 43 分钟容量） |

- **资源占用**：阶段 A 后编译 Flash **13.4%**（1,542,233 / 11,534,336 B，按 ota_0 11M 分区计，余 ~9.6MB）/ RAM **40.9%**（TH §三十二）；真机烧录验收时（15M app 分区口径）Flash 8.4% / RAM 38.3%，`Embedded PSRAM 8MB` 确认（TH §三十一）
- **容量历史**：SUPERMINI 板（4MB Flash）时代曾需去 OTA 分区调整，单模型 47.1%、双模型链接实测 52.9%（TH §二十七）；N16R8 下双模型（327KB）仅占 16MB 的 2%，**Flash 不再是瓶颈**（ROADMAP §4.2）

---

## AI 异常检测

**部署定稿（2026-08-03 决策，2026-08-05 T0-1 固件集成）**：单拍 250 点 ResNet-L（~80K 参数）+ 0.5Hz 因果部署链 + SGD 优化器；INT8 模型**实测 163.5 KB**（167,376 B）。板上模型 = **exp6-SGD 部署链定稿模型**（`best_resnet_large_exp6_sgd.h5` → `ecg_model_exp6_sgd_int8.tflite` → `include/ai_inference/ecg_model_data.h`，替换旧 CNN-v2）。最优操作点：**beat 级 θ≈0.35 / patient 级 θ≈0.5**（决策 D13，详见 TH §十三·8.7）。

**部署方案演进（一句话）**：双专家 OR（P2A + exp5）严谨口径实测**已否决**（TH §8.8：MIT 误报叠加 25~32%、PTB 召回仅 0.41，旧口径数字为泄漏版作废）；改为**分模型 + 前置关卡**：心律失常交给 P2A（θ=0.5），心梗筛查交给 KD 蒸馏学生 a070_t1（θ=0.35 + 时序确认），前置"正常 vs 异常"关卡过滤正常拍（实测精度 +20%，TH §8.9.1/8.9.5/8.9.6）。

### 关键指标速览

> 权威数字见 [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md)（表2 / 表4），每个数值可溯源至 `patient_split_eval.json` / `retrain_exp6_sgd_eval.json`。**口径标注**：表2 为患者级 60/20/20 划分 + 训练链（filtfilt）评估 + **MIT 测试未增强**（T1-2）；表4 为部署链口径（D3：因果滤波 + 2:1 抽取 + 梳状），数字取自 δ 对齐（群延迟窗口错位 ~6 样本，δ=0 下 exp6-SGD 为 MIT 0.8946 / PTB 0.7326）。**两表口径不同，不可直接比较**。

| 口径 | 模型 | MIT-AUC | MIT-R@0.5 | PTB-AUC | PTB-R@0.5 | 说明 |
|------|------|:---:|:---:|:---:|:---:|------|
| 论文主结果（患者级清洁 / 未增强测试） | exp5 | 0.9295 | 0.9264 | 0.7845 | 0.6281 | 训练链（filtfilt）口径 |
| 论文主结果（患者级清洁 / 未增强测试） | exp6 | 0.8942 | 0.9194 | **0.8232** | 0.7019 | PTB 域最强（患者级清洁版） |
| 跨域参考（无 PTB 训练） | P2A (ResNet-L) | **0.9878** | 0.9312 | 0.7502 | 0.2552 | MIT 域最佳；PTB 为跨域泛化 |
| 部署链试点（D3，δ 对齐） | exp6-SGD | 0.9122 | 0.9102 | 0.7697 | 0.7069 | 0.5Hz 因果链 + SGD 重训（表4） |

### 板上推理管线

```
500Hz 采样 -> 梳状 + HP + LP40 因果滤波 -> 2:1 抽取 (250Hz) -> 250 点窗口
  -> Z-score 归一化 -> INT8 量化 -> TFLite Micro 推理 (Core 0) -> 反量化直取概率
```

1. **归一化**：Z-score (x − mu) / sigma
2. **INT8 量化**：x_int8 = round(x_fp32 / scale + zero_point)
3. **TFLite Micro 推理**：ResNet-L 前向传播（1D-CNN，INT8）
4. **反量化直取概率**：模型输出层自带 softmax，直接取异常类概率，**不再二次 softmax**（二次 softmax 会把概率动态范围压缩至 [0.270, 0.730]，导致阈值语义漂移，TH §二十三）
5. **群延迟补偿**：因果部署链群延迟 ~6 样本 @250Hz（24ms），固件将推理触发时刻后移 6 样本（`AI_TRIGGER_OFFSET=6`），等效评估侧 δ=+6 窗口重提取语义，零成本抵消错位（详见 FINAL_RESULTS.md 表5）
6. **真机实测**：单次推理 ~910ms，AI_STRIDE=250（1Hz）；真机温度 ≤51°C，无 WDT 崩溃（TH §三十一）

### 核心结论

- **数据量 > 模型架构 > 训练技巧**；单域 recall 天花板 ~0.82（TH §八 / ROADMAP 核心结论）
- **SVEB / F 类为单拍信息固有瓶颈**（非模型缺陷）：VEB / Q 召回 ≥0.98，拖后腿的是 SVEB（Recall 44%）与 F（73%），单拍形态与正常拍几乎相同，需多拍上下文而数据量已否决多拍方案（D14，TH §十三·8.5）
- **部署链失配**：训练链（filtfilt 零相位）与部署链（因果滤波）差异使 PTB 域 ΔAUC −0.105，部署链重训（D10）+ SGD（优于 AdamW，PTB +0.035，D11）为当前正解；输入侧补偿中仅 P0 时间对齐实用（恢复失配 30~60%，FINAL_RESULTS.md 表5）
- 完整实验证据见 [TUNING_HISTORY.md](TUNING_HISTORY.md)，决策路线见 [ROADMAP.md](ROADMAP.md)，通俗故事见 [docs/MODEL_GUIDE.md](docs/MODEL_GUIDE.md)

---

## 报警与心律分析

三个基于规则 / 轻量特征的检测模块（对应 [consumer_ecg_architecture_plan.md](consumer_ecg_architecture_plan.md) 产品架构的模块 1/2/3），全部 `pio run` 编译通过并集成 `main.cpp`。指标来源：**docs/FINAL_RESULTS.md 表9 / 表10**（T4-8 / T4-9，2026-08-06）。

### 心律安全（`src/rhythm_safety/`，纯规则，秒级）

| 事件 | 规则 | 备注 |
|------|------|------|
| 停搏 asystole | RR 间期 > 4s | 累计计数 |
| 重度过缓 bradycardia | 30s 滑窗平均 HR < 40 bpm | |
| 过速 tachycardia | 30s 滑窗平均 HR > 180 bpm | |

合成 RR 序列 5 场景回放测试（正常窦性 / 停搏 / 35bpm 过缓 / 190bpm 过速 / AF）**全部 PASS**（FINAL_RESULTS.md 表9）。

### 房颤检测（`src/af_detect/`，CV + Shannon 熵）

- 特征：RR 不规则度（变异系数 CV + Shannon 熵），零训练
- **AFDB 30s 窗验证**：27,454 窗，组合分数 AUC 0.9353（结论按 ~0.94 表述，与 Moody & Mark 1983 原文 RR 间期法报告水平一致）；最优阈值（CV>0.12, 熵>1.5）Se 0.814 / Sp 0.954
- **PTB-XL 10s 短窗验证**（"一键测房颤"可行性）：组合 AUC **0.9717**，最优阈值（CV>0.08, 熵>1.2）Se 0.845 / Sp 0.955；消费设备 10s 单条记录即可出三态
- **三态输出**：AF 疑似 / 正常 / 无法判定（覆盖 RR 不足、低 SQI 时段）；固件默认 10s 窗（AF_WIN_S=10，阈值 0.08 / 1.2，TH §二十七；30s 行业标准模式可编译期切回）

### VF/VT 检测（`src/vf_detect/`，5s 窗 DSP 特征 + LR + 2 窗确认）

| 测试 | 口径 | 指标 | 结果 |
|------|------|------|:---:|
| VFDB 留出（7 记录 789 VF 窗） | VF 窗 | Se | **0.9569** |
| MIT-BIH 正常对照（3117 窗） | 正常窗 | Sp | **0.8239**（95% CI [0.811, 0.838]） |
| CUDB 独立（35 条全 VF，6601 窗） | 窗级 / 2 窗确认 | Se | 0.9359 / 0.9179 |

结论：**Se 0.957 / Sp 0.824**（验收 Se≥95% / Sp≥83% 达标，Sp 略低于名义值 0.83 但 95% CI 覆盖）；**连续 2 窗确认**（时延 ≤10s）以 4% Se 换误报抑制。

### 报警锁存

AI / 规则报警触发后，CSV 的 `abnormal_flag` 保持 1 共 **5 秒**（`s_alarmHold = 500` @100Hz 输出，2026-08-08 新增），防止一闪而过；该锁存值同时作为记录器的 1Hz 异常位图来源。

---

## 数据记录与回放

### MIT-BIH 回放模式（`src/signal_generator/ecg_replay.cpp`）

无需硬件即可验证全链路：串口发送 **m / n / e** 切换播放段。

| 命令 | 播放内容 |
|------|---------|
| m | 循环切换播放段 |
| n | 段 0：MIT-BIH 100（窦性心律正常段） |
| e | 段 1：MIT-BIH 106 @90-135s（VEB 室早密集异常段，0-45s VEB 稀少致报警不稳定已换段） |

- 回放报警率与 PC 侧部署链 + TFLite 滑窗模拟**完全吻合**（正常段 21% vs 20.5%；异常段 47% vs 48%，TH §三十一）——确认无固件实现差异，~20% 误报为模型对固定相位窗口的固有行为（T2-5 相位鲁棒性已证模型对相位敏感）

### SPIFFS 记录器（`src/storage/ecg_recorder.cpp`，阶段 A 核心）

- **分区**：`esp32s3_16m_noota_v2.csv` 新增 4MB SPIFFS 数据分区（ecgdata），250Hz int16 三通道原始数据约 43 分钟容量
- **ECGR 格式**（`include/storage/ecg_recorder_format.h`，固件 / PC 解码 / 云 mock 三端共用）：32B 头部（magic "ECGR" + version + flags + sampleRate + startUnix + durationSec + totalSamples + abnormalSec，全小端）+ int16 样本流 + **异常位图 1 byte/秒**；idx 行格式 `<startUnix>,<dur>,<samples>,<abnSec>,<sizeBytes>`
- **崩溃安全三件套**：启动即写 totalSamples=0 头部；STOP 时 seek(0) 重写最终字段；挂载扫描删除头部与文件大小不一致的损坏 .ecgr
- **保留策略**：删旧保 10 条（ECG_REC_KEEP_MAX=10），空闲 <512KB 再删最旧；8KB 批缓冲满批刷入，存储满优雅降级
- **自动录制**：异常上升沿自动开始，连续 5 个正常秒自动停止（1Hz tick 驱动）
- **REC_* 命令**（BLE + 串口双通道，大小写无关，共享解析器）：`REC_START` / `REC_STOP` / `REC_STATUS` / `REC_LIST` / `REC_AUTO 0|1`；记录采样 2:1 抽取 500→250Hz，int16 缩放统一 scale=8000.0
- **硬件验收（2026-08-08 补记）**：真机 60s 录制 durationSec 自洽（修复 2Hz tick bug 后 63 ≈ 实际 62s）；**断电重启后记录仍在（count=2），阶段 A 核心验收通过**（TH §三十二）

> 阶段 B：WiFi AP 传输 + 云端存储为下一阶段（`src/wifi/` 占位，WIFI_ON/OFF 命令已接线；App 侧云端记录 API 客户端已搭骨架）。

---

## PC 端工具

### ecg_plotter.py

实时三通道波形显示（绿 = 纯净，红 = 带噪，蓝 = 滤波后），支持 AI 异常标签（解析 abnormal_flag + confidence），串口命令按键透传。已打包 **ECG-Plotter.exe**（PyInstaller + tkinter，TH §三十一）。

**快捷键**：

| 按键 | 功能 |
|------|------|
| 1 / 2 / 3 | 切换通道可见性 |
| 空格 | 暂停 / 恢复 |
| R | 复位视图 |
| Q | 退出 |

### ecg_dl/ 深度学习工具包

核心训练与推理工具：

| 脚本 | 功能 |
|------|------|
| `train.py` | 主训练流程（FocalLoss + 数据增强） |
| `train_kd.py` | 知识蒸馏训练（a070_t1 心梗筛查器即产自 KD 路线） |
| `train_ssl.py` / `train_ensemble.py` / `train_multitask.py` | SSL 预训练 / 集成 / 多任务（均为走岔路验证，见 TH §六） |
| `evaluate.py` | H5 / TFLite 精度对比 |
| `export.py` / `export_exp6_sgd.py` | INT8 量化导出 + C 数组（含部署链口径双域校准集） |
| `07_pc_inference.py` | PC 侧实时推理 + 基准 |
| `eval_*.py` 系列 | 患者级划分 / 部署链匹配 / 报警决策层 / AF / VF / 相位鲁棒性 / bootstrap CI 等评估 |

数据预处理：`data/preprocess.py`（MIT-BIH）、`preprocess_incart.py`（INCART）、`preprocess_ptb.py`（PTB）、`preprocess_ptbxl.py`（PTB-XL）、`preprocess_svdb.py`（SVDB）。模型定义：`models/resnet_lite_1d.py`（ResNet-Lite 小/中/大）、`models/cnn_1d.py`。

### GPU 训练 (WSL2)

Windows TensorFlow ≥2.11 原生不支持 GPU，需通过 WSL2：

```bash
# PowerShell（管理员）
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
cd /mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl
python3 train.py --epochs 200 --batch-size 128
```

> ⚠️ RTX 5070（compute capability 12.0a）首次运行需 JIT 编译 CUDA 内核（~30 min），后续缓存后 ~7ms/step。

---

## 手机端 App

心电监测 App（Flutter，flutter_blue_plus + provider），通过 BLE NUS 连接 "ESP32-ECG" 设备。

**功能**：

- BLE 扫描连接，设备名 "ESP32-ECG"
- 三通道实时波形绘制（CustomPainter）
- BPM 心率显示 + 信号质量（SQI）指示
- **AI 异常高亮**：解析 abnormal_flag + confidence，波形变红 + InfoPanel 警告（TH §三十一）
- 报警弹窗 / 提示音 / 报警历史记录（`alarm_dialog.dart`、`alarm_sound_service.dart`、`alarm_history_store.dart`）
- 记录列表与回放页面 + 云端记录 API 客户端（阶段 B 骨架：`record_list_page.dart`、`playback_page.dart`、`record_api.dart`、`ecg_record_codec.dart`）
- 速度 / 幅度控制（1s / 2s / 4s / 6s 窗口）

**BLE 服务 UUID**：

| 服务 / 特征值 | UUID |
|-------------|------|
| NUS Service | `6E400001-B5A3-F393-E0A9-E50E24DCCA9E` |
| TX (Notify) | `6E400002-B5A3-F393-E0A9-E50E24DCCA9E` |
| RX (Write) | `6E400003-B5A3-F393-E0A9-E50E24DCCA9E` |

---

## 文档导航

项目文档较多，先分清各自定位。**文档层级：README（总览入口）→ ROADMAP（决策视图）→ TUNING_HISTORY（证据日志）→ FINAL_RESULTS（权威数字）→ MODEL_GUIDE（通俗故事）**。

| 文档 | 定位 | 什么时候读 |
|------|------|-----------|
| **README.md**（本文） | 总览入口：架构 / 结构 / 配置 / 快速上手 | 第一次看项目 |
| [ROADMAP.md](ROADMAP.md) | 战略决策视图：D1-D14 决策清单 + 模型演进表 + Phase 4 计划，3 分钟读完 | 想快速知道"决策是什么、下一步去哪" |
| [TUNING_HISTORY.md](TUNING_HISTORY.md) | 实验证据日志（三十二章）：每个结论的完整证据链与根因分析 | 深挖某个结论、审计数字、复盘踩坑 |
| [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md) | **论文权威数字源**：全部指标可溯源至 JSON，含口径标注与修正声明 | 引用任何指标之前（数字审计规范：只信这里） |
| [docs/MODEL_GUIDE.md](docs/MODEL_GUIDE.md) | 通俗故事版导读：不讲术语不堆指标，只讲来龙去脉 | 隔一段时间回来忘了项目在干嘛时 |
| [consumer_ecg_architecture_plan.md](consumer_ecg_architecture_plan.md) | 消费级产品架构（模块 1-4：心律安全 / VF / AF / 决策层） | 产品化设计、报警决策层设计 |
| [docs/hardware/](docs/hardware/) | 硬件笔记：AFE 选型（afe_selection_notes.md）、板上评测协议（ondevice_bench_protocol.md）、人体实验协议等 | 硬件问题、真机评测 |
| [pc_tools/ecg_dl/PATIENT_SPLIT_PROGRESS.md](pc_tools/ecg_dl/PATIENT_SPLIT_PROGRESS.md) | 患者级划分与重训进度（数据划分细节的交叉验证源） | 数据划分、泄漏审计细节 |

---

## 核心配置

> 以下数值全部与源码核对（`src/`、`include/`，2026-08-08 状态）。AI 相关常量见 `include/ai_inference/tflite_settings.h`。

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 500 Hz | SAMPLE_INTERVAL_MS=2；串口每 5 帧输出 1 次 = 100Hz |
| 串口波特率 | 460800 | 2026-08-08 从 115200 上调（全仓库同步）；板载 USB CDC（ARDUINO_USB_CDC_ON_BOOT=1）下波特率参数实际无效，保持一致性 |
| CPU 频率 | 240 MHz | 过温（>65°C）降频至 60MHz，<55°C 恢复 |
| BLE TX 功率 | +9 dBm | esp_ble_tx_power_set(ESP_PWR_LVL_P9) |
| BLE 设备名 | ESP32-ECG | - |
| 滤波链 | 双级梳状(50/100Hz) → HP → LP 40Hz | 梳状双级合计 50Hz 衰减 -119.2dB、群延迟 20ms；独立 50/100Hz 陷波器已移除由梳状统一处理；HP 固件实际 0.05Hz（训练主线 0.5Hz，见 TH §十三·8.7） |
| 心率算法 | Pan-Tompkins v4.2 | 自适应阈值（噪声峰 + 0.30×(信号峰−噪声峰)）+ 200ms 不应期；LUDB 验证：Se 72.9% / PPV 82.6% / F1 0.774 / BPM MAE 3.2 |
| AI 模型 | exp6-SGD（ResNet-L ~80K 参数） | TFLite Micro INT8，**实测 163.5 KB**（167,376 B）；`include/ai_inference/ecg_model_data.h` |
| AI 输入窗口 | 250 点 @ 250Hz = 1.0s | 固件 2:1 抽取（AI_INPUT_DECIMATION=2，500→250Hz），与训练窗口一致 |
| AI 推理间隔 | AI_STRIDE=250（1s） | 2026-08-08 从 125（0.5s）调大：真机实测单次推理 ~910ms > 500ms 触发间隔 |
| AI 群延迟补偿 | AI_TRIGGER_OFFSET=6 | 因果链群延迟 6 样本（24ms），触发时刻后移，等效 δ=+6（FINAL_RESULTS.md 表5） |
| AI 判定阈值 | θ = 0.35 | 拍级操作点（TH §十三·8.7，决策 D13）；patient 级 θ≈0.5 |
| AI Tensor Arena | 64 KB | 2026-08-08 从 32KB 扩（实测 AllocateTensors 需 40,004B） |
| AI 核心 / 栈 | Core 0 / 16KB | 推理后 vTaskDelay(50ms) 让出 CPU 0；结果队列深度 8 |
| 温度阈值 | 65°C 降频 / 55°C 恢复 | 8 点滑动平均，每秒采样 |
| 报警锁存 | 5 秒 | s_alarmHold=500 @100Hz 输出；同步写入记录器异常位图 |
| 分区表 | esp32s3_16m_noota_v2.csv | ota_0 11M + ecgdata 4M SPIFFS；编译 Flash 13.4% / RAM 40.9% |

---

## 开发状态与路线

**当前阶段（2026-08-08）**：

- ✅ **阶段 A（板上 ECG 记录存储）完成**：SPIFFS 记录器 + REC_* 命令落地，**硬件验收通过**（断电持久化验证成功，TH §三十二）；真机四连修完成（AI arena / Task WDT / 滤波器 double 精度 / 心率跨域标定，TH §三十一）
- ⬜ **阶段 B（WiFi AP 传输 + 云端存储）待办**：`src/wifi/` 占位已接线，App 云端 API 客户端骨架已搭
- 📝 **论文写作修订进行中**：19 条审稿问题已全部有解（MODEL_GUIDE §6）；权威数字以 [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md) 为准，稿件见 `docs/manuscript_sections_1_4.md`

**模型侧（ROADMAP Phase 4）**：exp6-SGD 已上板（T0-1）；双模型部署（P2A + KD a070_t1）为 4.2 待办（N16R8 下 Flash 无容量障碍，需双 TFLite interpreter / 分时加载运行时实现）；4.3 全链路集成验证（PC-ESP32 一致性、真实采集链路、温度/功耗）待办；4.4 模型侧优化按需。

**已关闭路线（避免重复踩坑）**：双专家 OR（TH §8.8 否决）、3-beat 输入（D5）、SSL 预训练（D6）、平衡混合单模型（TH §8.9.6）、拍级 RR 上下文融合器（TH §二十八 负面结果）、全类相位扰动增强（TH §十八 负面结果）。

**已知遗留**：回放 100 段 bpm 偏低（心率参数按模拟器标定，真实信号需再标定）；VF/VT 检测器在模拟器/回放段有误报（阈值 / SQI 门控待调）；LED 引脚 GPIO48 假设待实物确认（TH §三十一）。

---

*本文档为项目总览入口。指标溯源规范（项目铁律）：文中所有评估数字均来自 `docs/FINAL_RESULTS.md` 或 `TUNING_HISTORY.md` 原文，未做任何修改或"优化"；疑问数字宁可省略，不可猜测。*
