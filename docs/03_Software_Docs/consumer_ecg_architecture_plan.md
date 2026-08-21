# 消费电子双模式 ECG 系统架构计划

> 版本: v1.0 (2026-08-03) | 状态: 架构定稿，待评审
> 产品定位: **消费电子健康监测设备**（非临床筛查/诊断）
> 核心架构: **长程分析（云端）+ 短程报警（板上）双模式**，条件特定决策层分治

---

## 0. 摘要（TL;DR）

本项目产品定位为**消费电子**设备，不是临床心脏病筛查工具。因此架构必须围绕两个截然不同的任务设计：

1. **模式 A — 长程分析（云端）**: 多记录长期记录 → 云端计算/存储 → 判断用户心脏健康
   状况、给出可能诊断（如"疑似阵发性房颤，建议门诊复查"）。
2. **模式 B — 短程报警（板上实时）**: 用户陷入危重情形（心梗/重度缺血/房颤/室颤）时
   **准确检出、及时报警**（秒~分钟级）。

架构结论（本计划核心）：

- **废除"单拍二分报警"作为最终决策**——拍级二分类只作**特征层**，不再直接报警。
- **采用"条件特定决策层"分治架构**：每个危重条件配一个最擅长的检测器（心律安全逻辑 /
  VF-VT 检测器 / AF 检测 / 缺血-ST 分支），各司其职，互不干扰。
- **PTB 专家模型（KD a070_t1）定位 = 缺血/心梗分支的骨干**（单模型部署在此分支成立），
  叠加"ST 段量化测量 + 每用户心率相关基线 + 持续偏移判定"弥补其单拍形态分类的缺陷。
- **消费级报警三件套**（§8.9.8 结论 + 文献佐证）：段级/事件级确认 + 前置关卡 + 场景阈值
  与免责声明。先验塌缩（P→0.05~0.11 @ 患病率 1.5-3.4%）是贝叶斯物理下限，只能管理
  不能消除。

---

## 1. 产品定位与设计原则

### 1.1 定位

| 维度 | 本产品 | 不是 |
|---|---|---|
| 使用场景 | 日常佩戴/居家健康监测 | 医院筛查、临床诊断 |
| 用户 | 健康人群/慢病管理用户 | 住院患者 |
| 输出 | 报警提示 + 健康趋势报告 + 就医建议 | 诊断结论 |
| 法律定位 | 健康监测，非医疗器械（或按消费级 ECG 走 FDA 510(k)/CE Class IIa） | 临床决策依据 |

### 1.2 设计原则（每条都有文献/实验依据，见 §7）

1. **分治 > 单模型全能**: 危重条件形态差异大（缺血是 ST 形态、AF 是节律、VF 是波形
   崩塌），单模型二分做不到全 —— 每个条件独立检测器。
2. **特征层/决策层分离**: 拍级概率 = 特征，段级/事件级聚合 + 条件阈值 = 决策。
   单拍不可信（误报 21.6%，§8.9.9），必须上移到事件/段级确认。
3. **误报管理优先**: 消费设备误报会致用户恐慌与弃用 —— 报警阈值宁严勿松，
   配合"通知级"与"报警级"分级。
4. **诚实边界**: 不承诺检出所有危重；单导联 ST 灵敏度有文献明确上限（§7.4）；
   文档与 UI 必须免责声明。
5. **零训练优先**: 能纯逻辑/DSP 解决（停搏、过缓过速、VF 特征）不训模型；
   需模型的用现成双专家 + 微调，不重训。

---

## 2. 任务分解：双模式

### 2.1 模式 A：长程分析（云端）

- **输入**: 板上周期性上传的压缩数据（RR 间期序列 + 拍级特征 + 关键 30s 片段 + 事件日志）
- **计算**: 云端（无 INT8/内存限制，可用大模型、长序列模型）
- **输出**: 周报/月报 —— 可能诊断 + 置信度 + 趋势图 + 就医建议
- **时延**: 小时~天级，无实时要求
- **存储**: 云存储（脱敏、匿名化、用户授权）

### 2.2 模式 B：短程报警（板上实时）

- **输入**: 板上 250Hz 滤波后 ECG 流（现成采集链）
- **计算**: ESP32-S3 板上（TFLite Micro INT8 + DSP + 纯逻辑）
- **输出**: 分级报警 —— 秒级（VF/停搏）、分钟级（ST 持续偏移）、通知级（AF 疑似）
- **时延**: VF/停搏 ≤10s；ST 偏移 ≥5min 持续判定；AF 30-60s 窗口
- **关键约束**: 误报率必须极低（用户恐慌阈值）

### 2.3 双模式数据流总览

```
ESP32-S3 (板上)
┌─────────────────────────────────────────────────────────┐
│ 采集链 (现成): 500Hz→250Hz → HP/LP/Notch → 拍分割       │
│   │                                                      │
│   ├──► 特征层 (现成双专家): P2A (心律失常) + KD (心梗形态)│
│   │        每拍两个概率 + 形态特征                        │
│   │                                                      │
│   ├──► 模式 B 报警层 (新增, 本计划核心):                  │
│   │    ├ 心律安全逻辑    (停搏/过缓/过速, 纯逻辑, 秒级)   │
│   │    ├ VF/VT 检测器    (DSP 特征, 3-5s 窗, 秒级)       │
│   │    ├ AF 检测         (RR 不规则度, 30-60s 窗, 通知级) │
│   │    └ 缺血/ST 分支    (KD + ST测量 + 个人基线, 分钟级) │
│   │         └─► 报警分级 (震动/蜂鸣 → App 推送)          │
│   │                                                      │
│   └──► 数据记录层 (新增): RR序列+特征+30s片段 压缩存储    │
│              │ (WiFi/充电时批量上传, 省电)               │
└──────────────┼──────────────────────────────────────────┘
               ▼
云端 (模式 A 长程分析)
├── AF 负荷确认与报告      ├── HRV (SDNN/rMSSD/LF-HF) 趋势
├── 异位负荷 (PVC/PAC %)   ├── ST 段长期趋势 (间歇性缺血)
├── 形态聚类 (新异常形态)  └── 多周健康报告 + 就医建议
```

---

## 3. 文献证据基础（已核实，零编造）

> 引用规范遵循 AGENTS.md §6：真实 DOI/URL，可下载者已存 `papers/`。
> 编号 R1-R12 供 §7 决策表引用。

### 3.1 消费级 AF 检测（模式 B AF 分支 + 模式 A AF 负荷）—— 证据充分

- **R1** Perez MV, Mahaffey KW, Hedlin H, et al. Large-Scale Assessment of a Smartwatch to
  Identify Atrial Fibrillation. *N Engl J Med*. 2019;381(20):1909-1917.
  DOI: 10.1056/NEJMoa1901183 [仅DOI]
  → **Apple Heart Study**: 419,297 人、117 天监测，PPG 不规则脉冲通知 AF；
  通知率仅 0.52%，其中 34% 经 ECG 贴片确认 AF；PPV 0.84（同时段 ECG 确认）。
  **关键设计借鉴**: ①算法以"最小化假阳性"为目标（通知率 0.52% 即证据）；
  ②需要**多次确认**（4 次 PPG 复查）才通知 —— 消费级 AF 检测的"确认-再确认"范式。
  ③明确"不检测短阵 AF"——消费设备接受低负担 AF 漏检，换低误报。

- **R2** Hannun AY, Rajpurkar P, Haghpanahi M, et al. Cardiologist-level arrhythmia
  detection and classification in ambulatory electrocardiograms using a deep neural
  network. *Nat Med*. 2019;25(1):65-69. DOI: 10.1038/s41591-018-0268-3 [已下载]
  → 30s 单导联序列 → DNN 12 类心律（含 AF、VT、AVB）；PhysioNet 2017 AF 挑战
  F1 0.83。**支持**: 云/板上序列级 AF 分类可用端到端 DNN；30s 窗口是行业标准。

- **R3** Goldenthal IL, Sciacca RR, Riga T, et al. Recurrent atrial fibrillation/flutter
  detection after ablation or cardioversion using the AliveCor KardiaMobile device:
  iHEART results. *J Cardiovasc Electrophysiol*. 2019;30(10):2220-2228.
  DOI: 10.1111/jce.14160 [仅DOI]
  → 消融后每日 KardiaMobile 单导联 30s 记录：AF 复发检出时间提前（HR 1.56）。
  **支持**: 消费级单导联 30s 记录 + 算法判读是已验证的产品形态。

- **R4** VITAL-AF 子分析（KardiaMobile 1L 算法 vs 心内科判读, 30,000+ 条 30s 记录）:
  Performance of single-lead handheld electrocardiograms for atrial fibrillation
  screening in primary care. PMC11198293. [仅URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC11198293/]
  → 基层筛查实测: 算法判 AF 的 PPV 52%（心内科确认）；判"正常"的 NPV 极好
  （仅 0.2% 被 overread 为 AF）；**~20% 判读为"无法判定"**（equivoal）。
  **关键设计借鉴**: 消费级 AF 算法"正常"结论可靠（排除价值），"异常"需人工复核；
  **必须保留"无法判定"输出**，不强行二分类。

### 3.2 VF/VT 检测（模式 B VF 分支）—— 证据充分，轻量可部署

- **R5** Plešinger F, et al. Ventricular tachycardia and fibrillation detection in short
  (3-s) ECG blocks. *Computing in Cardiology*. 2018;45. DOI: 10.22489/CinC.2018.037
  [仅URL: https://www.cinc.org/archives/2018/pdf/CinC2018-037.pdf]
  → 3s 单导联块 + 5 个特征（频谱谐波比/导数比/变化范围/自相关）+ 逻辑回归：
  CUDB 训练 Se 95%/Sp 97% (AUC 0.99)，VFDB 独立测试 Se 95%/Sp 83%。
  **关键**: 特征极简、可在微型设备实现；**注意 VFDB 上 Sp 降到 83%**
  （正常块 78%、停搏块 79% 误分类）—— 消费设备需叠加心律逻辑过滤。

- **R6** Prabhakararao E, Manikandan MS. Efficient and robust ventricular tachycardia and
  fibrillation detection method for wearable cardiac health monitoring devices.
  *Healthc Technol Lett*. 2016;3(3):187-194. DOI: 10.1049/htl.2016.0010 [仅DOI]
  → ZCR（过零率）+ PPI（峰-峰间隔）两特征：6 库 18,000 段 Se 99.61%/Sp 99.96%，
  VT/VF 区分 Se 100%/Sp 99.70%；编码耗时仅 0.781ms。**板上实时可行**。

- **R7** Kwon S, Kim J, Chu C-H. Real-Time Ventricular Fibrillation Detection Using an
  Embedded Microcontroller in a Pervasive Environment. *Electronics*. 2018;7(6):88.
  DOI: 10.3390/electronics7060088 [仅DOI]
  → 微控制器上运行 5 种轻量 VF 算法实测：TD（时延/相空间重建）Se 96.56%/
  Sp 81.53%；板上检测比无线连续传输省电约 5 倍。**支持 ESP32 形态**。

### 3.3 缺血/心梗检测（模式 B 缺血分支，PTB 专家定位处）—— 证据关键且含重要局限

- **R8** Hopenfeld B, John MS, Fischell TA, Johnson SR. A Statistically Based Acute
  Ischemia Detection Algorithm Suitable for an Implantable Device.
  *Ann Biomed Eng*. 2012;40(12):2627-2638. DOI: 10.1007/s10439-012-0612-6 [仅DOI]
  → **本分支的黄金方法学**: ①启动期学习用户自己的 ST 偏移随心率分布（个人基线）；
  ②按心率分箱生成 ST 上下阈值；③ST 持续超阈值 ≥5min 才报急性缺血。
  PCI 球囊阻塞检测: LAD 17/17、LCX 7/8、RCA 8/9，对照组 10 天日常无假阳性。
  **直接移植**: "10s 段每 30s 处理一次 + 心率相关个人阈值 + 持续 5min 判定"。

- **R9** Jager F, Moody GB, Mark RG. Detection of transient ST segment episodes during
  ambulatory ECG monitoring. *Comput Biomed Res*. 1998;31(5):305-318.
  [仅URL: http://georgebmoody.com/publications/jager-cbr-1998.pdf]
  → ESC ST-T 数据库 (90 条 2h 记录, 379 段 ST 事件): ST 电平在 **J+80ms** 测量
  （HR>120bpm 用 J+60ms）；事件 = ST 偏离 ≥100μV 持续 ≥30s；需**区分缺血性与
  非缺血性 ST 变化（电轴移位）**。**奠定 ST 测量协议**。

- **R10** Aasvang EK, et al. Wireless Single-Lead versus Standard 12-Lead ECG, for
  ST-Segment Deviation during Adenosine Cardiac Stress Scintigraphy.
  *Sensors*. 2023;23(6):2962. DOI: 10.3390/s23062962 [仅DOI]
  → **必须诚实面对的局限**: 可逆前外侧缺血检测，单导联灵敏度仅 8.3%、
  12 导联 12.5%（特异性均 ~90%）。**单导联 ST 灵敏度有限是物理/解剖事实**
  （单导联只覆盖部分心肌向量）。→ 产品定位: 缺血分支提供**趋势与提示**，
  不承诺检出所有缺血；需在 UI 明确。

- **R11** Pickett T, et al. Linear-phase high-pass filtering and ST-segment distortion.
  *Computing in Cardiology*. 2023. [已下载: papers/Pickett2023_CinC_linear_phase_HP_ECG.pdf]
  → **0.5Hz IIR 高通滤波过冲会制造假性 ST 抬高**。→ 现固件 0.5Hz Butterworth
  HP 对 ST 测量是直接威胁，**ST 分支必须改用线性相位 HP 或补偿**（架构决策 D7）。

- **R12** Makimoto H, et al. Performance of Deep Learning Models in Detecting Myocardial
  Infarction Using 12-Lead ECG. *Sci Rep*. 2020;10:8445.
  DOI: 10.1038/s41598-020-65105-x [已下载]
  → CNN 对 MI 的注意力聚焦在 **ST-T 段**（Grad-CAM 可视化）；非 MI 时聚焦分散。
  **支持**: PTB 训练的拍级 CNN（即我们的 KD）确实是 ST-T 形态特征提取器 ——
  它看的就是缺血最相关的区域。

### 3.4 补充（模式 A 云端长程分析的形态学支撑）

- **R13** (仓库) Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL
  (Wagner P, et al. *Sci Data*. 2020;7:154. DOI: 10.1038/s41597-020-0495-6) [已下载 md]
  → 云端多分类（44 类诊断）标准基准与数据基础。
- **R14** (仓库) 基于深度学习的 P 波/QRS/T 波分类、心率失常 LSTM-CNN 联合模型等
  中文文献 [已下载 PDF，详见 papers/LITERATURE_MATRIX.md 条目]。

---

## 4. 系统架构：条件特定决策层分治（核心设计）

### 4.1 为什么必须分治（证据链）

| 危重条件 | 本质 | 单拍二分类能做到? | 正确检测器 | 文献 |
|---|---|---|---|---|
| 停搏/过缓/过速 | 节律+心率 | ✗ | 纯逻辑（RR 阈值） | 常识/ECG 标准 |
| VF/VT | 波形崩塌 | ✗（形态全乱） | DSP 特征 (ZCR/PPI/TD) | R5 R6 R7 |
| 房颤 | RR 不规则 | ✗ | RR 不规则度 + 多次确认 | R1 R2 R4 |
| 缺血/心梗 | ST-T 形态偏移 | 半（KD 看形态） | KD + ST 量化 + 个人基线 | R8 R9 R10 R12 |
| 心律失常(PVC 等) | 拍形态 | ✓ 但误报高 | 特征层 → 云端负荷统计 | §8.9.x |

### 4.2 板上报警层模块设计（模式 B）

#### 模块 1: 心律安全逻辑（纯逻辑，零模型，第一优先级，秒级）
- 复用现成 `heartrate.cpp` (Pan-Tompkins)：
  - **停搏**: 无 QRS ≥ 4s → 立即报警（最致命、最容易检）
  - **重度过缓**: HR < 40bpm 持续 ≥30s → 报警（需运动/噪声门控）
  - **重度过速**: HR > 180bpm 持续 ≥30s → 报警（排除运动，SQI 门控）
- 文献: ECG 监测标准（J+80ms 等来自 R9 协议家族）；无专用论文必要。

#### 模块 2: VF/VT 检测器（DSP 特征，3-5s 窗，秒级）
- 首选 **ZCR + PPI**（R6, 0.781ms/段）或 **5 特征逻辑回归**（R5, 3s 块）；
  次选相空间 TD（R7, 微控制器已验证）。
- 报警逻辑: 3-5s 窗判定 + **连续 2 窗确认**（对齐 R1 的多次确认范式压误报）；
  叠加模块 1 的心率判据（VF 通常伴随心率丧失规律性）。
- 数据: 需 VFDB/CUDB 训练与验证（§6 缺口）。
- 诚实说明: VF 时用户已意识丧失，报警价值 = 通知身边的人/紧急联系人。

#### 模块 3: AF 检测（RR 不规则度，30-60s 窗，通知级）
- RR 间期序列 → 不规则度指标（变异系数/Shannon 熵/R2 的 DNN 序列分类）：
  30s 窗口判"疑似 AF"；**需多次独立窗口确认才通知**（R1: 4 次确认范式）。
- 输出分三档: 正常 / **疑似 AF（通知）** / **无法判定**（R4: ~20% 判读保留
  "无法判定"，不强行二分类）。
- 板上用 DSP 不规则度；云端用 R2 式 DNN 序列分类复核（模式 A 融合）。

#### 模块 4: 缺血/心梗分支（★PTB 专家部署处，分钟级）
- **架构**: KD a070_t1 拍级形态分（R12 支持其聚焦 ST-T）+ ST 段量化测量 + 个人基线。
- **ST 测量协议**（R8/R9）: J+80ms 点（HR>120 用 J+60ms）ST 电平；PR 段作等电位
  参考；10s 段每 30s 出一次平均 ST；**0.5Hz HP 滤波必须线性相位**（R11）。
- **个人基线**（R8 直接移植）: 启动/每日学习"用户自己 ST 随心率分布" → 按心率分箱
  生成上下阈值；ST 持续超阈值 ≥5min → 报警。
- **KD 的角色**: ①作为 ST 测量的形态质量门控（SQI 低/异位拍剔除）；②在 ST 变化
  模糊时提供形态佐证（KD 高置信 + ST 漂移 → 提高报警权重）；③云端做 ST-T 形态
  聚类（模式 A）。
- **单模型部署结论**: 在此分支，**只需 KD（+ST 逻辑），不需要 P2A 参与** ——
  §8.9.9 论证的"单模型部署在缺血分支成立"落地。P2A 专管心律失常/PVC 特征。

#### 报警分级与触发汇总

| 级别 | 条件 | 时延 | 形式 | 误报容忍 |
|---|---|---|---|---|
| 危急 | 停搏/VF/VT | ≤10s | 震动+蜂鸣+App 强推+紧急联系 | 极低（宁漏不误→实际宁多报, 见 §7.5） |
| 严重 | ST 持续偏移 ≥5min | 分钟级 | App 推送"建议尽快就医" | 低（个人基线抑制） |
| 提示 | AF 疑似（多窗确认） | 30-60s | App 通知"建议门诊复查" | 中（R1 范式） |
| 报告 | PVC 负荷/HRV 趋势 | 天~周 | 云端周报 | 高（可解释） |

---

## 5. 云端长程分析层设计（模式 A）

### 5.1 上传协议（板上→云）
- 每拍: 时间戳 + RR 间期 + 拍级特征（P2A/KD 概率 + 形态嵌入）
- 每 30s: 压缩波形片段（疑似事件/ST 漂移段保留全波形）
- 事件日志: 报警/通知/停搏等时间线
- 上传时机: WiFi/充电时批量（省电, R7 佐证: 本地处理比连续无线传输省电 5 倍）

### 5.2 云端分析项
1. **AF 负荷确认**: 全天 RR 序列 → DNN 复核（R2）→ 发作段/总时长占比（临床金指标）
2. **异位负荷**: PVC/PAC 计数/总拍（>1% 持续 → 建议就医）
3. **HRV**: SDNN、rMSSD、LF/HF —— 自主神经状态长期趋势
4. **ST 趋势**: 多天 ST 电平曲线 + 每用户基线比对 → 间歇性缺血提示（R10 局限标注）
5. **形态聚类**: 拍级嵌入聚类 → "新出现的异常形态"事件提醒
6. **周报/月报**: 图 + 可能诊断 + 置信度 + 免责声明（"研究原型，非临床诊断"）

---

## 6. 现有资产盘点与数据缺口（诚实）

### 6.1 已有（可直接复用）

| 资产 | 状态 | 新架构角色 |
|---|---|---|
| P2A (MIT-BIH 形态专家) | ✅ 部署态 | 心律失常/PVC 特征层、云端形态聚类 |
| KD a070_t1 (PTB 形态专家) | ✅ 部署态 | **缺血/ST 分支骨干**（模块 4） |
| 能量包络心率检测 (v6) | ✅ 部署态 | 模块 1 全部基础、RR 序列（模块 3 输入） |
| TFLite Micro INT8 推理链 | ✅ 部署态 | 模块 4 复用 |
| 0.5Hz IIR HP + LP40 + Notch | ✅ 部署态 | ⚠️ 模块 4 需改线性相位 HP（R11） |
| §8.9.9 段级决策逻辑 | ✅ 研究态 | 模块 3/4 的"多窗确认"基础 |

### 6.2 缺口（需新增，按优先级）

| 缺口 | 用途 | 数据/文献依据 | 工作量 |
|---|---|---|---|
| **VF/VT 检测器** | 模块 2 | VFDB/CUDB（R5-R7 均用）| 中（DSP 特征即可，不必 CNN） |
| **AF RR 不规则算法** | 模块 3 | AFDB 验证；R1 范式 | 小（DSP） |
| **ST 测量 + 个人基线** | 模块 4 | ESC ST-T DB（R9）| 中（算法）+ 大（数据标注） |
| **线性相位 HP 滤波** | 模块 4 前置 | R11 | 小（固件滤波改动，仅编译检查） |
| **云端分析服务** | 模式 A | PTB-XL（R13）| 大（独立于固件路线） |

---

## 7. 关键架构决策表（含文献证据）

| # | 决策 | 依据 | 结论 |
|---|---|---|---|
| D1 | 拍级二分不再直接报警 | §8.9.8/8.9.9 实证（误报 21.6%、先验塌缩）| 拍级 = 特征层 |
| D2 | 条件特定决策层分治 | §4.1 表 | 4 模块并行 |
| D3 | PTB 专家做缺血分支骨干 | R12 + §8.9.9（单模型分支成立）| 模块 4 只用 KD |
| D4 | AF 检测多次确认 + 三档输出 | R1（4 次确认）、R4（无法判定档）| 模块 3 |
| D5 | VF 用轻量 DSP 特征 | R5/R6/R7（微控制器实测）| 模块 2 |
| D6 | ST 个人基线 + 心率分箱 + 持续 5min | R8（黄金方法学）| 模块 4 |
| D7 | ST 测量 J+80ms + 线性相位 HP | R9 + R11 | 模块 4 + 固件滤波改动 |
| D8 | 缺血提示定位"趋势/提示"非诊断 | R10（单导联 Se 8.3% 诚实上限）| UI/免责声明 |
| D9 | 模式 A 云端长程 + 模式 B 板上实时 | R1（Apple Heart 双模式先例）| 双模式架构 |
| D10 | 上报"无法判定"而非硬二分类 | R4（~20% equivocal 实测）| 模块 3/4 输出 |

---

## 8. 部署路径与里程碑

```
Phase C1 (纯逻辑层, 最快落地): 模块1 心律安全 + 数据记录层 + 上传协议
  ├── 零模型、零训练，当天可设计，仅固件逻辑 + App 显示
  └── 验收: 停搏/过缓/过速回放测试 (模拟器模式)

Phase C2 (AF 分支): 模块3 RR 不规则度 + 三档输出
  ├── DSP 算法 + AFDB 数据评估（下载 AFDB, 患者级划分）
  └── 验收: 30s 窗 AF 检出 AUC + "无法判定"比例报告

Phase C3 (VF 分支): 模块2 ZCR/PPI + 逻辑回归
  ├── 下载 VFDB/CUDB → 训练/验证 → INT8 导出（复用 export 链）
  └── 验收: VFDB 独立测试 Se/Sp 复现 (目标 ≥ R5: Se 95%/Sp 83%)

Phase C4 (缺血分支, 最难): 模块4 KD + ST 测量 + 个人基线
  ├── 固件改线性相位 HP (D7) + ST 测量协议 + ESC ST-T DB 评估
  └── 验收: 仿真 ST 漂移事件检出 + 个人基线误报率报告 (对齐 R8 方法)

Phase C5 (云端模式 A): 上传管道 + AF 负荷/HRV/ST 趋势/周报
  └── 验收: 端到端 (板上→云→报告) 演示 + 免责声明合规

Phase C6 (产品化): 功耗优化 + 法规路径 (FDA 510(k) 消费级 ECG / CE) + 临床验证
  └── 参考: Kardia FDA 510(k) (R4 生态), NICE MTG64 (R3 生态)
```

---

## 9. 风险与诚实边界

1. **单导联 ST 灵敏度上限（R10）**: 前外侧缺血单导联 Se 仅 8.3% —— 缺血分支
   明确定位为"持续监测+趋势提示"，UI 与说明书必须如实声明，不承诺检出所有缺血。
2. **先验塌缩不可消除**: 患病率 1.5-3.4% 下任何模型的 P 都塌到 0.05-0.11（§8.9.8/8.9.9
   实证）—— 报警后必须引导就医确认，产品靠"通知-确认"闭环而非"报警即诊断"。
3. **VF 数据量小（R5 自述 CUDB 35 条/VFDB 22 条）**: VF 模型跨库 Sp 会掉（83%），
   必须叠加心律逻辑（模块 1）压误报。
4. **消费设备"报警疲劳"**: R1 显示行业以"极低通知率"为设计目标；本设备同样
   宁严勿松，分级报警 + 可配置灵敏度。
5. **法规**: 若宣称"检测/诊断"则需医疗器械注册（消费级 ECG 有 510(k) 先例）；
   若仅"健康监测"则走健康类目 —— 架构上两种路径兼容（输出分级可配置）。
6. **数据隐私**: 云端 ECG 数据属敏感健康信息，需脱敏 + 匿名化 + 用户授权 +
   合规存储（参照 HIPAA/GDPR/《个人信息保护法》）。

---

## 10. 参考文献（已核实，AGENTS.md §6）

1. Perez MV, et al. Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation.
   *N Engl J Med*. 2019;381(20):1909-1917. DOI: 10.1056/NEJMoa1901183 [仅DOI]
2. Hannun AY, et al. Cardiologist-level arrhythmia detection and classification in
   ambulatory electrocardiograms using a deep neural network. *Nat Med*. 2019;25(1):65-69.
   DOI: 10.1038/s41591-018-0268-3 [已下载]
3. Goldenthal IL, et al. Recurrent atrial fibrillation/flutter detection after ablation or
   cardioversion using the AliveCor KardiaMobile device: iHEART results.
   *J Cardiovasc Electrophysiol*. 2019;30(10):2220-2228. DOI: 10.1111/jce.14160 [仅DOI]
4. Performance of single-lead handheld ECGs for AF screening in primary care (VITAL-AF
   子分析). PMC11198293. https://pmc.ncbi.nlm.nih.gov/articles/PMC11198293/ [仅URL]
5. Plešinger F, et al. VT/VF detection in short (3-s) ECG blocks. *Computing in
   Cardiology*. 2018. DOI: 10.22489/CinC.2018.037 [仅URL]
6. Prabhakararao E, Manikandan MS. Efficient and robust VT and VF detection method for
   wearable cardiac health monitoring devices. *Healthc Technol Lett*. 2016;3(3):187-194.
   DOI: 10.1049/htl.2016.0010 [仅DOI]
7. Kwon S, Kim J, Chu C-H. Real-Time Ventricular Fibrillation Detection Using an Embedded
   Microcontroller. *Electronics*. 2018;7(6):88. DOI: 10.3390/electronics7060088 [仅DOI]
8. Hopenfeld B, John MS, Fischell TA, Johnson SR. A Statistically Based Acute Ischemia
   Detection Algorithm Suitable for an Implantable Device. *Ann Biomed Eng*.
   2012;40(12):2627-2638. DOI: 10.1007/s10439-012-0612-6 [仅DOI]
9. Jager F, Moody GB, Mark RG. Detection of transient ST segment episodes during
   ambulatory ECG monitoring. *Comput Biomed Res*. 1998;31(5):305-318.
   http://georgebmoody.com/publications/jager-cbr-1998.pdf [仅URL]
10. Aasvang EK, et al. Wireless Single-Lead versus Standard 12-Lead ECG for ST-Segment
    Deviation during Adenosine Cardiac Stress Scintigraphy. *Sensors*. 2023;23(6):2962.
    DOI: 10.3390/s23062962 [仅DOI]
11. Pickett T, et al. Linear-phase high-pass filtering and ST-segment distortion.
    *Computing in Cardiology*. 2023. [已下载: papers/Pickett2023_CinC_linear_phase_HP_ECG.pdf]
12. Makimoto H, et al. Performance of Deep Learning Models in Detecting Myocardial
    Infarction Using 12-Lead ECG. *Sci Rep*. 2020;10:8445.
    DOI: 10.1038/s41598-020-65105-x [已下载]
13. Wagner P, et al. PTB-XL, a large publicly available electrocardiography dataset.
    *Sci Data*. 2020;7:154. DOI: 10.1038/s41597-020-0495-6 [已下载 md]
14. 项目内部证据: TUNING_HISTORY.md §8.9.1/§8.9.7/§8.9.8/§8.9.9（前置关卡 +20% P、
    事件级口径、先验塌缩、段级决策层研究）；models/binary_class_eval_all.json；
    models/record_level_eval.json。
