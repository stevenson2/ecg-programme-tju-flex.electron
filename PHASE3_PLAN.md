# ECG 模型进化 Phase 3 提示词

## 项目背景
ESP32-S3 便携心电采集 + AI 异常检测。当前最优：ResNet-L (63K params) on MIT-BIH+(263K beats)，Acc 96.01%，AUC 0.9669，Recall@0.5=82.40%。263K beat 数据量已达天花板。

## 核心瓶颈
- 数据量不足（263K beats），模型容量/技巧接近极限
- **SSL 预训练失败根因**：SimCLR 是无监督对比学习，与下游分类任务差距大；论文指出**监督预训练**（PTB-XL → finetune）在小数据集 regime 提升最大
- 均衡采样/FocalLoss α 调高牺牲 AUC 换 Recall — **治标不治本**
- SVDB 数据分布不匹配，反而降低性能
- **当前 PTB-XL 用法错误**：beat 级提取破坏了记录级上下文，应改为 record-level 监督预训练

## 文献依据（papers/ 目录）
1. **PTB-XL Benchmark** (Strodthoff et al 2021, IEEE JBHI): Concat pooling(GAP+max)比纯GAP好；kernel=5 最优；TTA滑动窗口+max聚合显著提升；1-cycle LR+AdamW；100Hz采样，2.5s窗口；xResNet-1D表现最佳
2. **Nature SCD** (Obermeyer et al 2026): 多任务学习（SCD+SCD vs其他死因+LVEF）在0.6%极端不平衡下有效；单导联模型几乎与12导联一样好；原始波形训练无需滤波；64层ResNet+128filter+kernel=16；TTA滑动窗口推理；Lockbox验证方法论
3. **DeepECG-Net** (Alghieth 2025, Sci Rep): Transformer+CNN混合；注意力去噪(SNR 5.2→14.5dB)；MIT-BIH上98.3%Acc；30MB on Pi4B；二进制异常分组

## PTB-XL 数据库利用策略（基于 Strodthoff 2021 论文深度解读）

### 为什么当前 PTB-XL 用法不对？
- 当前方案：PTB-XL → 提取 beat → binary label → 直接训练（效果差，记录级信息丢失）
- 论文方案：**PTB-XL 作为预训练资源** → record-level 5 超类分类 → 迁移到目标数据集

### 路线 K：PTB-XL 监督预训练 + 迁移学习 🔴 P0
- **阶段1**：在 PTB-XL 上训练 5 超类分类器 (NORM/MI/CD/STTC/HYP)
  - 输入：10s 记录 → 2500 采样点 (250Hz) 或 1000 采样点 (100Hz)
  - 评估指标：**macro AUC**（非 accuracy，因类别严重不平衡）
  - 使用 10-fold 标准划分 (fold 1-8 train, 9 val, 10 test)
- **阶段2**：替换分类头 → 在 MIT-BIH+INCART 上 finetune
  - 冻结 encoder 前 N 层，仅训练分类头 + 后几层
  - 论文证明：在小数据集 regime 下迁移学习提升**最大**
  - 本质区别：这是**监督**预训练（SimCLR 是无监督 SSL，已失败）
- **目标**：Recall 82%→88-90%，AUC 0.97→0.98+

### PTB-XL 关键设计决策（从论文提取）
| 技巧 | 论文依据 | 当前状态 |
|------|---------|---------|
| **Concat pooling** (GAP + MaxPool) | 比纯 GAP 一致更好 | ❌ 仅用 GAP |
| **Kernel=5 全网络统一** | 实验验证最优 | ❌ 7/5/3/3 混合 |
| **1-cycle LR + AdamW** | 收敛更快更稳 | ❌ CosineDecayRestarts |
| **100Hz 采样足够** | 论文全用 100Hz | ⚠️ 我们用 250Hz |
| **macro AUC 评估** | 避免大类主导 | ❌ 用 accuracy |
| **Dropout 0.25/0.5 在 FC 层** | BN + Dropout 组合 | ⚠️ 仅 0.3 在单 FC |
| **Ensemble 统计显著提升** | 多数任务 ensemble 最优 | ✅ 已有 Ensemble×3 |
| **Hidden stratification** | 发现亚组性能差异 | ❌ 未分析 |

### 提升 Recall 专项策略（综合三篇论文）

#### 1. 数据层面
- **PTB-XL 监督预训练**（路线 K）：增加有效数据量 → 编码器学到更通用特征
- **类平衡重采样 + Macro AUC 优化**：替代 accuracy 作为早停指标
- **Hard example mining**：对当前 Recall 低的记录/亚组针对性增广

#### 2. 模型层面
- **Concat pooling**（路线 H）：GAP + MaxPool → 保留峰值信息 + 全局信息
- **Pre-activation ResBlock**（BN→ReLU→Conv 替换 Conv→BN→ReLU）：梯度流更好
- **多任务学习**（路线 F）：已完成，辅助任务正则化共享编码器

#### 3. 推理层面
- **TTA 增强模式**（路线 G）：已完成，noise + amplitude jitter → mean 聚合
- **多拍确认 N=3**（路线 G）：按记录连续确认，非全局乱序
- **模型不确定性筛选**：预测 entropy 高的样本 → 提高确认阈值或人工复核

## Phase 3 路线

### 路线 K：PTB-XL 监督预训练 + 迁移学习 🔴 P0（新）
- 阶段1：PTB-XL 10s record-level 5超类分类 (NORM/MI/CD/STTC/HYP)
- 阶段2：替换分类头 → MIT-BIH+INCART beat-level finetune
- 10-fold CV，macro AUC 评估，1-cycle LR + AdamW
- 目标：Recall 82%→88-90%，AUC 0.97→0.98+

### 路线 F：多任务学习 🔴 P0
- 主任务：Normal/Abnormal 二分类（当前）
- 辅助任务1：心率回归（已有BPM标签可用）
- 辅助任务2：信号质量SQI预测（已有SQI特征）
- 辅助任务3：波形形态分类（P/QRS/T是否存在异常）
- 共享ResNet编码器+多头输出，loss加权求和
- 目标：等效数据量×3，Recall 82%→86-88%

### 路线 G：测试时增强TTA 🔴 P0
- 推理时增强模式（noise + amplitude jitter → mean 聚合）
- 滑动窗口模式保留给 ESP32 流式推理
- 连续 N=3 拍异常确认（按记录连续，非全局乱序）
- 目标：Recall +3-5%

### 路线 H：架构微调 🟡 P1
- Concat pooling: GAP + GlobalMaxPool 拼接（替换当前仅GAP）
- ResBlock kernel: 7/5/3 → 统一5
- LR schedule: CosineDecayRestarts → 1-cycle (warmup + linear decay)
- Pre-activation ResBlock: BN→ReLU→Conv 替换 Conv→BN→ReLU
- Dropout: 分类头改为 0.25/0.5 双层（论文标配）
- 目标：Acc +1-1.5%，AUC +0.5%

### 路线 I：原始波形训练 🟡 P1
- AI推理路径不经ESP32滤波链
- 仅在心率检测保留滤波
- 验证模型是否真正需要去噪预处理

### 路线 J：注意力去噪 🟢 P2
- ResNet尾部加轻量自注意力层
- 借鉴DeepECG-Net的MHSA设计

## 工程文件索引
- 源代码：pc_tools/ecg_dl/
- 模型定义：models/resnet_lite_1d.py
- 模型注册表：models/model_registry.json
- 图表生成：figures.py
- 全局配置：config.py
- 数据加载：data/dataset.py
- 损失函数：losses/focal_loss.py, losses/contrastive.py
- 训练脚本：train.py, train_ssl.py, train_ensemble.py, train_cls_head.py
- ESP32固件：src/ai_inference/, include/ai_inference/
- 文档：README.md, ROADMAP.md, TUNING_HISTORY.md

## 论文文件
- papers/s41598-025-07781-1.pdf (DeepECG-Net)
- papers/s41586-026-10674-6.pdf (Nature SCD)
- papers/Deep_Learning_for_ECG_Analysis_Benchmarks_and_Insights_from_PTB-XL.pdf
- papers/s41591-018-0268-3.pdf (Hannun et al 2019, Nature Medicine — 34层ResNet达心内科医生水平)
- papers/mathematics-11-00562-v2.pdf (Ahmed et al 2023 — 1D-CNN四分类, class_weight处理不平衡)

## 关键约束
- ESP32-S3 512KB SRAM，模型需控制在80-100KB INT8内
- 采样率250Hz，每次推理1s窗口(250点)
- Python训练在WSL2 Ubuntu中执行，GPU: RTX 5070 Laptop
- BLE/串口CSV格式: clean,noisy,filtered,bpm,true_bpm,sqi,motion,abnormal_flag,confidence

## 备选数据集
- MIT-BIH Arrhythmia (87K beats, 已用)
- INCART (176K beats, 已用)
- SVDB (184K beats, 已预处理但分布不匹配，暂弃)
- **PTB-XL (21K records, 18885 患者)**：**不用于 beat 级训练，用于 record-level 监督预训练（路线 K）**。优势：数据量大、多导联、SCP-ECG 标准标注、含诊断似然度、含 demographics（年龄/性别可作为辅助任务）
- 生理信号库中其他beat级标注数据集可探索

## 部署当前最优
- 单模型部署: ResNet-L (final_resnet_l.h5), Recall@0.35=84.18%
- 最高Acc: Ensemble×3 (ensemble_seed{42,123,456}.h5)
- 阈值: 0.35 + 2拍连续确认
