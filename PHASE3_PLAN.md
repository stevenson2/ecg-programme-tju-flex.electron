# ECG 模型进化 Phase 3 提示词

## 项目背景
ESP32-S3 便携心电采集 + AI 异常检测。当前最优：ResNet-L (63K params) on MIT-BIH+(263K beats)，Acc 96.01%，AUC 0.9669，Recall@0.5=82.40%。263K beat 数据量已达天花板。

## 核心瓶颈
- 数据量不足（263K beats），任何模型技巧无法突破
- SSL 预训练（PTB-XL SimCLR）未带来增益
- 均衡采样/FocalLoss α 调高牺牲 AUC 换 Recall
- SVDB 数据分布不匹配，反而降低性能

## 文献依据（papers/ 目录）
1. **PTB-XL Benchmark** (Strodthoff et al 2021, IEEE JBHI): Concat pooling(GAP+max)比纯GAP好；kernel=5 最优；TTA滑动窗口+max聚合显著提升；1-cycle LR+AdamW；100Hz采样，2.5s窗口；xResNet-1D表现最佳
2. **Nature SCD** (Obermeyer et al 2026): 多任务学习（SCD+SCD vs其他死因+LVEF）在0.6%极端不平衡下有效；单导联模型几乎与12导联一样好；原始波形训练无需滤波；64层ResNet+128filter+kernel=16；TTA滑动窗口推理；Lockbox验证方法论
3. **DeepECG-Net** (Alghieth 2025, Sci Rep): Transformer+CNN混合；注意力去噪(SNR 5.2→14.5dB)；MIT-BIH上98.3%Acc；30MB on Pi4B；二进制异常分组

## Phase 3 路线

### 路线 F：多任务学习 🔴 P0
- 主任务：Normal/Abnormal 二分类（当前）
- 辅助任务1：心率回归（已有BPM标签可用）
- 辅助任务2：信号质量SQI预测（已有SQI特征）
- 辅助任务3：波形形态分类（P/QRS/T是否存在异常）
- 共享ResNet编码器+多头输出，loss加权求和
- 目标：等效数据量×3，Recall 82%→86-88%

### 路线 G：测试时增强TTA 🔴 P0
- 推理时滑动窗口1s×3次（stride=0.25s），取max聚合
- 连续N拍异常确认（已有config，提升N=3）
- 目标：Recall +3-5%

### 路线 H：架构微调 🟡 P1
- Concat pooling: GAP + GlobalMaxPool 拼接（替换当前仅GAP）
- ResBlock kernel: 7/5/3 → 统一5
- LR schedule: CosineDecayRestarts → 1-cycle
- 目标：Acc +0.5-1%

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

## 关键约束
- ESP32-S3 512KB SRAM，模型需控制在80-100KB INT8内
- 采样率250Hz，每次推理1s窗口(250点)
- Python训练在WSL2 Ubuntu中执行，GPU: RTX 5070 Laptop
- BLE/串口CSV格式: clean,noisy,filtered,bpm,true_bpm,sqi,motion,abnormal_flag,confidence

## 备选数据集
- MIT-BIH Arrhythmia (87K beats, 已用)
- INCART (176K beats, 已用)
- SVDB (184K beats, 已预处理但分布不匹配，暂弃)
- PTB-XL (21K records, 记录级标签不适合beat级训练)
- 生理信号库中其他beat级标注数据集可探索

## 部署当前最优
- 单模型部署: ResNet-L (final_resnet_l.h5), Recall@0.35=84.18%
- 最高Acc: Ensemble×3 (ensemble_seed{42,123,456}.h5)
- 阈值: 0.35 + 2拍连续确认
