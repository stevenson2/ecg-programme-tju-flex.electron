# 心电图心律失常检测模型强化方案 (ModelPlan)

> **目标**: 将当前 88.1% 的测试精度提升至 ≥97%，同时满足嵌入式低功耗/小内存约束  
> **当前基线**: 1D-CNN v2 (15K 参数, INT8 24.8KB), MIT-BIH 48 条记录, 按病人分组训练, ESP32-S3-SUPERMINI (4MB Flash / 2MB PSRAM)  
> **核心约束**: Flash ≤ 1MB（含固件+模型）, SRAM ≤ 512KB（含 Tensor Arena）, 推理延迟 ≤ 15ms, 功耗 ≤ 200mW  

---

## 一、当前瓶颈诊断

| 瓶颈维度 | 当前状态 | 问题分析 |
|----------|----------|----------|
| **数据集** | MIT-BIH 仅 48 条记录, ~87K 心拍 | 样本多样性不足，无法覆盖真实场景中所有心律失常变异 |
| **模型容量** | 1D-CNN 15K 参数, 3 层卷积 | 表达能力有限, 对 P 波/T 波微形态等细粒度特征捕获不足 |
| **类别不平衡** | 正常:异常 ≈ 2:1 | 模型偏向多数类, 异常召回率不足 |
| **数据增强** | 已放弃 (增强毒化泛化) | 增强方式不当(10x 过度增强 + 噪声失真)，需更换策略 |
| **训练策略** | 标准交叉熵 + Adam + 早停 | 缺乏针对困难样本的聚焦机制 |
| **芯片限制** | ESP32-S3 4MB/2MB, 240MHz | 模型成长空间有限, 需在参数量与精度间精确权衡 |

---

## 二、数据集扩充方案

### 2.1 新增数据集

| 数据集 | 记录数 | 采样率 | 标注量 | 补充价值 |
|--------|--------|--------|--------|----------|
| **PTB-XL** | 21,799 条 | 500Hz/100Hz | ~500K+ 心拍 | 全球最大公开 ECG 数据集，覆盖 71 种诊断类别 |
| **CPSC 2018** | 6,877 条 | 500Hz | ~50K+ 心拍 | 中国人群数据，补充人种多样性 |
| **MIT-BIH SVT** | 78 条 | 128Hz | ~300K 心拍 | 室上性心律失常专项，解决当前对室上速的漏检 |
| **INCART** | 75 条 | 257Hz | ~175K 心拍 | 长时间记录 (30min)，覆盖持续性心律失常 |
| **MIT-BIH LT** | 7 条 | 128Hz | 14-24h 连续记录 | 真实长期佩戴场景的心率变异性 |

### 2.2 数据集整合策略

```
Phase 1 (快速验证):   MIT-BIH 48 + CPSC 2018 + INCART         → ~150K 心拍
Phase 2 (主力训练):   + PTB-XL (筛选心律失常用子集)            → ~400K 心拍
Phase 3 (终极强化):   + MIT-BIH SVT + MIT-BIH LT               → ~600K 心拍
```

### 2.3 统一预处理管线

| 处理步骤 | 参数 | 说明 |
|----------|------|------|
| 重采样 | 统一 → 250Hz | 与 ESP32-S3 采样率匹配 |
| 心拍切割 | R-peak 中心, 250 样本窗口 (1s) | 维持现有 250 点输入不变 |
| 滤波匹配 | HP 0.5Hz + LP 40Hz + Notch 50Hz | **关键**: PC 预处理滤波器参数必须与 ESP32 板载 IIR 滤波器完全一致 |
| 标签映射 | AAMI EC57 → Normal/Abnormal | 统一二分类标准 |
| 病人分组 | 按 patient_id 划分 (7:1.5:1.5) | 严格防止数据泄露 |

---

## 三、模型架构进化路线

### 3.1 现有架构回顾

```
Input(250,1) → Conv1D(16,k7) → BN+ReLU → MaxPool(2) → (125,16)
              → Conv1D(32,k5) → BN+ReLU → MaxPool(2) → (62,32)
              → Conv1D(64,k3) → BN+ReLU → GAP        → (64)
              → Dense(32) → Dropout(0.4) → Dense(2,Softmax)
参数量: ~15K, INT8: 24.8KB, 精度: 88.1%
```

### 3.2 推荐架构: ECG-ResNet-Lite

> 设计原则: **深度可分离卷积 + 残差连接 + SE 注意力**，在 50K-100K 参数内最大化精度

```
Input(250,1)
  │
  ├── Stem: Conv1D(16, k7, stride=2) → BN → ReLU                 (125, 16)
  │
  ├── Block A (×2): DepthwiseConv1D(k5) → Conv1D(16) → SE → +res  (125, 16)
  │     参数量: ~1,200 per block
  │
  ├── Block B (×2): DepthwiseConv1D(k5, stride=2) → Conv1D(32) → SE → +1x1res  (63, 32)
  │     参数量: ~3,500 per block
  │
  ├── Block C (×2): DepthwiseConv1D(k3, stride=2) → Conv1D(64) → SE → +1x1res  (32, 64)
  │     参数量: ~8,000 per block
  │
  ├── Block D (×1): DepthwiseConv1D(k3) → Conv1D(128) → SE        (32, 128)
  │     参数量: ~17,000
  │
  └── Head: GAP → Dense(64, ReLU) → Dropout(0.3) → Dense(2, Softmax)
                                            参数量: ~8,500

总参数量: ~55,000
INT8 大小: ~55 KB
Tensor Arena 预估: 60-80 KB
推理延迟 (ESP32-S3 @240MHz): 预估 8-12ms
```

### 3.3 关键设计决策

| 设计点 | 选择 | 理由 |
|--------|------|------|
| **深度可分离卷积** | ✅ 采用 | 参数量降低 ~80%, 精度损失 <1% |
| **SE (Squeeze-Excitation)** | ✅ 采用 | 通道注意力, 几乎无参数量代价, +1-2% 精度 |
| **残差连接** | ✅ 采用 | 解决深度增加带来的梯度消失, 使 8 层以上训练可行 |
| **Stride=2 代替 Pooling** | ✅ 采用 | 可学习的下采样, 比 MaxPooling 保留更多信息 |
| **BN + ReLU** | ✅ 保留 | 标准化激活分布, INT8 量化友好 |
| **Dropout 0.3** | ✅ 保留 | 仅在训练时启用, 推理时移除 |

### 3.4 备选方案 (如果 55K 参数超预算)

| 方案 | 参数量 | INT8 大小 | 说明 |
|------|--------|-----------|------|
| **ECG-MobileNet-Tiny** | ~25K | ~25 KB | 仅保留 Block A/B, 无 Block C/D |
| **MCU-Net (ECG版)** | ~30K | ~30 KB | 2-stage: 粗特征 + 细特征, 专为 MCU 设计 |
| **Knowledge Distillation** | 任意 | 可配 | 大模型 (ResNet-18, 11M 参数) → 小模型 (25K), 保留 95%+ 性能 |


---

## 四、训练策略强化

### 4.1 损失函数升级

| 损失函数 | 适用场景 | 推荐参数 |
|----------|----------|----------|
| **Focal Loss** (γ=2, α=0.75) | 类别不平衡, 聚焦困难样本 | `alpha=0.75` (Abnormal 权重高) |
| **Label Smoothing** (ε=0.1) | 防止过拟合, 提升泛化 | `label_smoothing=0.1` |
| **最终组合** | `FocalLoss(gamma=2, alpha=0.75) + LabelSmoothing(0.1)` | 同时解决不平衡和过拟合 |

### 4.2 学习率调度

```
Warmup:
  epoch 1-5:  lr 从 1e-6 线性增长至 1e-3

Cosine Annealing + Restarts:
  epoch 6-80: lr = 1e-3 → 1e-6 (余弦衰减, 周期 20 epochs)
  restart 后: lr 重置为 5e-4 → 再衰减

替代方案 (更简单):
  ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6)
```

### 4.3 数据增强（重新设计）

> **关键教训**: 上一版本因 10x 过度增强 + 噪声失真导致精度从 85.6% 降至 75.7%。  
> **新策略**: 温和增强, 最大 2x 扩充, 所有变换必须保持 ECG 生理合理性。

| 增强方法 | 参数范围 | 说明 |
|----------|----------|------|
| **Time Warping** | stretch ∈ [0.92, 1.08] | 模拟心率变异, 轻微拉伸不破坏 QRS 形态 |
| **Amplitude Scaling** | scale ∈ [0.85, 1.15] | 模拟电极接触阻抗变化 |
| **Baseline Wander** | 0.05Hz 正弦, amp=0.15 | 模拟呼吸引起的基线漂移 |
| **Gaussian Noise** | σ ∈ [0.005, 0.015] | 极轻微噪声, 仅模拟 ADC 量化噪声 |

**增强应用概率**: 每次训练迭代 50% 概率应用增强 (而非对所有样本增强)

### 4.4 Mixup 正则化

```python
# 1D ECG 信号 Mixup
λ ~ Beta(0.2, 0.2)  # 偏向两端的混合
x_mixed = λ * x_i + (1 - λ) * x_j
y_mixed = λ * y_i + (1 - λ) * y_j
```

Mixup 可提供 ~1-3% 泛化提升，几乎零计算开销。

### 4.5 完整训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 优化器 | AdamW (weight_decay=1e-4) | 带权重衰减的 Adam, 更好泛化 |
| 初始学习率 | 1e-3 | Warmup 后 |
| 批大小 | 64 → 128 | 更大 batch 利于 BN 统计量估计 |
| 训练轮数 | 100 (EarlyStopping patience=20) | 更多 epochs 充分收敛 |
| 验证策略 | 5-Fold Cross-Validation (按病人分组) | 更稳健的精度评估, 避免划分偏差 |
| 类别权重 | `{Normal: 1.0, Abnormal: 2.5}` | 配合 Focal Loss 双重解决不平衡 |
| 混合精度 | FP16 (PC 训练) → INT8 (部署) | 加速训练, 并为量化做准备 |


---

## 五、量化与压缩

### 5.1 QAT (Quantization-Aware Training)

> **重大问题**: 当前使用 Post-Training Quantization，INT8 精度 88.1% 竟然比 FP32 的 85.6% 更高 (量化正则化效应)，但这不具备可复现性。

改用 **QAT (量化感知训练)**:

```python
import tensorflow_model_optimization as tfmot

# 在训练最后 10 个 epoch 插入 Fake Quantization 节点
model = tfmot.quantization.keras.quantize_model(model)

# 微调 10 epochs (lr=1e-5)
model.fit(train_ds, epochs=10, ...)
```

| 量化方式 | 当前 PTQ | 推荐 QAT |
|----------|----------|----------|
| 精度损失 | 不可控 (可能 +2% 也可能 -5%) | 可控 (±1% 内) |
| 额外训练 | 无 | 10 epoch 微调 |
| Tensor Arena | 32KB → 80KB (需重新测算) | 同上 |

### 5.2 结构化剪枝

```
目标: 将 55K 模型剪枝至 35K 以内
方法: 训练后 → 按 L1 范数剪枝 30-40% 权重 → 微调恢复精度
工具: tfmot.sparsity.keras.prune_low_magnitude
预期: 精度损失 <1%，INT8 模型 ~35 KB
```

### 5.3 知识蒸馏（终极方案）

```
Teacher: ResNet-18-1D (1.1M 参数, 在 600K 心拍上训练到 98%+ 精度)
         ↓ Soft Targets (温度 T=3)
Student: ECG-ResNet-Lite (25K 参数)
         ↓ 损失函数: α·CE(y_true, y_student) + (1-α)·KL(y_teacher/T, y_student/T)·T²

预期: Student 在 25K 参数下达到 96-97% 精度


---

## 六、芯片/硬件平台评估

### 6.1 需求分析

| 指标 | 当前 (v2) | 强化后 (55K) | 上限 (带蒸馏 80K) |
|------|-----------|-------------|-------------------|
| 模型大小 (INT8) | 24.8 KB | ~55 KB | ~80 KB |
| Tensor Arena | 32 KB | 60-80 KB | 90-110 KB |
| 推理延迟 | ~5ms | 8-12ms | 12-18ms |
| Flash 占用 (Runtime+模型) | ~100 KB | ~150 KB | ~200 KB |
| SRAM 占用 | ~37 KB | ~100 KB | ~140 KB |
| 峰值功耗 | ~150mW | ~180mW | ~200mW |

### 6.2 芯片选型对比

| 芯片 | 核心 | 频率 | Flash | SRAM | AI加速 | 功耗 | 单价 | 推荐度 |
|------|------|------|-------|------|--------|------|------|--------|
| **ESP32-S3 (当前)** | LX7 ×2 | 240MHz | 4MB | 2MB | 无 | ~180mW | ¥15 | ⭐⭐⭐ |
| **ESP32-S3 高配** | LX7 ×2 | 240MHz | 8MB | 8MB | 无 | ~180mW | ¥25 | ⭐⭐⭐⭐ |
| **ESP32-P4** | RISC-V ×2 | 400MHz | 16MB | 16MB | **内置 NN 加速器** | ~250mW | ¥35 | ⭐⭐⭐⭐⭐ |
| **MAX78000** | M4+RISC-V | 100MHz | 2MB | 512KB | **CNN 硬件加速 (442KB)** | **<5mW (推理)** | ¥60 | ⭐⭐⭐⭐⭐ |
| **STM32H747** | M7+M4 | 480MHz | 2MB | 1MB | DSP 指令 | ~300mW | ¥80 | ⭐⭐⭐ |
| **GAP8** | RISC-V ×9 | 250MHz | 512KB | 512KB | **硬件卷积加速** | <10mW | ¥150+ | ⭐⭐⭐⭐ |

### 6.3 推荐硬件路线图

```
Phase 1 (验证期):  ESP32-S3-SUPERMINI (当前硬件)   → 承载 55K 模型 (够用)
Phase 2 (量产期):  ESP32-S3 8MB/8MB 升级版         → 承载 80K 模型 + 更好裕量
Phase 3 (旗舰版):  MAX78000 或 ESP32-P4            → 承载 100K+ 模型, 功耗 <10mW
```

### 6.4 当前 ESP32-S3 的可行性分析

**结论: 当前 ESP32-S3-SUPERMINI (4MB/2MB) 足够承载 55K 模型，但需要优化 Tensor Arena 使用。**

| 资源 | 需求 | 可用 | 余量 | 状态 |
|------|------|------|------|------|
| Flash (模型+Runtime) | ~150 KB | ~1.5 MB (固件后) | ~90% | ✅ 充足 |
| SRAM (Arena+栈) | ~100 KB | ~1.5 MB (PSRAM) | ~93% | ✅ 充足 |
| 推理时间 | 8-12ms | 窗口 500ms (50% 重叠) | ~98% | ✅ 充足 |
| 温度 | +1-2°C (额外计算) | 阈值 65°C | 安全 | ✅ 安全 |

**关键优化措施:**
1. Tensor Arena 使用 PSRAM (ESP32-S3 支持 SPIRAM malloc)
2. 算子融合: Conv+BatchNorm+ReLU 在导出时合并
3. TFLite Micro 算子注册精简: 仅包含实际使用的算子
4. DMA 加持: 模型权重存储在 Flash，通过 MMU Cache 加速访问

---

## 七、三步走实施路线图

### Phase 1: 数据 + 训练强化 (预计 2-3 周)

```
任务:
  ✅ 下载并预处理 PTB-XL, CPSC 2018, INCART 数据集
  ✅ 统一滤波管线 (匹配 ESP32 板载 IIR 滤波器)
  ✅ 实现 Focal Loss + Label Smoothing
  ✅ 实现 ECG-ResNet-Lite 架构 (55K 参数)
  ✅ 5-Fold Cross-Validation 训练 + 评估
  ✅ 温和数据增强 (2x 扩充)
  ✅ Mixup 正则化

目标精度: ≥ 94% (测试集, 按病人分组)
交付物: best_model.h5, 训练日志, 混淆矩阵
```

### Phase 2: 量化 + 部署验证 (预计 1-2 周)

```
任务:
  ✅ QAT 量化感知训练 (最后 10 epochs)
  ✅ 结构化剪枝 → 35K 参数版本
  ✅ INT8 TFLite 导出 + PC 端推理验证
  ✅ C 头文件导出 (.h 模型权重数组)
  ✅ ESP32-S3 板载推理测试 (串口 CSV 输出)
  ✅ 功耗与温度测量 (对比强化前后)

目标精度: ≥ 95% (INT8 部署后)
交付物: ecg_model_qat.h, TFLite 模型, 板载测试报告
```

### Phase 3: 蒸馏 + 极限精度 (可选, 预计 2-3 周)

```
任务:
  ✅ 训练 Teacher 模型 (ResNet-18-1D, 1.1M 参数, PTB-XL 全量数据)
  ✅ 知识蒸馏 → Student (25K 或 55K 参数)
  ✅ 集成学习: 3 个不同种子模型的 Soft Voting
  ✅ 芯片选型评估报告 (ESP32-P4 / MAX78000)
  ✅ 真实临床场景测试 (采集自己或志愿者 ECG)

目标精度: ≥ 97%
交付物: distilled_model.h, 芯片对比报告, 实测精度
```

---

## 八、风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|----------|
| 55K 模型 INT8 量化精度塌缩 | 中 | 高 | 使用 QAT（量化感知训练）而非 PTQ, 或退回到 FP16 |
| 多数据集标签不一致 | 中 | 中 | 统一使用 AAMI EC57 映射, 人工抽样校验 200 条 |
| PTB-XL 下载/版权问题 | 低 | 高 | 使用 PhysioNet 官方渠道, 确认学术使用许可 (已开放) |
| Tensor Arena 超过 PSRAM 可用量 | 低 | 中 | Arena 使用分块计算 (每次仅加载一层权重), pring-arena 复用 |
| 训练耗时过长 | 中 | 低 | 使用 GPU (CUDA), 混合精度 FP16, 数据预处理预缓存为 TFRecord |
| 真实场景精度远低于测试集 | 高 | 高 | 持续采集自测数据反馈训练, 在线难例挖掘 (OHEM) |
| 芯片供应/价格波动 | 中 | 中 | 维持 ESP32-S3 为主方案, MAX78000 为可选升级方案 |

---

## 九、对比总结

| 维度 | 当前 (v2) | Phase 1 目标 | Phase 2 目标 | Phase 3 目标 |
|------|-----------|-------------|-------------|-------------|
| **数据集** | MIT-BIH 48条 (~87K) | 多库融合 (~300K) | 不变 | 全库 (~600K) |
| **模型架构** | 1D-CNN 3层 | ECG-ResNet-Lite 8层+SE | 剪枝后 35K | 蒸馏 Student 25K |
| **参数量** | 15K | 55K | 35K | 25K / 55K |
| **INT8 大小** | 24.8 KB | ~55 KB | ~35 KB | ~25 KB / ~55 KB |
| **训练策略** | CE + Adam + EarlyStop | FocalLoss + Mixup + Warmup+Cosine | QAT + 剪枝微调 | 知识蒸馏 |
| **测试精度 (FP32)** | 85.6% | ≥ 94% | ≥ 95% | ≥ 97% |
| **测试精度 (INT8)** | 88.1% | ≥ 93% | ≥ 95% | ≥ 96.5% |
| **AUC** | 0.942 | ≥ 0.97 | ≥ 0.98 | ≥ 0.99 |
| **异常召回率** | 88% | ≥ 93% | ≥ 95% | ≥ 97% |
| **异常精确率** | 77% | ≥ 88% | ≥ 90% | ≥ 93% |
| **推理延迟** | 5ms | 8-12ms | 6-9ms | 5-8ms |
| **芯片兼容性** | ESP32-S3 | ESP32-S3 | ESP32-S3 | ESP32-S3 / MAX78000 |

---

## 十、立即开始的行动项

1. **今天**: 下载 PTB-XL 数据集 (PhysioNet: `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3`)
2. **本周**: 在 `pc_tools/ecg_dl/` 下新建 `preprocess_ptbxl.py`, 实现统一预处理
3. **本周**: 新建 `models/resnet_lite_1d.py`, 实现 ECG-ResNet-Lite 架构
4. **本周**: 新建 `losses/focal_loss.py`, 实现 Focal Loss + Label Smoothing
5. **下周**: 整合训练 pipeline, 启动 Phase 1 实验
6. **两周后**: 评测 Phase 1 结果, 决定是否直接推进 Phase 2

---

> 📌 **核心思路总结**: 不换芯片 → 数据翻倍 → 模型翻倍 → 训练策略强化 → QAT 量化 → 目标 97%。
> 在 ESP32-S3 4MB/2MB 的约束下，55K 参数的 ECG-ResNet-Lite 是最佳平衡点。
> 若精度仍有 1-2% 缺口，通过知识蒸馏可将更小模型 (25K) 推至同等精度。



---

## 十一、实验执行记录（2026-07）

### 11.1 已完成的实验

| # | 日期 | 实验 | 模型 | 数据集 | Test AUC | Test Acc | 结论 |
|---|------|------|------|--------|----------|----------|------|
| 1 | 07-27 | 基线复现 | CNN v2 (15K) | MIT-BIH | 0.935 | 88.7% | 比原版 85.6% 有提升，确认 pipeline 正常 |
| 2 | 07-27 | ESP32 滤波匹配 | **CNN v2 (15K)** | **MIT-BIH (滤波)** | **0.954** | **88.5%** | ✅ 当前最优，AUC +0.019 |
| 3 | 07-27 | 扩容 CNN | CNN v3 (30K) | MIT-BIH | 0.936 | 86.6% | ❌ 过参数化，精度反降 |
| 4 | 07-27 | 残差网络 | ResNet-Lite S (25K) | MIT-BIH | 0.761 | 62.7% | ❌ 深度可分离卷积在小数据集上梯度崩溃，全预测 Normal |
| 5 | 07-27 | 残差网络 | ResNet-Lite M (55K) | MIT-BIH | 0.882 | 80.8% | ❌ 过拟合严重 |
| 6 | 07-27 | PTB-XL 合并 | CNN v2 | MIT-BIH+PTB-XL | 0.764 | 71.6% | ❌ PTB-XL 诊断标签不适用于心拍级训练 |
| 7 | 07-27 | PTB-XL 减采 | CNN v2 | MIT-BIH+PTB-XL(3beat) | 0.777 | 73.3% | ❌ 减至3beat仍无效 |
| 8 | 07-27 | SVDB 合并 | CNN v2 | MIT-BIH+SVDB | 0.951 | 88.2% | ⚠️ SVDB 异常心拍仅76个，不平衡加剧 |
| 9 | 07-27 | 类别加权 | CNN v2 (class_weight) | MIT-BIH | 0.934 | 88.7% | ⚠️ class_weight 与 tf.data.Dataset 不兼容，未生效 |
| 10 | 07-28 | FocalLoss | CNN v2 | MIT-BIH | 0.751 | 67.3% | ❌ FocalLoss 实现有 bug，模型退化为单类预测 |
| 11 | 07-28 | 低学习率长训练 | CNN v2 (lr=0.0005) | MIT-BIH | 0.954 | 88.5% | 与实验2结果一致，验证稳定性 |
| 12 | 07-27 | GPU环境搭建 | — | — | — | — | ✅ WSL2/TF2.21/RTX5070 成功，~7ms/step |

### 11.2 关键发现

#### ✅ 有效改进

1. **ESP32 滤波器匹配是最大单项提升**（AUC +0.019）
   - 原因：训练数据与板载推理的数据分布对齐
   - 代价：零（仅修改预处理脚本）
   - **建议：此改进应永久保留，所有后续实验都应使用滤波数据**

2. **CNN v2 (15K) 是 MIT-BIH 上的最优架构**
   - 比 v3 (30K) 好：避免过参数化
   - 比 ResNet-Lite 好：标准卷积比深度可分离卷积在小数据上稳定
   - **结论：MIT-BIH 87K 心拍的天花板约 89% Acc / 0.955 AUC**

3. **GPU 训练环境已就绪**
   - WSL2 + TF 2.21 + RTX 5070 可用
   - 需要 `--break-system-packages` 和 LD_LIBRARY_PATH 设置

#### ❌ 失败的尝试（重要经验）

4. **PTB-XL 不能直接用于心拍级训练**
   - 根因：PTB-XL 的 SCP 标签是**记录级诊断**（如"心肌梗死"），不是**心拍级标签**
   - 一条"心肌梗死"记录 ≠ 每个心拍都是异常心拍
   - 解决方案：仅使用节奏异常子集（AFIB/PVC/PAC 等），但需足够样本量

5. **深度可分离卷积在小数据集上不稳定**
   - ResNet-Lite Small (25K) 直接崩溃为单类预测
   - 根因：DepthwiseConv 梯度流弱，需大量数据支撑
   - 解决方案：坚持标准卷积，或先在大数据集预训练再迁移

6. **FocalLoss 当前实现有 bug**
   - 表现为模型输出完全坍塌（所有样本同一类）
   - 需修复后再测试，优先验证 LabelSmoothing 关闭时的行为

7. **class_weight 对 tf.data.Dataset 不生效**
   - Keras 3.x 的 `model.fit(class_weight=...)` 与 `tf.data.Dataset` 配合有问题
   - 替代方案：使用 `sample_weight` 或自定义训练循环

8. **SVDB 不适合作为独立数据源**
   - 78条记录中异常心拍极少（76个），大量正常心拍稀释了训练信号
   - 如果使用，需做降采样或仅提取异常片段

### 11.3 未来尝试（按优先级排序）

| 优先级 | 方向 | 预期收益 | 难度 | 说明 |
|--------|------|----------|------|------|
| 🔴 P0 | **MIT-BIH + INCART 合并** | +2-4% | 低 | INCART 75条记录，beat-level 标注，与 MIT-BIH 格式一致 |
| 🔴 P0 | **修复 FocalLoss** | +1-2% | 中 | 修复后配合 α=0.75, γ=1.0（降 gamma 避免梯度消失） |
| 🟡 P1 | **温和数据增强** | +1-3% | 低 | TimeWarp(±8%) + AmpScale(±15%)，≤2x扩充，已验证不破坏 ECG |
| 🟡 P1 | **阈值调优** | +2-3% | 低 | 训练后在 val 集搜索最佳分类阈值，优先提升异常召回率 |
| 🟡 P1 | **3-Seed Ensemble** | +2-3% | 中 | 训练3个不同seed的v2，SoftVoting，总参数量 45K 仍在 ESP32 范围内 |
| 🟢 P2 | **QAT 量化感知训练** | ±1% | 中 | 代替 PTQ，确保 INT8 部署精度可控 |
| 🟢 P2 | **多导联融合** | +2-4% | 高 | 用 Lead I + II 双通道，需修改模型输入层 |
| ⚪ P3 | **知识蒸馏** | +2-5% | 高 | Teacher(1.1M)→Student(25K)，需先有大模型达到高分 |
| ⚪ P3 | **CPSC 2018 数据集** | +3-5% | 高 | 中国人群 6877 条记录，需要 R-peak 检测标注 |

### 11.4 更新后的路线图

```
已完成 ✅
├── ESP32 滤波匹配 (AUC 0.935→0.954)
├── GPU 环境 (WSL2/RTX 5070)
├── 多数据集预处理管线 (PTB-XL, SVDB)
├── CNN v3 / ResNet-Lite 架构实验
└── FocalLoss / Mixup 代码实现

短期 (1-2周) 🔴
├── INCART 下载 + 预处理 + 合并训练     → 目标 AUC 0.96+
├── 修复 FocalLoss bug                  → 目标 AUC 0.96+
└── 温和数据增强 (TimeWarp + AmpScale)  → 目标 Acc 90%+

中期 (2-4周) 🟡
├── 阈值调优 + 3-Seed Ensemble          → 目标 Acc 93%+
└── QAT 量化感知训练                    → 确保 INT8 无损

长期 (Phase 3) ⚪
├── CPSC 2018 数据集 (6877条)           → 目标 Acc 95%+
└── 知识蒸馏 (Teacher→Student)          → 目标 Acc 97%
```

> ⚠️ **重要修正**: 原计划认为 ESP32-S3 能承载 55K 参数的 ResNet-Lite。实验证明 **标准卷积 15K 的 CNN v2 比深度可分离 25K 的 ResNet-Lite 表现好得多**。
> 当前最佳方案是 CNN v2 (15K, INT8 ~15KB) + Ensemble (3×15K=45K)，而不是单个大模型。

