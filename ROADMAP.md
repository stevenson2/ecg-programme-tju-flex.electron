# 心电 AI 模型路线图

> 从 Phase 1 (15K CNN, Acc 93.98%) → Phase 2 (SimCLR 预训练 + 600K CNN, 目标 Recall ≥88%)

---

## Phase 1: 已完成（MIT-BIH + INCART, FocalLoss）

### 最佳模型

| 指标 | 数值 |
|------|------|
| 模型 | CNN v2 (15K params, INT8 ~15KB) |
| 数据 | MIT-BIH (87K) + INCART (176K) = 263K beat级心拍 |
| Loss | FocalLoss (γ=1.0, α=0.75, bug已修复) |
| Acc | **93.98%** |
| AUC | **0.9716** |
| Abnormal Recall | 72% |
| Abnormal Precision | 84% |

### 9 项实验完整矩阵

| # | 实验 | Acc | AUC | A.Recall | 结论 |
|---|------|-----|-----|----------|------|
| 0 | 基线 MIT-BIH+CE | 88.50% | 0.9540 | 0.75 | - |
| 1 | FocalLoss Bug | 67.3% | - | - | ❌ alpha_t被label smoothing污染 |
| 3 | **FocalLoss修复** | 89.66% | 0.9549 | 0.83 | ✅ |
| 4 | +ECG1000 | 86.56% | 0.9237 | 0.58 | ❌ 记录级标签毒化 |
| 5 | **+INCART** 🏆 | **93.98%** | **0.9716** | 0.72 | ✅ 最大单项提升 |
| 6 | α=0.85 | 94.23% | 0.9716 | 0.69 | ❌ 牺牲Recall |
| 8 | PTBXL两阶段 | 94.49% | 0.9720 | 0.70 | ⚠️ Acc↑Recall↓ |
| 9 | PTBXL直接合并 | 崩溃 | 0.76 | - | ❌ 噪声标签 |

### 关键发现

1. **FocalLoss Bug**: `alpha_t` 使用了被 Label Smoothing 污染的软标签，Normal 权重膨胀 2.9x。修复：在平滑前保存 `y_true_hard`
2. **INCART**: +4.32% Acc 是最大单项提升
3. **记录级标签不可用**: ECG1000 / PTB-XL 的记录级诊断与 beat 级训练不兼容
4. **AUC 0.97 但 Recall 72%**: 模型能区分异常，但预测概率偏保守。需更大的模型容量和时序输入

### 改进贡献度

| 排名 | 改进 | Acc提升 | 状态 |
|------|------|---------|------|
| 🥇 | +INCART | +4.32% | ✅ |
| 🥈 | FocalLoss修复 | +2.87% | ✅ |
| ❌ | +PTBXL两阶段 | +0.51% | ⚠️ Recall↓ |
| ❌ | +PTBXL直接合并 | 崩溃 | 不可行 |

---

## Phase 2 Plan: SimCLR 对比学习 + 大模型

### 目标

| 指标 | Phase 1 | Phase 2 目标 |
|------|---------|-------------|
| Acc | 93.98% | ≥95% |
| AUC | 0.9716 | ≥0.98 |
| **Abnormal Recall** | **72%** | **≥88%** |
| 模型参数量 | 15K | ~600K |

### PTB-XL 的正确用法：自监督学习

```
Phase 1 失败做法: PTB-XL 记录级标签 → beat级监督 → 训练崩溃
Phase 2 正确做法: PTB-XL 信号波形 → 自监督对比学习 → 预训练编码器

┌──────────────────────────────────────────┐
│  同一 ECG 段 × 2个增强 → 嵌入向量相似    │
│  不同 ECG 段 → 嵌入向量不相似            │
│  → 模型自主学习 QRS/P波/T波形态          │
│  → 21K 条 PTB-XL 记录，无需任何标签！    │
└──────────────────────────────────────────┘
                    ↓ 权重迁移
┌──────────────────────────────────────────┐
│  MIT-BIH+INCART (263K beat级)            │
│  FocalLoss 精调 → 学习决策边界           │
└──────────────────────────────────────────┘
```

### 模型架构: ECG-CNN-M

```
输入: 3拍拼接 (750, 1)  ← 时序上下文（R-R间期）

Stem:   Conv1D(32, k15, s=2) → BN → ReLU         (375, 32)
Block1: Conv1D(64, k7) → BN → ReLU → MaxPool(2)  (187, 64)
Block2: Conv1D(128, k5) → BN → ReLU → MaxPool(2) (93, 128)
Block3: Conv1D(256, k5) → BN → ReLU → MaxPool(2) (46, 256)
Block4: Conv1D(256, k3) → BN → ReLU → GAP        (256)

对比学习: Projection Head Dense(128)→Dense(64)
分类:     Classifier Head Dense(128)→Dropout→Dense(2)

参数量: ~600K | INT8: ~600KB | SRAM: ~250KB | 推理: ~15ms
```

### 三个核心改进 vs Phase 1

| 改进 | Phase 1 | Phase 2 | 原因 |
|------|---------|---------|------|
| 模型容量 | 15K | 600K | 263K数据可支撑更大模型 |
| 时序输入 | 单拍250点 | 3拍750点 | PAC形态与Normal相同，区别在R-R间期 |
| PTB-XL利用 | 错误当标签用 | 自监督信号 | 21K条波形是无标签教材 |

### 实施步骤

| 步骤 | 文件 | 内容 |
|------|------|------|
| 1 | `models/cnn_m.py` | Encoder + 双头架构 |
| 2 | `data/preprocess_3beat.py` | 单拍→3拍序列拼接 |
| 3 | `losses/contrastive.py` | NT-Xent 对比损失 + ECG增强 |
| 4 | `train_ssl.py` | Stage1: PTB-XL预训练(~3h GPU) + Stage2: INCART精调(~30min) |

### 指标含义速查

| 指标 | 含义 | 心电监测优先级 |
|------|------|--------------|
| Accuracy | 总体正确率 | 低（Normal占比高虚高） |
| AUC | 模型底层判别力 | 中 |
| Precision | 报警中有多少真异常 | 中 |
| **Recall** | 真异常中抓到多少 | **🔴 最高（漏报致命）** |
