# 4.4-4 患者级划分 — 进度存档 (2026-08-01 完成)

> 目标: 论文级严谨性。历史模型按记录/拍级划分评估, 存在数据泄漏,
> 需在患者级划分下重新评估全部历史模型, 确认排名是否变化。

## 一、已完成事项

### 1. 数据完整性审计与修复
| 数据集 | 预期 | 修复前 | 修复后 | 说明 |
|--------|------|--------|--------|------|
| MIT-BIH | 48 记录/47 患者 | **37 条** | ✅ 48 条 | 13 条 .dat 截断, 已重新下载 |
| INCART | 75 记录/32 患者 | 75 条 | ✅ 75 条 | 完整 (32 患者/75 记录, 记录级划分泄漏严重) |
| PTB | 549 记录/290 患者 | 533 条 | 533 条 | 16 条信号质量差 (XQRS 0 峰), 论文中如实报告排除 |

### 2. 重新预处理 (完成)
- MIT-BIH: **658,962 拍** (N=543,786 / A=115,176), 含 6 倍增强
- PTB: **69,482 拍** (N=10,416 / A=59,066)
- 3-beat: **834,495 序列** (MIT 48 条 + INCART)

### 3. 患者级划分模块
- `data/patient_split.py`: MIT 47 患者 / INCART 32 患者 / PTB 290 患者
- seed=42, 患者级 60/20/20, **train∩test 患者 = 空集 (已验证)**

### 4. 全模型患者级评估 (完成, 含修正与多阈值)
- `eval_patient_split_all.py`: 13 模型 × MIT/PTB 双域 × θ∈{0.35,0.5,0.65,0.8}
- 结果: `models/patient_split_eval.json`

## 二、蹊跷点核查结论 (8 项)

| # | 蹊跷点 | 结论 | 证据 |
|---|--------|------|------|
| 1 | exp4 与 ResNet-L(focal) PTB 指标逐位相同 | **exp4-final 是 exp3 副本** (sha256 相同); 真 exp4 = best 检查点; "ResNet-L(v2)" 实为 ResNet-M 架构 (158K 参数, 标注错误) | `compare_model_weights.py` h5py 逐层对比 |
| 2 | 历史 37 条 MIT npz 生成配置 | **旧 npz 不含增强拍** (中位 2,260 拍/记录); 新 48 条含 6 倍增强; config 904f6d5 时代 `noise_std=[0.02]`(list) → 9777076 起 `0.015`(float) | `verify_split_consistency.py` + git 考古 |
| 3 | seed42 choice vs permutation 语义 | **不一致**, 交集仅 12/58 — 历史 eval 脚本与 patient_split 的 PTB 测试集几乎完全不同 | `verify_split_consistency.py` |
| 4 | MIT 201/202 同患者 | 推断成立 (官方"48 records from 47 subjects"), 论文引用需谨慎 | 官方 Directory |
| 5 | 单阈值 0.5 不公平 | **已补多阈值** θ∈{0.35,0.5,0.65,0.8} | `eval_patient_split_all.py` |
| 6 | 训练 1s vs 部署 0.5s 窗口不匹配 | **坐实**: 固件 500Hz, 部署 250 点=0.5s, 训练 250 点=1s, 2 倍时间尺度失配 | ecg-expert 全链路只读分析 |
| 7 | PTB 3-beat test 13,322 > 13,058 | stitch_3beat 丢首尾拍致 14 记录消失、6 患者丢失 (286→280), permutation 输入变化 → 测试患者漂移 (交集 17/57) | `verify_split_consistency.py` |
| 8 | 历史评估泄漏 | exp5 PTB 训练 seed42 全患者抽拍 ~17% 见过; MIT 域 INCART 记录级泄漏 | 已确认, 论文表述用 |

## 三、修复措施

1. **模型清单修正** (`eval_patient_split_all.py`):
   - exp4 → `best_resnet_large_exp4_ptb.h5` (final 是 exp3 副本)
   - "ResNet-L(v2)" → "ResNet-M(存档v2)" (实为 158K 参数 ResNet-M)
   - 多任务模型输出处理 (取分类头, 非 `[:,1]`)
2. **陷阱脚本修复** (5 个历史 eval 脚本):
   - `best_resnet_large.h5` (现=exp6 权重) → `best_resnet_large_exp5_ptb_capped.h5`
   - 涉及: eval_ptb_holdout / eval_deploy_decision / eval_expert_combo / eval_fusion / eval_weighted_fusion
3. **患者级训练路径** (`dataset.py` + `train.py`):
   - 新增 `--patient-split` CLI: MIT+INCART 患者级划分 + PTB 训练拍仅取 train 患者
   - 冒烟测试通过: 79 患者 (49/15/15), PTB 172/286 train 患者 41,730 拍

## 四、患者级评估结果 (修正后, 多阈值)

### MIT 域 (患者级 test: 163,078 拍)
| 模型 | AUC | R@0.5 | P@0.5 | F1@0.5 | R@0.35 |
|------|:---:|:---:|:---:|:---:|:---:|
| CNN-M (750点) | **0.982** | 0.717 | 0.897 | 0.797 | 0.817 |
| P2A (部署) | **0.974** | 0.901 | 0.699 | 0.788 | 0.914 |
| 多任务 | 0.968 | 0.891 | 0.610 | 0.724 | 0.915 |
| ResNet-M / 存档v2 | 0.971 | 0.901 | 0.624 | 0.737 | 0.914 |
| SSL 微调 | 0.962 | 0.906 | 0.557 | 0.690 | 0.920 |
| ResNet-L(focal) | 0.959 | 0.885 | 0.548 | 0.677 | 0.915 |
| Ensemble(seed42) | 0.959 | 0.898 | 0.541 | 0.676 | 0.919 |
| exp6 (域平衡) | 0.942 | 0.918 | 0.399 | 0.556 | 0.928 |
| ResNet-S(v3) | 0.890 | 0.965 | 0.237 | 0.381 | 0.978 |
| CNN-v2 | 0.885 | 0.687 | 0.419 | 0.520 | 0.832 |
| **exp5 (PTB限量)** | **0.841** | 0.945 | 0.202 | 0.333 | 0.956 |
| exp4 (ptb全量, 真权重) | 0.822 | 0.850 | 0.228 | 0.359 | 0.877 |

### PTB 域 (患者级 test: 13,058 拍)
| 模型 | AUC | R@0.5 | P@0.5 | F1@0.5 | R@0.35 | 说明 |
|------|:---:|:---:|:---:|:---:|:---:|------|
| exp5 (见过PTB) | **0.994** | 0.939 | 0.993 | 0.965 | 0.959 | 域内 (训练侧有泄漏) |
| exp6 (见过PTB) | 0.990 | 0.614 | 1.000 | 0.761 | 0.646 | 域内 |
| exp4 (真权重, 见过PTB) | 0.945 | 0.779 | 0.969 | 0.864 | 0.808 | 域内 (修正后) |
| ResNet-L(focal) | 0.791 | 0.247 | 0.952 | 0.393 | 0.297 | 跨域基线 |
| SSL 微调 | 0.773 | 0.184 | 0.978 | 0.310 | 0.221 | 跨域 |
| ResNet-M / 存档v2 | 0.770 | 0.198 | 0.938 | 0.327 | 0.225 | 跨域 |
| 多任务 | 0.769 | 0.233 | 1.000 | 0.378 | 0.265 | 跨域 |
| Ensemble(seed42) | 0.761 | 0.307 | 0.984 | 0.468 | 0.354 | 跨域 |
| P2A (未见PTB) | 0.750 | 0.255 | 0.972 | 0.404 | 0.285 | 跨域 |
| CNN-v2 | 0.645 | 0.317 | 0.937 | 0.474 | 0.374 | 跨域 |
| ResNet-S(v3) | 0.660 | 0.579 | 0.886 | 0.700 | 0.665 | 跨域 |
| CNN-M | 0.617 | 0.084 | 0.940 | 0.155 | 0.117 | 跨域 |

## 五、关键结论

1. **历史排名基本可信**: MIT 域领先者 (CNN-M 0.982 / P2A 0.974) 与历史一致
2. **exp4 修正后 PTB 能力显现**: AUC 0.945 / R 0.779 (原误用 exp3 副本时仅 0.791/0.247)
3. **ResNet-M vs 存档v2**: 显示精度内指标一致但权重独立 (同配方邻近最优点 + 舍入)
4. **双专家 OR 融合方案依然成立** (P2A + exp5)
5. **部署窗口缺陷** (蹊跷点6): 固件 500Hz, 部署 0.5s vs 训练 1s, 属 4.2/4.3 缺陷, 择期修复 (方案A: push 前 2:1 抽取)

## 六、Limitations (论文表述用)

- 测试集含 6 倍增强拍 (与原始拍高度相关, 有效独立样本数被高估)
- PTB 域 exp5/exp6/exp4 为"域内评估" (训练见过 PTB), 非跨域验证
- 3-beat PTB test (13,322 拍) 与 250 点 test (13,058 拍) 测试患者集合不同
- 16 条 PTB 坏记录排除标准需如实报告

## 七、待办

- [x] **重训 exp4/exp5/exp6 (患者级清洁)**: 分别使用 `--ptb-abn-max 100000`、
       `--ptb-abn-max 10000`、`--ptb-abn-max 10000 --domain-balanced`，均启用
       `--patient-split`；完成最终双域评估。
- [x] 部署窗口缺陷修复 (方案A: 固件 2:1 抽取已实施, `pio run` 通过; PC 部署一致性配对验证
       (64,941 raw 拍) 发现更深一层 filtfilt-vs-因果链失配, PTB 域 ΔAUC 最高 −0.105,
       决策: 部署链重建训练数据并重训 → 见 TUNING_HISTORY 第十三章)
- [ ] 部署链重训 exp4/5/6 (阶段 1.5: exp6 配置试点先行, 数据 `*_deploy.npz`)
- [ ] 论文撰写支撑数据整理

## 八、患者级清洁重训结果 (2026-08-02)

### 训练归档

| 实验 | 最佳权重 | 最终权重 | 训练历史 |
|------|----------|----------|----------|
| exp4 | `best_resnet_large_exp4_patient_clean.h5` | `final_resnet_l_exp4_patient_clean.h5` | `train_history_exp4_patient_clean.csv` |
| exp5 | `best_resnet_large_exp5_patient_clean.h5` | `final_resnet_l_exp5_patient_clean.h5` | `train_history_exp5_patient_clean.csv` |
| exp6 | `best_resnet_large_exp6_patient_clean.h5` | `final_resnet_l_exp6_patient_clean.h5` | `train_history_exp6_patient_clean.csv` |

### 双域结果 (阈值 0.5, MIT 域为 6× 增强测试口径)

> ⚠️ **口径说明 (T1-2 后)**：本表 MIT 域数字为**增强测试口径**（2026-08-02 评估时代）。
> 2026-08-05 T1-2 已用未增强原始拍重评并切换为主结果口径——exp5 MIT-AUC 0.8874 → **0.9295**、
> exp6 0.8245 → 0.8942、P2A 0.9740 → 0.9878（权威数字见 `docs/FINAL_RESULTS.md` 表2，
> 增强版条目保留为对照）。

| 模型 | MIT-AUC | MIT-R | MIT-P | PTB-AUC | PTB-R | PTB-P |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| exp4 (患者级清洁) | 0.8669 | 0.858 | 0.428 | 0.7319 | 0.536 | 0.927 |
| exp5 (患者级清洁) | 0.8874 | 0.902 | 0.358 | 0.7845 | 0.628 | 0.909 |
| exp6 (患者级清洁) | 0.8245 | 0.894 | 0.272 | 0.8232 | 0.702 | 0.926 |

与历史 exp5 (MIT-AUC 0.8405 / PTB-AUC 0.9939) 对比，清洁评估不再包含测试患者训练泄漏；
清洁 exp6 的 PTB AUC 最高，但清洁 exp5 的 MIT AUC 最高。完整 15 模型、多阈值结果见
`models/patient_split_eval.json`。

## 九、相关文件索引
- 划分模块: `data/patient_split.py`
- 全模型评估: `eval_patient_split_all.py` (含多阈值)
- 评估结果: `models/patient_split_eval.json`
- 验证脚本: `verify_split_consistency.py` (蹊跷点 2/3/7)
- 权重对比: `compare_model_weights.py` (蹊跷点 1)
- 冒烟测试: `smoke_patient_split.py` (患者级训练路径)
- 数据备份: `data/processed/mit_bih_processed_37rec_backup.npz`
