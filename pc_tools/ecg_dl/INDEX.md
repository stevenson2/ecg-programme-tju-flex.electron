# pc_tools/ecg_dl/INDEX.md — 脚本编目（R17-CLEANUP P4 建立）

> 状态：**active** = 主线管线；**round** = 历史实验轮（保留）；
> **archived** = 一次性脚本/日志（`archive/`，不再维护）。
> 参考数据库路径统一走 `ecg_refs.py`（project_paths `ecg_refs_*`，
> Windows `C:\ecg_data`）；禁止在脚本里写死库路径。

## 工具

| 脚本 | 用途 | 状态 |
|---|---|---|
| `ecg_refs.py` | 统一资料库路径助手（project_paths ecg_refs_*） | active |
| `README.md` | ecg_dl 管线说明（历史） | active |
| `config.py` | 管线共享配置 | active |

## 评测

| 脚本 | 用途 | 状态 |
|---|---|---|
| `eval_aami_3beat.py` | eval_aami_3beat.py — AAMI-class recall breakdown for the 3-beat (750pt) model. | active |
| `eval_aami_breakdown.py` | AAMI-category recall breakdown for the patient-split evaluation. | active |
| `eval_aami_matrix.py` | eval_aami_matrix.py — 批量 AAMI 逐类矩阵评估（患者级 + 部署链） | active |
| `eval_aami_matrix_noaug.py` | eval_aami_matrix_noaug.py — AAMI 矩阵 noaug 口径（部署链，患者级） | active |
| `eval_alarm_decision.py` | eval_alarm_decision.py — T1-4: 报警决策层 (误报率指标 + 融合粒度 + 三态输出) | active |
| `eval_binary_all.py` | 统一二分类评估: 所有关键模型在患者级+部署链(D3)口径下的完整指标矩阵 | active |
| `eval_bootstrap_ci.py` | eval_bootstrap_ci.py — T3-6/M8: 主结果患者级 bootstrap 95% CI | active |
| `eval_capacity_probe.py` | eval_capacity_probe.py — 容量/架构探针评估: 门判定 + fc1 嵌入分离度 (TH §109) | active |
| `eval_clean_baseline.py` | eval_clean_baseline.py — 干净基线模型的患者级测试集评估 (TH §99) | active |
| `eval_clean_baseline_v3.py` | eval_clean_baseline_v3.py — clean_baseline_v3 模型的患者级测试集评估 (TH §100) | active |
| `eval_clean_baseline_v4.py` | eval_clean_baseline_v4.py — v4 硬负例模型的患者级测试集评估 (TH §103) | active |
| `eval_clean_baseline_v5.py` | eval_clean_baseline_v5.py — v5 SVDB 扩训模型的患者级测试集评估 (TH §104) | active |
| `eval_clean_test.py` | eval_clean_test.py — 干净测试集重评：exp7c 家族全模型双口径记分表 | active |
| `eval_context_fusion.py` | eval_context_fusion.py — 融合决策器实验: 拍级 CNN 概率 + RR 上下文 → 事件级评估 | active |
| `eval_cross_arch.py` | eval_cross_arch.py — 跨架构部署链失配对照评估 | active |
| `eval_deploy_compensation.py` | eval_deploy_compensation.py — T1-3: 部署链失配分量消融 + 输入侧补偿原型 (v2) | active |
| `eval_deploy_decision.py` | exp5 vs P2A: MIT测试集 + PTB独立测试的多阈值工作点 | active |
| `eval_deploy_match.py` | eval_deploy_match.py — Deployment-identical ECG Evaluation Harness | active |
| `eval_distill_pilot.py` | eval_distill_pilot.py — 蒸馏试点 val 侧预注册门 + 双模型同协议评估 (TH §106) | active |
| `eval_distill_pilot_models.py` | eval_distill_pilot_models.py -- evaluate one or more Stage-2 distill-pilot | active |
| `eval_dual_expert_triage.py` | eval_dual_expert_triage.py — 双专家分诊 PC 模拟（A2） | active |
| `eval_event_clean_v6_val_test.py` | eval_event_clean_v6_val_test.py — 无泄漏 v6 事件级验证/测试 | active |
| `eval_event_exp7c_cooldowns.py` | eval_event_exp7c_cooldowns.py — v3b θ=0.85 下扫描 alert_cooldown | active |
| `eval_event_exp7c_sweep_high_causal.py` | eval_event_exp7c_sweep_high_causal.py — exp7c INT8 vs ECGFounder v3b QAT INT8 事件级 1-of-5 | active |
| `eval_event_grid_from_cache.py` | eval_event_grid_from_cache.py — 使用缓存概率细扫 θ × alert_cooldown | active |
| `eval_event_v3b_cooldowns.py` | eval_event_v3b_cooldowns.py — v3b θ=0.85 下扫描 alert_cooldown | active |
| `eval_event_v3b_sweep.py` | eval_event_v3b_sweep.py — exp7c INT8 vs ECGFounder v3b QAT INT8 事件级 1-of-5 | active |
| `eval_event_v3b_sweep_high.py` | eval_event_v3b_sweep_high.py — exp7c INT8 vs ECGFounder v3b QAT INT8 事件级 1-of-5 | active |
| `eval_event_v3b_sweep_high_causal.py` | eval_event_v3b_sweep_high_causal.py — exp7c INT8 vs ECGFounder v3b QAT INT8 事件级 1-of-5 | active |
| `eval_event_v3b_sweep_max_causal.py` | eval_event_v3b_sweep_max_causal.py — exp7c INT8 vs ECGFounder v3b QAT INT8 事件级 1-of-5 | active |
| `eval_event_v3b_vs_exp7c.py` | eval_event_v3b_vs_exp7c.py — exp7c INT8 vs ECGFounder v3b QAT INT8 事件级 1-of-5 | active |
| `eval_event_v4_sweep_high_causal.py` | eval_event_v4_sweep_high_causal.py — exp7c INT8 vs ECGFounder v3b QAT INT8 事件级 1-of-5 | active |
| `eval_event_v5_sweep_max_causal.py` | eval_event_v5_sweep_max_causal.py — exp7c INT8 vs ECGFounder v3b QAT INT8 事件级 1-of-5 | active |
| `eval_exp6_deploy.py` | eval_exp6_deploy.py — exp6 deploy-chain retrain evaluation | active |
| `eval_exp7.py` | eval_exp7.py — P0-2 exp7 修正后因果链口径评估 + 阈值重校准 | active |
| `eval_exp7b.py` | eval_exp7b.py — P0-2 exp7b (lr warmup 重训) 修正后因果链口径评估 | active |
| `eval_exp7c.py` | eval_exp7c.py — exp7c (真实数据微调) 修正后因果链口径 MIT/PTB AUC 防回归评估 | active |
| `eval_exp7c_policy_sweep.py` | eval_exp7c_policy_sweep.py — exp7c 部署报警策略扫描（θ × K-of-N） | active |
| `eval_exp7c_v4.py` | eval_exp7c_v4.py — exp7c_v4 联合验收评估 | active |
| `eval_expert_combo.py` | 双专家组合验证 (方案 B 细化): | active |
| `eval_expert_combo_patient.py` | 双专家 OR 严谨口径实测 (4.2-0) | active |
| `eval_final_models.py` | 最终模型评估: exp2'(CE+滑窗dup2) vs exp3(FocalLoss) vs P2A部署模型 | active |
| `eval_fusion.py` | 双模型融合验证 (方案 B): | active |
| `eval_gate_model.py` | eval_gate_model.py — A1 关卡验收（患者级、部署链） | active |
| `eval_gate_model_3class.py` | eval_gate_model_3class.py — 三分类关卡测试混淆矩阵 | active |
| `eval_int8_ecgfounder_v3.py` | eval_int8_ecgfounder_v3.py — ECGFounder v3 QAT INT8 评估 | active |
| `eval_int8_nosoftmax.py` | eval_int8_nosoftmax.py — T3-6/M3: INT8 去 softmax 对照 (量化误差 vs softmax 压缩分离) | active |
| `eval_patient_split_all.py` | 历史模型患者级重评估 (4.4-4) | active |
| `eval_patient_split_noaug.py` | eval_patient_split_noaug.py — T1-2: 未增强测试拍患者级重评 | active |
| `eval_phase_robustness.py` | eval_phase_robustness.py — T2-5: 测试时偏移敏感性曲线 + 相位鲁棒性指标 | active |
| `eval_ptb_holdout.py` | PTB 独立测试: exp5(见过PTB) vs P2A(未见过PTB) 在 PTB 拍上的检测能力 | active |
| `eval_ptbxl_mi_subtypes.py` | eval_ptbxl_mi_subtypes.py — PTB-XL MI 亚类分层（离线复用已存记录级分数） | active |
| `eval_ptbxl_mi_vs_abnormal.py` | eval_ptbxl_mi_vs_abnormal.py — 阴性集扩充：MI vs 其他异常（STTC/CD/HYP） | active |
| `eval_ptbxl_record_level.py` | eval_ptbxl_record_level.py — PTB-XL 记录级验证（完全复刻板上部署链） | active |
| `eval_ptbxl_record_level_lead_scan.py` | eval_ptbxl_record_level_lead_scan.py — 12 导联逐导扫描汇总（离线） | active |
| `eval_rhythm_af.py` | eval_rhythm_af.py — T4-8: 模块1(心律安全逻辑) + 模块3(AF RR 不规则度检测) | active |
| `eval_rhythm_af_ptbxl.py` | eval_rhythm_af_ptbxl.py — 下一步待办#5: AF 短窗验证 (PTB-XL 10s 节律标签) | active |
| `eval_sliding_experiment.py` | 统一评估: 滑窗增强实验三模型 + 既有部署模型, 含 θ=0.5/0.35 | active |
| `eval_st_morphology.py` | eval_st_morphology.py — 下一步待办#6: ST 形态学预研（模块 4 地基, 不入主结果） | active |
| `eval_triage_gate.py` | 分诊式设计验证 (8.9): 正常拍关卡 + 双专家 | active |
| `eval_v6_constrained_calibrated.py` | eval_v6_constrained_calibrated.py — v6 约束选参 + 校准重扫 | active |
| `eval_vf_detect.py` | eval_vf_detect.py — T4-9: VF/VT 检测器 (ZCR + 特征, 固定阈值, 独立测试 v2) | active |
| `eval_vf_detect_ablation.py` | VF 特征变体消融: 定位 CUDB Se 下降来源 (2026-08-16) | active |
| `eval_vf_detect_ablation2.py` | VF 消融第二组 + AFE 增益估计 (2026-08-16) | active |
| `eval_vf_detect_v2.py` | eval_vf_detect_v2.py — VF/VT 检测器 v2: 固件逐位复刻 + AFE mV 尺度校准 (2026-08-16) | active |
| `eval_weighted_fusion.py` | 加权分数融合: P = α·P2A + (1-α)·exp5 | active |
| `eval_binary_all.sh` | set -e | active |

## 训练

| 脚本 | 用途 | 状态 |
|---|---|---|
| `train_capacity_probe.py` | train_capacity_probe.py — 容量/架构探针 Stage 2: 从零重训更大架构 (TH §109) | active |
| `train_clean_baseline.py` | train_clean_baseline.py — 从零重训干净基线 (clean-split baseline, TH §99) | active |
| `train_clean_baseline_v3.py` | train_clean_baseline_v3.py — 配比再平衡从零重训 (TH §100) | active |
| `train_clean_baseline_v4.py` | train_clean_baseline_v4.py — v3 A 配方 + 训练划分内硬负例从零重训 (TH §103) | active |
| `train_clean_baseline_v5.py` | train_clean_baseline_v5.py — v3 A 配方 + SVDB 扩训从零重训 (TH §104) | active |
| `train_cls_head.py` | Phase 2C+: SSL Encoder Classification Head Fine-tuning | active |
| `train_cross_arch.py` | train_cross_arch.py — 跨架构部署链失配对照：训练外部模型 | active |
| `train_distill_pilot.py` | train_distill_pilot.py — 蒸馏试点 Stage 2: KD 重训 (TH §106) | active |
| `train_distill_pilot_seed.py` | train_distill_pilot_seed.py -- Stage 2 multi-seed / variant training for TH 106+. | active |
| `train_ensemble.py` | Phase 2C: 3-Seed Ensemble Training for ECG Anomaly Detection | active |
| `train_gate_model.py` | train_gate_model.py — 双专家前置关卡训练（A1） | active |
| `train_gate_model_3class.py` | train_gate_model_3class.py — 三分类关卡训练（A1 备选） | active |
| `train_kd.py` | Knowledge-Distillation (KD) training for ECG-ResNet-Lite-Large. | active |
| `train_mixed_balanced.py` | Balanced-Mixed Single-Model training for ECG-ResNet-Lite-Large. | active |
| `train_multitask.py` | Route F: Multi-task Learning Training Script. | active |
| `train_normfix_probe.py` | train_normfix_probe.py — §105 Phase A-2 归一化修复探针训练 | active |
| `train_pretrain.py` | Route K Stage 1: PTB-XL Supervised Pretraining (5-superclass multi-label). | active |
| `train_ssl.py` | Phase 2C: SimCLR Self-Supervised Pre-training for ECG | active |
| `train_two_stage.py` | Two-stage training: PTB-XL pretrain -> MIT-BIH+INCART finetune | active |
| `train_v6_recipe.py` | train_v6_recipe.py -- v6a fine-tune recipe. | active |

## 运行器(run_*.sh 等)

| 脚本 | 用途 | 状态 |
|---|---|---|
| `run_3beat_deploy.sh` | set -e | active |
| `run_aami_breakdown.sh` | set -e | active |
| `run_bal_mixed.sh` | set -e | active |
| `run_clean_ab_3beat.sh` | set -e | active |
| `run_clean_ab_single.sh` | set -e | active |
| `run_cross_arch_all.sh` | set -e | active |
| `run_eval_aami_3beat.sh` | set -e | active |
| `run_exp4_patient_clean.sh` | set -e | active |
| `run_exp5_patient_clean.sh` | set -e | active |
| `run_exp6_deploy.sh` | set -e | active |
| `run_exp6_hp005.sh` | set -e | active |
| `run_exp6_phase.sh` | set -e | active |
| `run_exp6_sgd.sh` | set -e | active |
| `run_exp7.sh` | set -e | active |
| `run_exp7b.sh` | set -e | active |
| `run_kd_screen.sh` | set -uo pipefail | active |
| `run_retest_ai.sh` | cd '/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_t | active |

## 绘图

| 脚本 | 用途 | 状态 |
|---|---|---|
| `plot_clean_reeval.py` | plot_clean_reeval.py — §98 双口径重评可视化 | active |
| `plot_hard_normal_clean_samples.py` | plot_hard_normal_clean_samples.py -- preview clean hard-normal samples. | active |
| `plot_hard_normal_noincart_samples.py` | plot_hard_normal_noincart_samples.py -- preview hard-normal samples after | active |
| `plot_hard_normal_samples.py` | plot_hard_normal_samples.py -- preview sampled hard-normal beats. | active |
| `plot_history.py` | Plot training history from Keras CSVLogger output. | active |
| `plot_incart_all_lead2_montage.py` | plot_incart_all_lead2_montage.py -- visual montage of filtered Lead II for | active |
| `plot_incart_annotation_alignment.py` | plot_incart_annotation_alignment.py -- diagnose INCART beat segmentation. | active |
| `plot_incart_annotation_refined.py` | plot_incart_annotation_refined.py -- compare raw annotation positions, | active |
| `plot_incart_leads.py` | plot_incart_leads.py -- visualize INCART raw multi-lead ECG for lead | active |
| `plot_incart_leads_filtered.py` | plot_incart_leads_filtered.py -- filtered INCART limb-lead ECG preview. | active |
| `plot_ssl_history.py` | Plot SSL training history from CSV log. | active |

## 报告生成

| 脚本 | 用途 | 状态 |
|---|---|---|
| `write_capacity_probe_prereg.py` | write_capacity_probe_prereg.py — 容量/架构探针预注册 (TH §109) | active |
| `write_distill_pilot_prereg_v2.py` | write_distill_pilot_prereg_v2.py -- write the pre-registration file for the | active |
| `write_v6_recipe_prereg.py` | write_v6_recipe_prereg.py -- pre-register v6a fast recipe. | active |
| `write_v6a2_prereg.py` | write_v6a2_prereg.py -- preregister v6a2 low-LR + original-train anchor. | active |
| `write_v6a3_prereg.py` | write_v6a3_prereg.py -- preregister targeted hard-normal fine-tune. | active |
| `write_v6a4_prereg.py` | write_v6a4_prereg.py -- preregister v6a4 MIT/INCART-only hard-normal fine-tune. | active |
| `write_v6a5_prereg.py` | write_v6a5_prereg.py -- preregister v6a5 small-dose clean hard-normal fine-tune. | active |
| `write_v6b_prereg.py` | write_v6b_prereg.py -- preregister v6b from-scratch on the 100k mined pool. | active |
| `write_v6c_prereg.py` | write_v6c_prereg.py -- preregister v6c MIT/INCART-only balanced from-scratch. | active |
| `write_vfdb_calibration_prereg.py` | write_vfdb_calibration_prereg.py -- pre-registration for the VFDB/CUDB | active |

## 数据集构建

| 脚本 | 用途 | 状态 |
|---|---|---|
| `build_3beat_deploy.py` | Build deploy-chain 3-beat dataset (mit_incart_3beat_deploy.npz). | active |
| `build_deploy_npz.py` | build_deploy_npz.py — 部署链 (D3) 训练数据重建 (阶段 1.5, TUNING_HISTORY 十三章) | active |
| `build_noaug_testset.py` | build_noaug_testset.py — T1-2: 重建未增强 MIT-BIH 测试拍 (训练数据不变) | active |
| `build_normfix_arrays.py` | build_normfix_arrays.py — §105 Phase A-1: MIT+INCART normfix 数组重建 | active |
| `build_v6a3_dataset.py` | build_v6a3_dataset.py -- v6a3 dataset: original v3-A train + 50k hard normal. | active |
| `build_v6a4_dataset.py` | build_v6a4_dataset.py -- v6a4 dataset: original v3-A train + MIT/INCART hard normal only. | active |
| `build_v6a5_dataset.py` | build_v6a5_dataset.py -- v6a5 small-dose clean hard-normal dataset. | active |
| `build_v6a6_dataset.py` | build_v6a6_dataset.py -- v6a6 dataset: original v3-A train + 10k clean | active |
| `build_v6a_dataset.py` | build_v6a_dataset.py -- build the 100k balanced v6a fine-tune dataset. | active |
| `build_v6c_dataset.py` | build_v6c_dataset.py -- v6c: MIT/INCART-only balanced 32k dataset. | active |

## 验证

| 脚本 | 用途 | 状态 |
|---|---|---|
| `verify_exp6_sgd_int8.py` | verify_exp6_sgd_int8.py — T0-1 验证: INT8 导出模型的部署链评估 vs retrain_exp6_sgd_eval.json (FP32 D3) | active |
| `verify_fw_ai_hp_coeffs.py` | verify_fw_ai_hp_coeffs.py — P0-2 Task 2: 固件 AI_HP_* 系数与训练侧因果链一致性验证 | active |
| `verify_heartrate_ludb.py` | verify_heartrate_ludb.py — LUDB 金标准验证固件 Pan-Tompkins 心率算法 | active |
| `verify_heartrate_ludb_v5.py` | verify_heartrate_ludb_v5.py — LUDB 金标准验证固件能量包络心率算法 (v5, 2026-08-14) | active |
| `verify_heartrate_ludb_v6.py` | verify_heartrate_ludb_v6.py — LUDB 心率 v6 参数扫描 (完整状态机仿真) | active |
| `verify_ptb_normal_only.py` | Phase 3B 验证: 只加 PTB 正常拍 (10.4K) 进训练集, 3-epoch 冒烟 | active |
| `verify_rr_feature.py` | 可行性验证 (2026-08-03): 单拍 + RR 间期特征 对 SVEB/整体 Recall 的提升 | active |
| `verify_split_consistency.py` | 4.4-4 患者级划分一致性验证 (蹊跷点 2/3/7) | active |
| `verify_waveform_hypothesis.py` | verify_waveform_hypothesis.py — 验证"心律失常是否发生在完整心电波形上" | active |

## 微调

| 脚本 | 用途 | 状态 |
|---|---|---|
| `finetune_exp7c.py` | finetune_exp7c.py — exp7b 真实数据域微调 (TH §40 B 方案) | active |
| `finetune_exp7c_ecgfounder.py` | finetune_exp7c_ecgfounder.py — exp7c + ECGFounder 代理硬负样本弱标签微调 | active |
| `finetune_exp7c_ecgfounder_v2.py` | finetune_exp7c_ecgfounder_v2.py — exp7c + ECGFounder 代理硬负样本弱标签微调 v2 (低权重) | active |
| `finetune_exp7c_ecgfounder_v3.py` | finetune_exp7c_ecgfounder_v3.py — 仅加入 ECGFounder 筛选的真实相似正常拍 | active |
| `finetune_exp7c_ecgfounder_v4.py` | finetune_exp7c_ecgfounder_v4.py — 仅加入 ECGFounder 筛选的真实相似正常拍 v4 (top50, 更低权重) | active |
| `finetune_exp7c_hardneg.py` | finetune_exp7c_hardneg.py — exp7c_v2 真实 AFE 数据 + 合成硬负样本后训练 | active |
| `finetune_exp7c_mild.py` | finetune_exp7c_mild.py — 温和版后训练：真实 AFE 正常拍 + 少量合成硬负样本 | active |
| `finetune_exp7c_v4.py` | finetune_exp7c_v4.py — exp7c_v4 多域平衡后训练（公共库保持 + 真实 AFE 正常抑制） | active |

## 导出

| 脚本 | 用途 | 状态 |
|---|---|---|
| `export_dual_headers.py` | 双专家 TFLite -> C 头文件 (ESP32 双模型部署) | active |
| `export_dual_tflite.py` | 双专家模型 TFLite INT8 导出 + 量化损失评估 | active |
| `export_exp6_sgd.py` | export_exp6_sgd.py — T0-1: exp6-SGD 部署链定稿模型 INT8 导出 → 固件 C 头文件 | active |
| `export_exp7b.py` | export_exp7b.py — P0-2 Task 3: exp7b 修正后因果链 INT8 导出 → 固件 C 头文件 | active |
| `export_exp7c.py` | export_exp7c.py — exp7c INT8 导出 → 固件 C 头文件 (与 export_exp7b.py 同流程) | active |
| `export_exp7c_v2.py` | export_exp7c_v2.py — 校准集加量变体: MIT+INCART 2000 + PTB 3000 + 真实 200 | active |
| `export_kd_a070_t1.py` | export_kd_a070_t1.py — KD a070_t1 (心梗筛查专家) INT8 导出 | active |
| `export_v3a_int8.py` | export_v3a_int8.py -- export v3-A clean baseline to INT8 TFLite. | active |

## 图示

| 脚本 | 用途 | 状态 |
|---|---|---|
| `fig_aami_breakdown.py` | fig_aami_breakdown.py — AAMI-category recall breakdown figure | active |
| `fig_exp6_deploy.py` | Generate exp6 deploy figures: training curves + eval bar chart. | active |
| `fig_kd_pilot.py` | fig_kd_pilot.py — KD 试点图表生成器 | active |

## QAT/量化

| 脚本 | 用途 | 状态 |
|---|---|---|
| `qat_exp7c.py` | qat_exp7c.py — exp7c INT8 量化感知训练（QAT） | active |
| `qat_exp7c_v3.py` | qat_exp7c_v3.py — exp7c_ecgfounder_v3 INT8 QAT | active |
| `qat_exp7c_v3b.py` | qat_exp7c_v3b.py — exp7c_ecgfounder_v3 INT8 QAT (MIT 加强版) | active |
| `qat_exp7c_v4.py` | qat_exp7c_v4.py — exp7c_ecgfounder_v3 INT8 QAT + v3b hard-normal | active |
| `qat_exp7c_v5.py` | qat_exp7c_v5.py — exp7c_ecgfounder_v3 INT8 QAT + full PTB-XL hard-normal | active |
| `qat_exp7c_v6_clean.py` | qat_exp7c_v6_clean.py — 无泄漏 QAT：基于 exp7b + 仅 train 患者数据 | active |

## P3 阶段

| 脚本 | 用途 | 状态 |
|---|---|---|
| `p3c_map_check.py` | p3c_map_check.py — P3c 前置: 数组行 ↔ 重建链映射验证 | active |
| `p3c_teacher_10s.py` | p3c_teacher_10s.py — P3c: 教师真实上限探针 (真 10s 上下文窗, 消除 tile 域偏移) | active |

## 预处理

| 脚本 | 用途 | 状态 |
|---|---|---|
| `preprocess_afdb_deploy.py` | preprocess_afdb_deploy.py -- AFDB deploy-causal beat extraction audit. | active |
| `preprocess_incart_lead2_deploy.py` | preprocess_incart_lead2_deploy.py -- rebuild INCART deploy-causal arrays | active |
| `preprocess_real_exp7c.py` | preprocess_real_exp7c.py — 真实 AFE 记录 → exp7c 微调正常拍 | active |
| `preprocess_svdb_deploy.py` | preprocess_svdb_deploy.py — SVDB 部署因果链 (deploy_causal) 数组构建 | active |
| `preprocess_vfdb_deploy.py` | preprocess_vfdb_deploy.py -- VFDB/CUDB deploy-causal beat extraction audit. | active |

## ECGFounder

| 脚本 | 用途 | 状态 |
|---|---|---|
| `ecgfounder_embed_1lead.py` | ecgfounder_embed_1lead.py — ECGFounder 1-lead 离线特征提取 / 硬负样本准备 | active |
| `ecgfounder_hardmine.py` | ecgfounder_hardmine.py — ECGFounder 1-lead 域距离硬负样本挖掘 | active |
| `ecgfounder_hardneg_beats.py` | ecgfounder_hardneg_beats.py — 将 ECGFounder 硬负样本候选映射回单导联 250 点拍 | active |
| `ecgfounder_mine_full_normal_v3b.py` | ecgfounder_mine_full_normal_v3b.py — v3b 全量 PTB-XL 正常拍 hard normal 挖掘 | active |
| `ecgfounder_normal_beats.py` | ecgfounder_normal_beats.py — 提取与真实 AFE 最相似的 PTB-XL 正常记录拍 | active |

## 测试

| 脚本 | 用途 | 状态 |
|---|---|---|
| `test_causal_chain_consistency.py` | test_causal_chain_consistency.py — P0-2 Step 2 一致性测试 | active |
| `test_domain_balanced_patient_split.py` | from data.dataset import prepare_datasets | active |
| `test_gpu.sh` | export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/l | active |

## 其他（无前缀历史脚本）

| 脚本 | 用途 | 状态 |
|---|---|---|
| `07_pc_inference.py` | ﻿#!/usr/bin/env python3 | round |
| `__init__.py` |  | round |
| `analyze_ondevice_bench.py` | analyze_ondevice_bench.py — On-Device Benchmark Analysis | round |
| `analyze_v5_peaks.py` | analyze_v5_peaks.py — v5 检测峰级诊断 (为 LUDB v6 参数扫描做准备) | round |
| `audit_provenance.py` | audit_provenance.py — 历史训练脚本患者级泄漏总审计 | round |
| `compare_models.py` | 模型对比可视化工具 | round |
| `compute_ai_hp_coeffs.py` | compute_ai_hp_coeffs.py — P0-2 Step 1: 计算 AI 输入链 HP 0.5Hz 修正系数 | round |
| `data_audit_balance.py` | 数据盘点: 三个数据集类别规模 + 平衡混合方案估算 (2026-08-03) | round |
| `debug_sliding_collapse.py` | 滑窗增强验证集崩溃根因诊断 (Phase 3A) | round |
| `debug_sliding_dual.py` | 滑窗增强修复验证 (Phase 3A 诊断 4) | round |
| `debug_sliding_shift0.py` | 滑窗增强崩溃机制锁定 (Phase 3A 诊断 3) | round |
| `diag_v3.py` | diag_v3.py — v3 配置验收失败诊断 (概率分布 + 分域分解 + 逐记录误报) | round |
| `distill_pilot_targets.py` | distill_pilot_targets.py — 蒸馏试点 Stage 1: 教师软目标生成 (TH §106) | round |
| `download_full.py` | MIT-BIH 全量数据集下载器 (48条记录) | round |
| `evaluate.py` | ﻿#!/usr/bin/env python3 | round |
| `exp6_deploy_train.sh` | export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/l | round |
| `export.py` | ﻿#!/usr/bin/env python3 | round |
| `figures.py` | ECG Model Figure Generator — Publication-Quality Comparison Plots | round |
| `filter_clean_pool_noincart.py` | filter_clean_pool_noincart.py -- create a clean hard-normal pool excluding | round |
| `gen_paper_figures_task10.py` | Generate AAMI breakdown + dual-track + threshold-sweep figures for ECG paper. | round |
| `gen_paper_figures_task12.py` | gen_paper_figures_task12.py | round |
| `gen_qrs_bpf.py` | QRS 带通 8-25Hz 系数 (butter 2阶 x2, fs=500) | round |
| `generate_esp_model.py` | generate_esp_model.py -- generate ESP-IDF model.cc/model.h for v3-A INT8. | round |
| `launch_exp7b.sh` | cd "/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_t | round |
| `make_replay_data.py` | 提取 MIT-BIH 正常/异常连续段 → 500Hz float C 数组 (固件回放用) | round |
| `nested_threshold_select.py` | nested_threshold_select.py — 患者级嵌套阈值选择（修复测试集偷看漏洞） | round |
| `normfix.py` | normfix.py — §105 逐拍归一化修复 (normfix) 共享参考实现 | round |
| `normfix_rms_measure.py` | normfix_rms_measure.py — §105 规格常数测量 (仅训练侧数据) | round |
| `p3_smoke.py` | p3_smoke.py — P3 前置冒烟: 教师加载/输入契约/输出顺序/吞吐。 | round |
| `p3b_student_readout.py` | p3b_student_readout.py — P3b: 学生 fc1 线性读出对照 (与 P3 教师同协议) | round |
| `posttrain_calibration_int8.py` | posttrain_calibration_int8.py — exp7c INT8 后训练分数校准与阈值重扫（MIT 部署链缓存） | round |
| `precompute_teacher_logits.py` | Precompute teacher soft-target logits for knowledge distillation. | round |
| `probe_embed_separation.py` | probe_embed_separation.py — P2: 被误报的正常拍在嵌入空间里挨着谁? (TH §105 之后) | round |
| `probe_teacher_separability.py` | probe_teacher_separability.py — P3: 教师特征能否分开"被误报正常拍"与"异常拍"? (val 侧) | round |
| `probe_val_overalert_v3.py` | probe_val_overalert_v3.py — P1: val 划分侧是否可见"自信正常误报"症状 (TH §105) | round |
| `profile_ssl.py` | SSL Performance Profiler v2 — test XLA compilation. | round |
| `retest_ai_rec_latest.py` | retest_ai_rec_latest.py — esp_timer 后 AI 输入链离线重测 (rec_latest.ecgr) | round |
| `rr_discriminative.py` | 检验 RR 特征对 SVEB 的判别力 (核心问题: SVEB 联律间期是否显著提前) | round |
| `scan_posthoc_v6.py` | scan_posthoc_v6.py — v5 检测峰事后滤波粗扫描 (排序候选门限组合) | round |
| `setup_esp_dual_board.py` | setup_esp_dual_board.py -- prepare dual-board ESP-IDF build configs. | round |
| `smoke_deploy_chain.py` | smoke_deploy_chain.py — --deploy-chain 数据开关冒烟测试 (阶段 1.5) | round |
| `smoke_patient_split.py` | 4.4-4 冒烟测试: 验证 patient_split=True 训练路径正确工作。 | round |
| `sweep_val_policy_v3.py` | sweep_val_policy_v3.py — v3 A 模型在 val 患者全序列上的阈值/策略扫描 (TH §101) | round |
| `train.py` | ﻿#!/usr/bin/env python3 | round |
| `tune_focal_params.py` | Phase 2A-2: FocalLoss γ/α 网格搜索 | round |
| `tune_threshold.py` | 阈值调优: 在验证集上搜索最佳 Abnormal 分类阈值。 | round |
| `wavelet_experiment.py` | wavelet_experiment.py — 真实 ECG 上小波 vs 现固件显示链 (量化对比) | round |

## archived（`archive/`，41 个文件）

一次性脚本与日志：acceptance_*、audit_*（除被 import 的 `audit_provenance.py`）、check_*、mine_*、sim_*、_v5_*；
其中 `audit_provenance.py` 被 `eval_clean_test.py` import，保留在根。
