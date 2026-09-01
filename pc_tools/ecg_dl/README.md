# ecg_dl 深度学习工具包 — 分类目录

> **本文件是 `pc_tools/ecg_dl/` 下全部脚本的唯一权威索引**（2026-08-21 整理）。
> 新会话/AI 先读本文再找脚本，避免"不知用途、不知先后、重复造轮子"。
>
> **状态标记**：🟢 现役（活跃使用）｜🔵 证据链（对应 TH/FINAL_RESULTS 数字，勿删勿改）｜⚪ 历史路线（已关闭，留档）
>
> **铁律**：① 评估数字必须能溯源"数字 → models/*.json → 本目录脚本"；
> ② 声称已完成的实验必须有落盘 JSON；③ 已关闭路线清单见 `docs/03_Software_Docs/AGENTS.md` §10。

## 快速定位（按任务）

| 我要… | 去哪 |
|--------|------|
| 复现论文主结果（患者级表2） | `eval_patient_split_all.py` → `eval_patient_split_noaug.py` |
| 重训主模型 | `train.py`（配置见 `config.py`；启动器 `run_*.sh`） |
| 部署链口径评估 | `build_deploy_npz.py` → `eval_deploy_match.py` |
| 导出固件 INT8 模型 | `export.py` / `export_exp6_sgd.py` / `export_exp7c.py` / `export_kd_a070_t1.py` |
| 验证心率检测器 | `verify_heartrate_ludb_v6.py`（当前 v6；v5/初版为历史对照） |
| PTB-XL 记录级/MI 评估 | `eval_ptbxl_record_level.py`（+ `--save-scores` 配合 `nested_threshold_select.py`） |
| 关卡+双专家级联 | `eval_triage_gate.py`（§8.9 模拟）；规划见 `docs/03_Software_Docs/dual_expert_deployment_plan.md` |
| 患者重叠/泄漏核查 | `check_ptbdb_ptbxl_overlap.py` / `audit_leakage.py` / `check_patient_leakage.py` |
| 画论文图 | `figures.py` / `gen_paper_figures_task10.py` / `gen_paper_figures_task12.py` |

## 目录结构（包核心，**勿动勿改名**）

| 路径 | 内容 |
|------|------|
| `config.py` | 全局配置：路径、采样率、窗口。**被 ~80 个脚本 import，改一行影响全库** |
| `data/` | 数据集加载与预处理模块（dataset/patient_split/preprocess_* 等） |
| `models/` | 模型架构（resnet_lite_1d / cnn_1d / cnn_m / resnet_multitask_1d）。注意：训练产物 JSON 在 `models/` **输出目录**，与此包同名不同物 |
| `losses/` | focal_loss / kd_loss / multitask_loss / contrastive |
| `inference/` | PC 端推理封装（tta 等） |
| `utils/` | 通用工具 |
| `tests/` | 单元测试（test_kd_loss / test_kd_dataset / test_teacher_logits_alignment） |
| `test_vectors/` | 固件联调用的 CSV 测试向量 |
| `reports/` | 报告产物目录 |
| `_archive_oneoff/` | **一次性调试脚本归档**（25 个，含 MANIFEST.md，2026-08-21 整理） |

## A. 数据流水线

| 脚本 | 用途 | 状态 |
|------|------|------|
| `download_full.py` | MIT-BIH 全量 48 记录下载 | 🟢 |
| `preprocess.py` / `preprocess_incart.py` | MIT-BIH / INCART 预处理 → 250Hz beat npz | 🟢 |
| `preprocess_ptb.py` | PTB(ptbdb) 预处理（含硬编码路径，见下方警告） | 🟢 |
| `preprocess_ptbxl.py` / `preprocess_ptbxl_records.py` / `preprocess_ptbxl_rhythm.py` | PTB-XL beat 级 / 记录级 / 节律(AF) 预处理 | 🟢 |
| `preprocess_svdb.py` | SVDB 预处理（历史域平衡实验用） | ⚪ |
| `preprocess_real_exp7c.py` | 真实 AFE 记录 → exp7c 微调数据 | 🟢 |
| `build_deploy_npz.py` | 部署链(D3)训练集重建（十三章阶段1.5） | 🔵 |
| `build_noaug_testset.py` | T1-2 未增强 MIT 测试集重建 | 🔵 |
| `build_3beat_deploy.py` | 3-beat 部署集（D5 已否决路线的配套） | ⚪ |
| `make_replay_data.py` | MIT-BIH 片段 → 500Hz C 数组（固件回放） | 🟢 |
| `data_audit_balance.py` | 数据集规模/平衡审计 | 🔵 |

## B. 训练

| 脚本 | 用途 | 状态 |
|------|------|------|
| `train.py` | **主训练入口**（FocalLoss+增强；exp4/5/6 系列由此产出） | 🟢 |
| `train_cross_arch.py` | 跨架构部署链失配对照（审计 §2.2 P1 实验） | 🟢 进行中 |
| `finetune_exp7c.py` | exp7b→exp7c 真实数据微调（TH §40） | 🔵 |
| `train_mixed_balanced.py` | 平衡混合单模型（TH §8.9.6 否决路线） | ⚪ |
| `train_cls_head.py` | SSL encoder 分类头微调（Phase 2C） | ⚪ |
| `tune_focal_params.py` | FocalLoss α/γ 扫描（D1 证据） | 🔵 |
| `tune_threshold.py` | 验证集阈值扫描 | 🔵 |

启动器：`run_exp4_patient_clean.sh` / `run_exp5_patient_clean.sh` / `run_exp6_{deploy,hp005,phase,sgd}.sh` / `run_bal_mixed.sh` / `run_clean_ab_*.sh` / `run_exp7{,b}.sh` / `run_kd_screen.sh` / `run_cross_arch_all.sh` / `run_retest_ai.sh` / `run_3beat_deploy.sh` / `run_eval_aami_3beat.sh` / `run_aami_breakdown.sh` / `launch_exp7b.sh` / `check_exp7b.sh` / `check_h19.sh` / `exp6_deploy_train.sh` / `test_gpu.sh` —— 均为 WSL 训练启动器，保留原位。

## C. 知识蒸馏（KD → 心梗专家）

| 脚本 | 用途 | 状态 |
|------|------|------|
| `precompute_teacher_logits.py` | 教师 soft-target 预计算 | 🟢 |
| `train_kd.py` | KD 训练（a030/a050/a070 × t1/t3/t5 网格） | 🟢 |
| `export_kd_a070_t1.py` | KD a070_t1 INT8 导出（心梗腿现役模型） | 🟢 |

## D. 历史路线（SSL/集成/多任务/两阶段——均已关闭，留证）

`train_ssl.py`(D6 否决) / `train_pretrain.py`(Route K) / `train_ensemble.py`(P2C) / `train_multitask.py`(Route F) / `train_two_stage.py` / `profile_ssl.py` / `plot_ssl_history.py`

## E. 评估——患者级主结果链（FINAL_RESULTS 表2/表3/表8）

| 脚本 | 用途 | 状态 |
|------|------|------|
| `evaluate.py` | 模型评估基础（h5/tflite 对比） | 🟢 |
| `eval_patient_split_all.py` | 全模型患者级离线评估（4.4-4） | 🔵 |
| `eval_patient_split_noaug.py` | T1-2 未增强测试重评（表2 主口径） | 🔵 |
| `eval_final_models.py` | exp2'/exp3/P2A 三模型对比 | ⚪ |
| `compare_models.py` | 模型对比可视化 | ⚪ |
| `eval_bootstrap_ci.py` | M8 bootstrap 95% CI | 🔵 |
| `audit_leakage.py` | M10 划分泄漏审计（66.3% 结论来源） | 🔵 |
| `check_patient_leakage.py` | 多架构训练患者泄漏核查 | 🔵 |
| `verify_split_consistency.py` | 患者级划分一致性验证 | 🔵 |
| `eval_aami_breakdown.py` | AAMI 逐类分解（表 T11；被多个脚本 import，勿动） | 🔵 |
| `eval_aami_3beat.py` | 3-beat 口径 AAMI（D5 对照） | ⚪ |
| `audit_s_class.py` | M6 S 类构成效应分析 | 🔵 |
| `audit_precision.py` | §30 类内精确率恒等式审计 | 🔵 |
| `eval_binary_all.py`(+`.sh`) | 关键模型 FP32 vs D3 统一评估 | 🔵 |
| `nested_threshold_select.py` | **漏洞#1 修复：患者级嵌套阈值选择** | 🟢 证据链 |

## F. 评估——部署链（D3 / exp7 系列）

| 脚本 | 用途 | 状态 |
|------|------|------|
| `eval_deploy_match.py` | **部署链评估总装**（100KB 核心件；被 8 个脚本 import，勿动） | 🔵 |
| `eval_deploy_compensation.py` | T1-3 失配消融+补偿原型（表5） | 🔵 |
| `eval_phase_robustness.py` | T2-5 相位鲁棒性（负面结果） | 🔵 |
| `eval_exp6_deploy.py` | exp6 部署链重训评估 | 🔵 |
| `eval_exp7.py` / `eval_exp7b.py` | exp7/exp7b 因果链评估+阈值校准 | 🔵 |
| `eval_exp7c.py` | exp7c 微调后 MIT/PTB AUC + 置信度复核 | 🔵 |
| `check_int8_compare.py` | exp7b vs exp7c INT8 同解释器逐拍对比 | 🔵 |
| `check_int8_exp7c.py` | INT8 精度核验（tflite vs float32） | 🔵 |
| `eval_int8_nosoftmax.py` | M3 INT8 去 softmax 对照 | 🔵 |
| `verify_exp6_sgd_int8.py` | T0-1 INT8 板上模型参数核对 | 🔵 |
| `smoke_deploy_chain.py` | --deploy-chain 数据管线冒烟 | ⚪ |

## G. 评估——PTB-XL 记录级 & MI（现役 P0/P1 实验）

| 脚本 | 用途 | 状态 |
|------|------|------|
| `eval_ptbxl_record_level.py` | **记录级验证（完全复刻板上部署链）**；`--save-scores` 供嵌套阈值 | 🟢 |
| `eval_ptbxl_record_level_lead_scan.py` | 12 导联逐导扫描 | 🟢 |
| `eval_ptbxl_mi_subtypes.py` | MI 亚类分层（IMI/AMI/ASMI…） | 🟢 |
| `eval_ptbxl_mi_vs_abnormal.py` | MI vs 其他异常阴性集扩充 | 🟢 |
| `check_ptbdb_ptbxl_overlap.py` | **漏洞#2 修复：ptbdb↔PTB-XL 患者重叠三层核查** | 🟢 证据链 |
| `eval_st_morphology.py` | ST 形态学预校验模块 4 消融 | 🔵 |
| `eval_ptb_holdout.py` | exp5 vs P2A PTB 留出对比（Phase 3B） | ⚪ |

## H. 评估——报警决策 / 融合 / 关卡

| 脚本 | 用途 | 状态 |
|------|------|------|
| `eval_alarm_decision.py` | T1-4 报警策略（表6；§29 bug 修正主角） | 🔵 |
| `eval_context_fusion.py` | 拍级 CNN+RR 融合实验（TH §二十八 负面结果） | 🔵 |
| `eval_fusion.py` / `eval_weighted_fusion.py` / `eval_expert_combo.py` / `eval_expert_combo_patient.py` / `eval_deploy_decision.py` | 双专家 OR/加权组合验证（§8.8 否决证据链） | 🔵 |
| `eval_triage_gate.py` | §8.9 分诊式关卡模拟（**现役关卡方案的前身验证**） | 🟢 |
| `sim_temporal_agg.py` | N-of-M 时间聚合模拟（零训练） | 🔵 |
| `sim_record_level.py` | 记录级策略研究（§8.9.9） | 🔵 |
| `sim_mixed_testset.py` | 真实比例混合测试集构造 | 🔵 |
| `sim_balance.py` / `sim_rr_rhythm.py` | 类平衡模拟 / RR 截断 SVEB 模拟 | 🔵 |
| `rr_discriminative.py` / `verify_rr_feature.py` / `sanity_rr.py`* | RR 特征判别力验证（S 类 pre-RR AUC 0.964） | 🔵 |
| `verify_waveform_hypothesis.py` | "精度损失源于电极极化漂移"假设验证 | 🔵 |
| `wavelet_experiment.py` | 小波 vs 固件滤波显示对比 | ⚪ |

\* sanity_rr 已移入 `_archive_oneoff/`。

## I. 评估——节律安全（AF / VF）

| 脚本 | 用途 | 状态 |
|------|------|------|
| `eval_rhythm_af.py` | T4-8 心律安全+AF（AFDB 30s 窗） | 🔵 |
| `eval_rhythm_af_ptbxl.py` | AF 短窗验证（PTB-XL 10s，"一键测房颤"） | 🔵 |
| `eval_vf_detect_v2.py` | **VF/VT 检测器 v2**（固件逐位复刻口径；被 ablation/check_vf import，勿动） | 🟢 |
| `eval_vf_detect.py` | VF v1 历史 | ⚪ |
| `eval_vf_detect_ablation.py` / `eval_vf_detect_ablation2.py` | CUDB Se 下降归因 + AFE 尺度校准 | 🔵 |
| `check_vf_v2_on_ludb.py` | LUDB 正常集上 VF v2 误报检查 | 🔵 |

## J. 心率检测器验证（LUDB 金标准）

| 脚本 | 用途 | 状态 |
|------|------|------|
| `verify_heartrate_ludb_v6.py` | **v6 参数扫描与验证（当前固件算法）** | 🟢 |
| `verify_heartrate_ludb_v5.py` | v5 历史对照（表9附） | 🔵 |
| `verify_heartrate_ludb.py` | 初版 Pan-Tompkins 验证（v4.x 对照） | 🔵 |
| `analyze_v5_peaks.py` | v5 峰级诊断（为 v6 门限标定备料） | 🔵 |
| `scan_posthoc_v6.py` | v5 记录后验门限扫描 | 🔵 |

## K. 导出与固件一致性

| 脚本 | 用途 | 状态 |
|------|------|------|
| `export.py` | 通用 INT8 导出 + C 数组 | 🟢 |
| `export_exp6_sgd.py` | exp6-SGD 定稿导出（T0-1） | 🔵 |
| `export_exp7b.py` / `export_exp7c.py` / `export_exp7c_v2.py` | exp7 系列导出（v2 = 校准集扩展版） | 🔵 |
| `export_dual_tflite.py` / `export_dual_headers.py` | 双专家双头文件导出（旧 OR 方案遗留） | ⚪ |
| `compute_ai_hp_coeffs.py` | AI 输入链 HP 0.5Hz 系数设计（P0-2 Step1） | 🔵 |
| `verify_fw_ai_hp_coeffs.py` | 固件 AI_HP_* 系数与训练侧一致性验证 | 🔵 |
| `gen_qrs_bpf.py` | QRS 8-25Hz 带通系数生成（心率 v6 配套） | 🟢 |
| `07_pc_inference.py` | PC 实时推理 + 基准 | 🟢 |
| `retest_ai_rec_latest.py` | esp_timer 后 AI 链离线重测（rec_latest.ecgr） | 🔵 |
| `test_causal_chain_consistency.py` | P0-2 因果链一致性测试 | 🔵 |

## L. 论文图表与审计

| 脚本 | 用途 | 状态 |
|------|------|------|
| `figures.py` | 出版级对比图生成器 | 🟢 |
| `gen_paper_figures_task10.py` / `gen_paper_figures_task12.py` | 论文图组（AAMI/双轨/阈值扫描等） | 🟢 |
| `fig_aami_breakdown.py` / `fig_exp6_deploy.py` / `fig_kd_pilot.py` | 单图脚本 | 🔵 |
| `plot_history.py` / `plot_ssl_history.py` | 训练曲线 | 🔵 |
| `audit_manuscript.py` | H19 稿件数值一致性审计（57 项） | 🟢 |

## M. 硬编码路径警告 ⚠️

以下脚本内含**绝对路径**（`C:\Users\cai\...` 或 `/mnt/c/Users/cai/...`），换机器/换用户名时需同步修改：
`audit_leakage.py`, `audit_s_class.py`, `eval_context_fusion.py`, `eval_deploy_decision.py`, `eval_deploy_match.py`, `eval_expert_combo.py`, `eval_fusion.py`, `eval_ptb_holdout.py`, `eval_vf_detect_ablation.py`, `eval_vf_detect_ablation2.py`, `eval_weighted_fusion.py`, `export_dual_tflite.py`, `gen_paper_figures_task12.py`, `smoke_patient_split.py`, `train_ssl.py`, `verify_exp6_sgd_int8.py`, `verify_split_consistency.py`, `preprocess_ptb.py`（ECG-Database 双路径探测，Windows/WSL 各一）。
其余脚本一律经 `config.py` 相对定位——**改路径先改 config.py，不要散改**。

## N. 整理规则（2026-08-21 起）

1. 新脚本必须带中文 docstring 头（用途/输入/输出/验收标准），否则视为临时脚本放入 `_archive_oneoff/`。
2. 移动/删除任何脚本前，先全仓库 grep 文件名（md/sh/py/json）确认零引用。
3. 被 FINAL_RESULTS/TUNING_HISTORY 点名的脚本永久保留（证据链）。
4. 一次性真机调试脚本用完即移入 `_archive_oneoff/` 并登记 MANIFEST。

## O. ECGFounder 离线特征 / 硬负样本挖掘（2026-08-24 起）

| 脚本 | 用途 | 状态 |
|---|---|---|
| `ecgfounder_embed_1lead.py` | 用 ECGFounder 1-lead checkpoint 提取 PTB-XL/真实 AFE 10s 段 1024 维特征 | 🟢 |
| `ecgfounder_hardmine.py` | 计算真实 AFE↔公共异常/正常距离，输出硬负样本候选 JSON/CSV | 🟢 |
| `ecgfounder_hardneg_beats.py` | 将 top 异常候选映射为 exp7c 部署链 250 点拍 | 🟢 |
| `ecgfounder_normal_beats.py` | 将 top 真实相似正常候选映射为 250 点拍 | 🟢 |
| `finetune_exp7c_ecgfounder*.py` | ECGFounder 候选数据 + exp7c 后训练实验 | 🟢 实验中 |

依赖：ECGFounder 权重位于 `项目根/../ECGFounder/checkpoint/`，使用 CPU PyTorch 2.4。

| `qat_exp7c_v3.py` / `qat_exp7c_v3b.py` | ECGFounder v3 路线 QAT 导出 | 🟢 |
| `eval_int8_ecgfounder_v3.py` | ECGFounder v3 QAT INT8 评估 | 🟢 |

| `audit_v3b_leakage.py` | v3b/v4/v5 患者级泄漏审计 | 🔵 |
| `qat_exp7c_v6_clean.py` | 无泄漏 clean QAT（exp7b base + train-only） | 🟢 |
| `eval_event_clean_v6_val_test.py` | clean v6 验证选参/测试冻结事件评估 | 🟢 |
