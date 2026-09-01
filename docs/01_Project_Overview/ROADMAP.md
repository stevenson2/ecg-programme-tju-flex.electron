# ESP32-ECG 项目路线图（2026-08-21 更新）

> 本文档替代旧版 ROADMAP.md（2026-08-05 快照）。
> 当前状态以本文档 + `docs/03_Software_Docs/dual_expert_deployment_plan.md` + `docs/01_Project_Overview/PROJECT_SUMMARY.md` 为准。
> 实验证据以 `docs/03_Software_Docs/TUNING_HISTORY.md` 为准；权威数字以 `docs/FINAL_RESULTS.md` 为准。

---

## 1. 项目定位

基于 **ESP32-S3-WROOM-1-N16R8** 的便携式单导联心电采集系统：

- 500Hz 实时采样
- 板上数字滤波 + 心率检测 + TFLite Micro INT8 逐拍异常检测
- 心律安全 / AF / VF / 停搏检测
- BLE 推流到 Flutter App
- SPIFFS 记录 + WiFi AP 下载
- 产品定位：心律失常检测 + 显示（消费级，非临床诊断）
- 论文核心创新：训练-部署滤波链失配的系统量化与补偿

---

## 2. 当前状态快照

| 维度 | 状态 |
|---|---|
| 硬件 | ESP32-S3-WROOM-1-N16R8，AD8232 AFE，3 电极单导联 |
| 固件 | Arduino + PlatformIO；esp_timer 500Hz；HR v6；五路统一报警；SPIFFS 记录；WiFi AP；BLE 125Hz |
| 板上模型 | `exp7c`（ResNet-L INT8，167,376 B，θ=0.60 + 5 拍确认）已上板，真机 0 误报 |
| 双专家方案 | P2A（心律失常）+ KD a070_t1（心梗）已具备；**关卡模型尚未训练** |
| 评估 | 患者级划分、部署链匹配、PTB-XL 记录级验证、MI 亚类分层均已完成一轮 |
| 论文 | Sensors 目标；尚未投稿；阻塞项：伦理审批、导师签字、剩余硬件实验 |
| 仓库 | 已完成大文件/历史清理，`.git` 约 3.56 MiB |

---

## 3. 已完成的关键里程碑

### 3.1 模型与评估
- [x] 患者级划分与泄漏审计
- [x] 部署链匹配重训（exp6-SGD 试点，D3 口径）
- [x] exp7c 真实 AFE 数据微调并上板
- [x] PTB-XL 记录级验证复刻板上部署链
- [x] KD a070_t1 INT8 导出与验证
- [x] M0：阈值扫描 + KD 三种聚合对比
- [x] P1 补证：MI 亚类分层 / 12 导联扫描 / 阴性集扩充
- [x] 患者级嵌套阈值选择修复

### 3.2 固件与硬件
- [x] esp_timer 500Hz 硬件采样
- [x] AFE 过采样 4→1，主循环 ~496Hz
- [x] BLE 125Hz 降载 + 重连订阅泄漏修复
- [x] VF/VT v2 + 无组织心律互锁
- [x] 五路统一报警 `updateUnifiedAlarm`
- [x] LUDB v6 心率重验（F1 0.868）
- [x] SPIFFS 录制保留策略修复
- [x] 录制断电持久化验收

### 3.3 工程基础设施
- [x] 构建参数注入（固件 AP 密码 / Flutter token）
- [x] Git 历史清理与生成物忽略
- [x] 文档体系整合（PROJECT_SUMMARY / HANDOFF / paper_submission_status）

---

## 4. 当前主线

### 4.1 目标 1：跨架构部署链失配验证（普适性证明）

> 目的：证明“训练链 vs 部署链失配”不是本项目 ResNet 家族特有的问题，而是**跨架构、跨文献模型的系统性问题**。
> 详细设计见 `docs/04_Paper_Submission/innovation_and_rigor_audit.md` §2.2。

**实验设计**：3 个外部架构 × 3 个条件

| 条件 | 训练预处理 | 测试预处理 | 作用 |
|---|---|---|---|
| A | 训练链 | 训练链 | 基线（预期 Δ≈0） |
| B | 训练链 | 部署链 | 测失配落差 Δ |
| C | 部署链 | 部署链 | 测部署链重训的修复效果 |

**外部架构池**：
1. `lstm_cnn` —— 张异凡式 LSTM+CNN 并行组合（复现文献）
2. `cnn_standard` —— 标准 1D-CNN（无残差）
3. `resnet1d` —— 通用 ResNet-1d

**验收标准**：
- 所有架构条件 B 的 AUC 下降均与本项目 ΔAUC ≈ −0.105 可比 → 失配为架构无关的系统性问题
- 条件 A 为阴性对照，排除“部署链数据本身更难”的解释
- ⚠️ 实测未通过该验收标准：失配仅在 PTB 域纯 CNN/ResNet 上显著，LSTM+CNN 与 MIT/INCART 域不满足“架构无关”假设，结论已按实际证据修正。

**当前进度**：
- 已完成 3 架构 × 2 链训练：`models/cross_arch/{lstm_cnn,cnn_standard,resnet1d}_{baseline,deploy}.h5` + meta/history
- 已完成 `eval_cross_arch.py`，输出 `models/cross_arch_eval.json`（A/B/C AUC + 患者级 bootstrap CI）
- 结果摘要（Δ=B−A，负值表示部署链测试上 AUC 下降）：
  | 架构 | MIT Δ | PTB Δ |
  |---|---|---|
  | `lstm_cnn` | +0.025 | −0.001 |
  | `cnn_standard` | +0.005 | −0.181 |
  | `resnet1d` | +0.005 | −0.103 |
- 注意：MIT 域未出现 B 下降；PTB 域 `cnn_standard`/`resnet1d` 有下降，`lstm_cnn` 无显著下降。根因分析已完成（TUNING_HISTORY 第六十六章 §7）：cross_arch 的 D3 链是“因果 0.05Hz+LP40+梳状+抽取”，4Hz 仅显示链；当前固件 AI 链另有最终 0.5Hz 因果高通。部署链群延迟两域相同（δ*≈−6），PTB 的 MI 诊断依赖 0.5–5Hz ST/T 形态，对因果滤波相位/时移敏感；MIT/INCART 心律失常主要依赖 5–40Hz QRS 形态，因果链影响小甚至因梳状去噪而提升。纯 CNN 局部模板对相位/时间错位敏感，LSTM 分支提供时序鲁棒性。
- 已完成 AAMI 部署链逐类矩阵：`eval_aami_matrix.py` → `models/aami_matrix_deploy_patient.{json,csv}`；包含 exp5/exp6-SGD/exp7b/**exp7c float32/INT8**/P2A 与六个 cross-arch 模型。结论：aggregate AUC 会掩盖阈值和类别效应；exp7c INT8 在 PC @0.60 下 FAR 更低但 recall 更低，应作为 deployment anchor 单独讨论（TUNING_HISTORY 第六十七章）。

**下一步**：
- [x] 完成 3 架构 × 2 链（baseline/deploy）训练
- [x] 运行 `eval_cross_arch.py`，输出 A/B/C AUC + 患者级 bootstrap CI
- [x] 分析 MIT vs PTB 不一致原因（部署链测试集构成、模型容量/训练收敛、domain-balanced 影响）→ 根因已写入 TUNING_HISTORY 第六十六章 §7：任务频段（PTB 低频 ST/T vs MIT/INCART 高频 QRS）+ 模型归纳偏置（纯 CNN 局部模板敏感、LSTM 分支时序鲁棒）
- [x] 加入 exp7c 部署锚点与 AAMI 逐类矩阵；已确认当前 AAMI matrix 与 FINAL_RESULTS noaug causal cache 不是同一测试集口径，不能直接横比
- [x] 完成 exp7c `θ × K-of-N` 报警策略扫描：当前 INT8 `θ=0.60 + 5-of-5` 在 MIT/INCART 序列上几乎无事件召回；降阈值可提高事件召回但 alert rate / FP 同步上升（TUNING_HISTORY 第六十八章）
- [x] 补充多拍 episode 口径与 `1-of-N` 策略：GT 异常拍按 ≤5 拍间隙回并 episode；`θ=0.50, 1-of-5/1-of-7` 在 test 上达到 event recall 0.87–0.88、event precision 0.77–0.78，且 FP/record 较单拍下降（TUNING_HISTORY 第六十九章）
- [x] 在显式约束下选出 PC 候选报警操作点：约束 `false_alarm_blocks/record <= 10` 最大化 event F1，推荐 **INT8 θ=0.50, 1-of-5**（test event recall 0.874 / precision 0.770；TUNING_HISTORY 第七十一章）。真实 AFE 长时 FP/hour 验证待硬件阶段执行
- [x] 补 noaug / non-augmented AAMI 矩阵：`eval_aami_matrix_noaug.py` → `models/aami_matrix_deploy_patient_noaug.{json,csv}`；exp7c INT8 AUC 0.8894 与历史 causal-cache 口径一致（TUNING_HISTORY 第七十章）
- [ ] 将结果写入论文章节：按“architecture-, domain-, and operating-point-dependent deployment-chain sensitivity”表述，不再声称架构无关普适失配

### 4.2 目标 2：关卡 + 双专家部署

> 详细规划见 `docs/03_Software_Docs/dual_expert_deployment_plan.md`。

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | KD a070_t1 验证 / INT8 导出 | ✅ 完成 |
| M1 | 关卡模型训练（A1） | ❌ 初版未过门槛（二分类/三分类均未达标，TUNING_HISTORY 第七十二/七十三章） |
| M2 | PC 级联模拟（A2/A3/A4） | ❌ 初版未过门槛（MIT E_A>18%，PTB-R<45%） |
| M3 | INT8 导出与 Flash/SRAM 预算（B1/B2） | 待办 |
| M4 | 固件分诊状态机（B3/C1） | 待办 |
| M5 | 板级与真实 AFE 验收（C2/C3） | 待办 |
| M6 | 跨库记录级验证（D） | 待办 |

**当前正在做**：
1. **A1 关卡模型训练**
   - 目标：正常 vs 异常双域二分类关卡
   - 验收：误放行 ≤5%，异常保留 ≥95%
   - 备选：三分类关卡（正常 / 心律失常疑似 / 心梗疑似）

2. **A2 级联 PC 模拟**
   - 关卡 × {P2A / exp7c / exp6-SGD} × KD a070_t1
   - 阈值网格 + 时间确认（N=3/5/7）
   - 产物：`dual_expert_triage_eval.json`

---

## 5. 下一步（按优先级）

| # | 任务 | 说明 |
|---|---|---|
| 1 | 跨架构部署链失配验证 | 当前第一目标：完成外部架构训练与 A/B/C 评估，输出普适性结论 |
| 2 | 关卡模型训练 | 直接决定双专家方案是否成立 |
| 3 | 级联 PC 模拟 | 输出决策门槛所需指标 |
| 4 | 论文补证 | 把跨架构验证、P1 补证、exp7c、部署链结果同步进稿件 |
| 5 | 固件双模型集成 | 单解释器分时复用，或按需切换模型 |
| 6 | 全链路验证 | PC-ESP32 一致性、真实采集、温度/功耗 |
| 7 | 人体实验 | 依赖伦理审批，目前阻塞 |
| 8 | 投稿准备 | 作者信息、模板转换、图表导出、润色 |

---

## 6. 关键决策与已关闭路线

### 仍然有效的核心决策
- **患者级划分是铁律**：记录级划分存在泄漏，论文只使用患者级口径。
- **部署链失配的正解**：部署链重训 + SGD + 群延迟补偿（`AI_TRIGGER_OFFSET=6`）。
- **INT8 量化近无损**：\|ΔAUC\| ≤ 0.006。
- **当前板上模型 = exp7c**：真实数据微调版，优先于 exp6-SGD。
- **双专家方案 = 关卡 + 正确专家 + 时间确认**，不是裸 OR。

### 已关闭路线（勿重试）
- 双专家裸 OR（P2A + exp5_clean）——严谨口径不可用
- 3-beat / 长窗输入——数据量不足
- SimCLR SSL 预训练——无增益
- 全类相位扰动增强——负面结果
- 拍级 RR 上下文融合器——无净收益
- 二次 softmax——压缩概率动态范围，已删除

---

## 7. 风险与回退

| 风险 | 回退方案 |
|---|---|
| 关卡模型达不到误放行 ≤5% | 放宽到 10% + 时间确认；或三分类关卡 |
| 级联后 PTB 召回仍低 | 心梗专家改记录级模型（10s 窗口），拍级只做心律失常 |
| SRAM 紧张 | 只部署“关卡 + 单专家”，按模式切换 |
| 板上延迟不达标 | 关卡用更小模型，专家仅在关卡触发时运行 |
| 全链路收益不及单模型 | 回退单模型 exp7c，保留 PTB-XL 记录级评估作为研究产出 |

---

## 8. 文档导航

| 文档 | 用途 |
|---|---|
| `docs/03_Software_Docs/dual_expert_deployment_plan.md` | 双专家/关卡部署详细计划（当前最接近执行态） |
| `docs/01_Project_Overview/PROJECT_SUMMARY.md` | 项目现状总结 |
| `docs/01_Project_Overview/HANDOFF.md` | 交接与接手提示词 |
| `docs/03_Software_Docs/TUNING_HISTORY.md` | 实验证据日志 |
| `docs/FINAL_RESULTS.md` | 论文权威数字 |
| `docs/04_Paper_Submission/paper_submission_status.md` | 投稿状态与阻塞项 |
| `README.md` | 项目总览与使用方式 |

---

*本文档由旧版 ROADMAP.md 重写，反映 2026-08-21 的项目真实状态。*