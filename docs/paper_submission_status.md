# 论文投稿状态与待办清单（整合版，2026-08-13）

> **定位**：本文整合原 `SUBMISSION_READY.md`（投稿就绪审查）、`human_qa_checklist.md`（学生/导师 H1-H21 待办）、`paper_figure_inventory.md`（图表清单）三份 08-03 文档，是论文投稿准备的**单一事实源**。
> **论文**：*A Flexible Dry-Electrode ECG System with Dual-Expert On-Device Deep Learning on an ESP32-S3 Microcontroller: System Design, Deployment-Chain-Matched Training, and Patient-Level Evaluation*
> **目标期刊**：*Sensors* (MDPI)，IMRaD 格式。

---

## 1. 总体判定

**STATUS: 尚未就绪（NOT YET SUBMISSION-READY）。** 方法学核心（患者级评估、泄漏审计、部署链分析、AAMI 逐类）已完成且内部一致（57 项审计 0 失配）；三类阻塞中 **②硬件数据采集已部分解除**（2026-08-14：真实 AFE 采集 183s 静息 ECG + exp7c 真实数据微调上板，真机 46 次推理 0 误报），仍阻塞：①伦理审批 ③导师审阅签字。

---

## 2. 四项创新主张（已证据支撑）

| 支柱 | 主张 | 证据 |
|------|------|------|
| 1 | 双专家端侧 AI（心律失常 + 心梗 OR 融合） | ⚠️ 严谨口径实测已否决 OR 融合（误报叠加），论文已降级为"系统特性 + 诚实实测"，主线改分模型+前置关卡 |
| 2 | 部署链匹配重训（ΔAUC −0.105 量化） | FINAL_RESULTS 表4 + 审计 0 失配 |
| 3 | 患者级多数据集评估 + 泄漏审计 | 表2 患者级清洁 vs 方法学对照；审计 57/57 |
| 4 | 完整可复现低成本全栈 | 电极配方→AD8232→ESP32-S3→BLE→Flutter 全程记录 |

---

## 3. 图表状态（F1-F10 / T1-T6）

| 项 | 状态 | 说明 |
|----|------|------|
| F1 系统框图 | ✅ 已生成 | `models/figures/patient/fig1_system_arch.png` |
| F4 固件数据流 | ✅ 已生成 | `fig4_firmware_dataflow.png` |
| F6 ROC/PR | ✅ 已生成 | `fig6_roc_pr_{mit,ptb}.png` |
| F10 部署链消融 | ✅ 已生成 | `fig10_deploy_ablation.png` |
| F5 训练曲线 | 🟡 素材存在 | 需精选合并 |
| F7 阈值扫描 | 🟡 旧版存在 | 需患者级重生成 |
| F3 AFE/PCB | 🟡 部分存在 | 以 AD8232 版为准重组 |
| F2 电极表征 | 🔴 待实验数据 | 依赖电极 T1-T5 |
| F8 板上基准 | 🔴 待实验数据 | 依赖 ondevice bench |
| F9 App 截图 | 🔴 待截图 | Flutter 运行时 |
| T1-T6 主表 | 🟡/🔴 | T3/T6 素材存在，其余待生成/待数据 |

---

## 4. 学生/导师待办清单（H1-H21）

### Block A：伦理与机构（🔴 最高优先级，阻塞）

| # | 项 | 说明 |
|---|-----|------|
| H1 | 知情同意书导师审阅 | `docs/hardware/consent_form_zh.md` 发导师批准 |
| H2 | TJU 伦理审批路径确认 | 走简易审查 or 导师备案，获批前不启动实验 |
| H3 | 伦理声明更新 | 获批后将批准号填入 `submission_package/ethics_statement.md` 与稿件 §4.5 |
| H4 | 知情同意书联系信息填写 | `[导师姓名]/[学生姓名]/[邮箱]` 占位符 |

### Block B：物理实验（🔴 数据阻塞）

| # | 项 | 协议 |
|---|-----|------|
| H5 | 电极 T1-T5 表征 | `docs/hardware/electrode_char_protocol.md` + `electrode_char_checklist.md` |
| H6 | 人体实验（n=5-10） | `docs/hardware/human_subject_protocol.md` |
| H7 | 板上基准（延迟/功耗/INT8/温升） | `docs/hardware/ondevice_bench_protocol.md` |
| H8 | 实测数据填入稿件 [待补充] | §5.1 / §5.4 / §5.5 |

> **进度更新（2026-08-14）**：Block B 已部分解除——真实 AFE 采集完成（183s 静息 ECG，有效采样率实测 225.68Hz），exp7c 真实数据微调已上板（真实正常拍置信度 0.732→0.417，MIT AUC 0.8964 / PTB 0.8015，INT8 0.8979/0.7880，真机 46 次推理 0 误报，见 TUNING_HISTORY §48 / deploy_match/{finetune_exp7c,retrain_exp7c_eval,int8_exp7c_check}.json）。H5-H7 仍待执行；H8 中 exp7c 域适配结果已写入稿件 §4.3/§5.2。

### Block C：作者与投稿决策

| # | 项 |
|---|-----|
| H9 | 作者行确认（第一作者 + 通讯作者 + CRediT） |
| H10 | 通讯作者身份确认 |
| H11 | 基金信息确认（国家自然科学基金号等） |
| H12 | 审稿人建议（3-5 名 wearable ECG / embedded AI 专长） |

### Block D：稿件制作（投稿前）

| # | 项 |
|---|-----|
| H13 | 英文润色（可选） |
| H14 | Sensors 模板转换（.docx/.tex） |
| H15 | 图导出 ≥600 dpi |
| H16 | 参考文献重编号 + 格式统一 |
| H17 | Abstract 压缩至 ~200 词 |
| H18 | 非数据类占位符填写（作者邮箱/单位等） |

### Block E：完整性核查

| # | 项 |
|---|-----|
| H19 | 数值一致性核查（稿件 ↔ FINAL_RESULTS ↔ JSON） |
| H20 | 文献矩阵引用覆盖检查 |
| H21 | 硬件数值 [待复核] 手动验证 |

---

## 5. 阻塞项依赖链

```
(a) 伦理审批 ──┐
              ├──▶ (c) 人体实验 ──▶ T4 表 + §5.4
(b) 导师签字 ─┘
(c) 电极表征 ──▶ F2 图 + T2 表 + §5.1
(c) 板上基准 ──▶ F8 图 + T5 表 + §5.5
(d) 模板转换 + (e) 图导出 + (f) 作者信息 ──▶ 可投稿稿件
```

---

## 6. 推荐执行顺序

伦理审批（H1-H2）→ 电极表征（H5，可先于伦理做材料测试）→ 人体实验（H6）→ 板上基准（H7）→ 填入稿件（H8）；作者/投稿决策（H9-H12）与稿件制作（H13-H18）可并行。

---

*本文档整合自 08-03 的三份投稿准备文档（SUBMISSION_READY / human_qa_checklist / paper_figure_inventory），状态与待办内容未变，仅合并去重。*
