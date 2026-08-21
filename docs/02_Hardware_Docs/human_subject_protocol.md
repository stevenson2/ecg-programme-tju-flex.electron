# Human-Subject ECG Acquisition Protocol
## Prototype Validation — Flexible PEDOT:PSS Dry Electrode vs Commercial Ag/AgCl Gel Electrode

> **Status**: 待与导师（杨辉老师）确认天津大学伦理简易审批/知情同意流程后执行  
> **Device identity**: Research prototype — NOT for clinical diagnosis  
> **Author**: [学生姓名] | **Advisor**: 杨辉 / 黄显 | **Date**: [填写日期]

---

## 1. Ethical Statement

本实验为**设备原型验证（research prototype validation）**，非临床试验。正式执行前须完成以下伦理前置步骤：

1. **天津大学伦理简易审批**：确认本实验是否适用天津大学涉及人的生物医学研究伦理简易审查流程。若需要，提交本协议 + 知情同意书模板，获得伦理批准备案号。
2. **导师签字确认**：杨辉（或黄显）老师审阅并签字批准实验方案与知情同意书。
3. **受试者知情同意**：每位受试者在充分理解后签署《知情同意书》（见 `consent_form_zh.md`），获得签章副本。

> ⚠️ **待办（student action）**：联系杨辉老师确认路径——走学校伦理简易审查 or 导师内部备案？获得明确书面/邮件批准后方可启动实验。论文投稿时须在 Methods 中声明伦理合规路径及批准编号（或导师备案说明）。

---

## 2. Inclusion & Exclusion Criteria

### 2.1 纳入标准

| # | Criterion |
|---|-----------|
| I1 | 年满 18 周岁（≥18 岁） |
| I2 | 健康状况良好，无已知心血管疾病史 |
| I3 | 自愿参加并签署知情同意书 |
| I4 | 实验前一晚睡眠 ≥6 小时（自述），实验前 2 小时内未摄入咖啡因/酒精/剧烈运动 |

### 2.2 排除标准

| # | Criterion |
|---|-----------|
| E1 | 已知心脏病史（含心律失常、冠心病、心肌炎等任何心脏相关诊断） |
| E2 | 植入心脏起搏器、ICD 或其他电子植入器械 |
| E3 | 已知皮肤过敏史（尤其是对医用胶带、电极凝胶、导电聚合物过敏） |
| E4 | 正在服用可能影响心率的药物（β 阻断剂、钙通道阻滞剂等，口服避孕药除外） |
| E5 | 妊娠（如适用，口头询问即可，不强制检测） |
| E6 | 测试部位（左胸/右胸/左肋）皮肤破损、皮疹或大面积疤痕 |

### 2.3 设备安全声明

- 心电采集系统由 USB 供电（5V, ≤500mA），与受试者电气隔离由 AD8232 AFE 内部仪表放大器提供（单电源 3.3V，共模抑制比 ≥86dB）。
- 柔性 PEDOT:PSS 干电极材料成分：PEDOT:PSS:PVA:TA（20:40:40）+ EG。PVA 和 TA 为常见生物相容性材料；PEDOT:PSS 在文献中广泛用于表皮电极且无细胞毒性报告（参见 Cao et al. 2022, ACS AMI）。但不排除个别受试者对成分有轻微皮肤刺激反应。
- 商用 Ag/AgCl 一次性心电电极（标准医用级），过敏风险低但凝胶可能引起个别受试者轻微不适。

---

## 3. Sample Size Justification

**n = 5–10 healthy volunteers**

本实验为设备原型对比验证（非临床试验），目标不是统计假设检验的显著性推断，而是：

1. **定性验证**：证明柔性干电极能采集与商用 Ag/AgCl 凝胶电极质量可比的心电波形（目视对比 + SNR 对比）。
2. **全链路功能验证**：验证从电极→AD8232→ESP32-S3 AI 推理→BLE→Flutter App 的完整系统链在真实人体环境下无功能故障。
3. **文献对标**：同类柔性电极原型验证论文常用样本量 n=3–15（如 Vidhya 2025 Sci Rep n=50 为 clinical study；本科/硕士原型系统论文通常 n=5–10）。

在以下场景中，n=5–10 足够：
- 每位受试者做自身对照（同一天内先后测试两种电极），消除了个体间变异。
- 采集静态（静息）+ 动态（运动）多场景心电数据，每人可产生 ≥6 段 CSV 文件。
- 正常窦性心律人群的 ECG 波形变异有限，信号质量对比（SNR / SQI / 可检测 QRS 比例）是主要的对比指标。

如需在论文中呈现定量统计对比（如 Wilcoxon 符号秩检验比较两电极 SQI），n≥10 为优。建议先执行 n=5 预实验确认流程可行后再扩展至 n=10。

---

## 4. Equipment & Preparation

### 4.1 所需设备

| Item | Specification | Quantity |
|------|---------------|----------|
| ESP32-S3-SUPERMINI 主板 | ESP32S3FH4R2, 已烧录固件 | 1 |
| AD8232 心电模拟前端模块 | 3 导联（RA/LA/LL），单通道，导联 II | 1 |
| 柔性 PEDOT:PSS 干电极 | PEDOT:PSS:PVA:TA + EG，自制（Cao 2022 配方） | 3 片（RA/LA/LL 各 1） |
| 商用 Ag/AgCl 一次性电极 | 标准医用级，凝胶型 | 3 片（RA/LA/LL 各 1） |
| 电极导线 | 与 AD8232 3.5mm 母头兼容 | 1 组 |
| USB 数据线 + 5V 电源 | 连接 ESP32-S3 至 PC | 1 |
| PC 笔记本 | 运行 `ecg_plotter.py` 记录串口数据 | 1 |
| 医用酒精棉片 | 75% 乙醇，皮肤清洁 | 若干 |
| 医用胶带 | 固定干电极（如需） | 若干 |
| 计时器 / 手机秒表 | 动作计时 | 1 |
| 椅子 + 开放空间 | 坐位静息 + 原地踏步/摆臂空间 | — |

### 4.2 实验前准备（student check）

- [ ] ESP32-S3 固件编译通过并烧录（`pio run -t upload`）
- [ ] `pc_tools/ecg_plotter.py` 运行正常，串口数据流确认
- [ ] 9 列 CSV 输出格式验证通过（`abnormal_flag` + `confidence` 列存在）
- [ ] 柔性干电极制备完成并目视检查无破损
- [ ] 商用 Ag/AgCl 电极在有效期内，包装完好
- [ ] AD8232 模块工作正常（串口观察波形有节律 QRS）
- [ ] 知情同意书打印 10 份
- [ ] 数据记录文件夹创建（`data/human_subject/S01/` … `S10/`）
- [ ] BLE 可选：Flutter App 安装到测试手机，验证连接

---

## 5. Electrode Comparison Scheme

### 5.1 采用方案：同受试者序贯测量（Within-Subject Sequential）

**理由**：
- AD8232 为单通道 AFE，无法同时接入两组电极进行双通道同步采集。
- 若使用双 AD8232 模块 + ESP32 多通道 ADC 同步，硬件改动大且不是当前系统设计目标。
- **序贯测量**是实际可行且可复现的方案：同一位受试者在同一天内、同一测试环境、相同电极位置先后测试两种电极。

### 5.2 测量顺序

**固定顺序**：先测商用 Ag/AgCl → 后测柔性 PEDOT:PSS 干电极

**理由**：
- 商用 Ag/AgCl 凝胶电极可提供更稳定的基线信号，先测作为每个受试者的"参考标准"。
- 柔性干电极可能留下轻微皮肤印记（PVA 粘附），若先测干电极再贴凝胶电极，残余物可能影响凝胶接触。
- 若凝胶电极在皮肤上留下的残留物（粘胶），换干电极前用酒精棉片彻底清洁并晾干。

### 5.3 电极放置

**统一导联位置（导联 II 近似）**：

| 电极 | 位置 |
|------|------|
| RA（右臂） | 右锁骨中线，锁骨下方约 2cm |
| LA（左臂） | 左锁骨中线，锁骨下方约 2cm |
| LL（左腿） | 左下肋骨边缘，腋前线内侧约 5cm |

> ⚠️ **重要**：两种电极须使用**完全相同的位置**标记（用记号笔标点），确保对比在同一解剖位置进行。

---

## 6. Test Protocol — Per Subject

### 6.1 总时间估算

| Phase | Duration |
|-------|----------|
| 知情同意 + 皮肤准备 + 电极粘贴 | ~10 min |
| 动作段 A-F（Ag/AgCl） | ~8 min |
| 电极更换 + 皮肤清洁 | ~5 min |
| 动作段 A-F（PEDOT:PSS） | ~8 min |
| 拆除 + 整理 | ~3 min |
| **合计 / subject** | **~35 min** |

### 6.2 动作段详细说明

每个动作段采集一个独立 CSV 文件。文件名格式：`subj_S0X_<action>_<electrode>.csv`

| 段 | 动作 | 时长 | 质控要求 | 文件命名示例 |
|----|------|------|----------|-------------|
| A | **静息坐位**（睁眼，自然呼吸，不动） | 5 min | 波形稳定后记录最后 3min；目视确认清晰的 P-QRS-T 波群 | `subj_S01_rest_AgAgCl.csv` |
| B | **手臂摆动**（坐位，双臂前后交替摆，幅度 ~45°） | 30 s | 全程记录；动作从 ~10s 开始持续至结束 | `subj_S01_armswing_AgAgCl.csv` |
| C | **恢复静息**（坐位，不动，自然呼吸） | 60 s | 心率回落至静息水平的过渡记录 | `subj_S01_recov1_AgAgCl.csv` |
| D | **原地踏步**（站立，正常步频 ~90-110 bpm 踏步，双臂自然下垂） | 60 s | 全程记录；目视确认运动噪声增幅 | `subj_S01_march_AgAgCl.csv` |
| E | **恢复静息**（坐位） | 60 s | 同上 | `subj_S01_recov2_AgAgCl.csv` |
| F | **深呼吸**（坐位，闭眼，吸气 5s→呼气 5s，节拍器辅助或实验者口令） | 30 s | 全程记录；可见呼吸性窦性心律不齐（RSA）引起的 RR 间期波动 | `subj_S01_deepbreath_AgAgCl.csv` |

> 两种电极各执行一遍 A–F。总文件数 / 受试者 = 6 × 2 = 12 个 CSV。
> n=10 → 总计 120 个 CSV 文件。

### 6.3 动作段操作 SOP

1. **开始记录前**：在 PC 端启动 `ecg_plotter.py` 并确认串口数据流正常（目视 QRS 清晰可见）。若有 BLE 记录需求，同步启动 Flutter App。
2. **分段标记**：实验者用语音标记或拍手信号（拍手在 ECG 上产生一次性尖峰作为时间标记），配合秒表记录每段起止时间。
3. **实时质控**：目视观察 `ecg_plotter.py` 波形，若信号丢失、失连或噪声过大（SQI < 0.3 持续 >10s），暂停记录检查电极接触。
4. **电极更换**：完成 Ag/AgCl 全段后，移除电极，用酒精棉片清洁皮肤，晾干 ≥2 min，然后粘贴柔性干电极，标记相同位置后开始第二轮。
5. **异常处理**：
   - 电极脱落 → 暂停 CSV 记录，重新粘贴后重启新文件，标注原文件部分数据无效。
   - 受试者感到不适 → 立即停止实验，记录停止时间点。
   - 设备发热（ESP32-S3 >65°C）→ 自动降频，临时暂停或切换 USB 供电。

---

## 7. Data Collection & Management

### 7.1 数据格式

CSV 文件，每行 9 列，无表头：

```
<clean>,<noisy>,<filtered>,<bpm>,<true_bpm>,<sqi>,<motion>,<abnormal_flag>,<confidence>
```

| Col | Field | Unit | Description |
|-----|-------|------|-------------|
| 1 | clean | V | 纯净心电 / 去偏置 ADC 信号 |
| 2 | noisy | V | 含噪声原始信号 |
| 3 | filtered | V | 数字滤波后信号（双级梳状 50/100Hz → HP 0.05Hz → LP 40Hz，与固件 filter.cpp 一致） |
| 4 | bpm | bpm | ESP32 板上能量包络心率检测值（v6） |
| 5 | true_bpm | bpm | 真实心率（仅模拟器模式有效，人体实测时写入 0 或与 bpm 相同） |
| 6 | sqi | — | 信号质量指数 [0–1] |
| 7 | motion | 0/1 | 运动标志（ESP32 板载加速度计判定） |
| 8 | abnormal_flag | 0/1 | AI 异常标志（0=正常, 1=异常） |
| 9 | confidence | — | AI 异常置信度 [0–1] |

> 人体实测中 `true_bpm` 不使用（ESP32 固件中为模拟器辅助字段），统一填写 `0`。心率以 `bpm` 列为准。

### 7.2 匿名化方案

- **受试者编号**：`S01`, `S02`, …, `S10`（按招募顺序分配）。
- **不记录**：姓名、学号、身份证号、电话号码、面部照片、视频。
- **仅记录**：性别（M/F）和年龄（岁），用于数据分组描述统计。
- **实验日志**（纸质/电子）：只按受试者编号记录实验日期、电极类型与动作段完整性与异常事件。
- **数据文件夹**结构：
  ```
  data/human_subject/
  ├── S01/
  │   ├── S01_metadata.txt         ← 性别/年龄/日期/备注
  │   ├── subj_S01_rest_AgAgCl.csv
  │   ├── subj_S01_armswing_AgAgCl.csv
  │   ├── subj_S01_recov1_AgAgCl.csv
  │   ├── subj_S01_march_AgAgCl.csv
  │   ├── subj_S01_recov2_AgAgCl.csv
  │   ├── subj_S01_deepbreath_AgAgCl.csv
  │   ├── subj_S01_rest_PEDOT.csv
  │   ├── subj_S01_armswing_PEDOT.csv
  │   ├── subj_S01_recov1_PEDOT.csv
  │   ├── subj_S01_march_PEDOT.csv
  │   ├── subj_S01_recov2_PEDOT.csv
  │   └── subj_S01_deepbreath_PEDOT.csv
  ├── S02/
  │   └── ...
  └── S10/
      └── ...
  ```

### 7.3 Metadata 模板 (`SXX_metadata.txt`)

```
Subject ID: S01
Gender: M
Age: 24
Date: 2026-08-15
Experimenter: [学生姓名]
Electrode order: Ag/AgCl first, PEDOT:PSS second
Notes: No discomfort reported. Skin clean after electrode removal.
Anomalies: None
```

---

## 8. Data Analysis Plan (Post-Acquisition)

采集完成后，由 Agent（或学生 Python 脚本）执行以下分析（对应论文 Results 章节）：

| Analysis | Method | Output |
|----------|--------|--------|
| 波形质量对比 | 计算 SNR (QRS peak / baseline RMS noise)，分动作段 | Boxplot per action per electrode |
| 信号质量指数 SQI 对比 | ESP32 板载 SQI 统计 | Mean ± SD SQI per action per electrode |
| 心率一致性 | Ag/AgCl vs PEDOT:PSS 的 BPM 散点图 + Bland-Altman | BA plot + Pearson r |
| AI 异常误报率 | 静息段 abnormal_flag=1 的比例 | FPR per electrode |
| 运动噪声鲁棒性 | 运动段（B/D/F）的 SQI 衰减量 | ΔSQI = SQI(motion) − SQI(rest) |

---

## 9. Risk Management

### 9.1 可预见的风险

| 风险 | 概率 | 严重性 | 缓解措施 |
|------|------|--------|----------|
| 皮肤轻微刺激/发红（干电极 PVA 粘附） | 低 | 轻微 | 实验后清水清洗；若持续 >30min 记录并中止后续测试 |
| 电极凝胶过敏（Ag/AgCl） | 极低 | 轻微 | 标准医用级电极；测试前询问过敏史（排除标准 E3） |
| 运动时电极脱落 | 中 | 轻微 | 医用胶带辅助固定；脱落即时重新粘贴 |
| 数据记录丢失/中断 | 低 | 中 | PC 端实时监视 CSV 文件大小增长；每段完成后即刻备份至本地 + OneDrive |
| ESP32-S3 过热 | 低 | 轻微 | 芯片内置温度保护（>65°C 自动降频）；暂停或更换 USB 线缆 |
| 受试者疲劳/不适 | 低 | 轻微 | 可在任一段落结束时提出暂停或退出，不受任何约束 |

### 9.2 应急预案

- 受试者提出退出 → **无条件终止**，不追问原因，数据按受试者意愿保留或删除。
- 设备故障 → 暂停实验，同类问题 2 次以上停止当天实验，排除故障后再恢复。
- 安全事件（皮肤严重过敏、设备短路等）→ 立即停止，联系导师，记录事件细节。

---

## 10. Execution Checklist (Student Copy)

### 阶段 1：伦理前置
- [ ] 杨辉老师确认伦理路径（简易审查 / 导师备案）
- [ ] 知情同意书定稿（含导师 + 学生联系信息）
- [ ] 获得书面/邮件批准

### 阶段 2：准备
- [ ] 柔性干电极制备（≥6 片可用）
- [ ] Ag/AgCl 电极采购（≥30 片）
- [ ] 固件烧录验证 + 串口数据流测试
- [ ] PC 端 `ecg_plotter.py` 环境就绪
- [ ] 数据文件夹创建（`data/human_subject/S01/` … `S10/`）
- [ ] 知情同意书打印 × 10

### 阶段 3：实验执行
- [ ] 受试者 S01 – 知情同意 → 实验 → 数据完整性检查
- [ ] 受试者 S02 – …
- [ ] …… S10

### 阶段 4：数据汇总
- [ ] 所有 CSV 文件验证（行数 > 0，9 列正常）
- [ ] Metadata 文件完整性检查
- [ ] 数据备份至 OneDrive + 本地 U 盘
- [ ] 提交数据至 `/data/human_subject/` 供 Agent 分析

---

## 11. References

1. Cao, J. et al. (2022). Stretchable and Self-Adhesive PEDOT:PSS Blend with High Conductivity as Epidermal Electrodes. *ACS Applied Materials & Interfaces*, 14(39), 44909–44921. DOI: [10.1021/acsami.2c11921](https://doi.org/10.1021/acsami.2c11921)
2. Vidhya, N. et al. (2025). Clinical comparison of dry vs gel electrodes for ECG acquisition. *Scientific Reports*, 15, 95057. DOI: [10.1038/s41598-025-95057-z](https://doi.org/10.1038/s41598-025-95057-z)
3. Wan, H. et al. (2021). Flexible 12-Lead ECG System Based on Stretchable Electrode Array. *Advanced Materials Technologies*, 6(7), 2100904. DOI: [10.1002/admt.202100904](https://doi.org/10.1002/admt.202100904)
4. 本系统 README.md — 系统架构 / CSV 9列格式定义 / 滤波器参数
5. 本系统 制备方案.md — PEDOT:PSS 电极材料配方与工艺

---

> **本协议 Version 1.0 — 2026-08-03**  
> 正式发表前须确认天津大学伦理简易审批/知情同意流程（与杨辉老师确认）。  
> 引用时请标注：*Research prototype — not for clinical diagnosis.*
