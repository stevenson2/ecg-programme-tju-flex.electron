# Literature positioning：三条线分开定位（2026-08-22）

> 目的：避免把 PTB-XL 记录级多类诊断、本项目患者级心拍二分类、ESP32-S3 INT8 部署模型放进同一张“精度排名表”。
> 本文是写作定位稿；所有数字以 `docs/01_Project_Overview/FINAL_RESULTS.md`、`models/*.json` 和 `docs/03_Software_Docs/TUNING_HISTORY.md` 为准。

---

## 1. 总原则

1. **不跨口径硬比数字。**  
   12 导联记录级 macro-AUC、单导联心拍二分类 AUC、AAMI 逐类 recall、真机 FAR 不是同一指标空间。

2. **分三条线写。**  
   - Line A：文献主流诊断模型；
   - Line B：本研究型患者级模型；
   - Line C：exp7c 部署锚点模型。

3. **当前创新点表述不再使用“架构无关普适失配”。**  
   跨架构实验支持的说法是：  
   > Deployment-chain sensitivity is architecture-, domain-, and operating-point-dependent.  

4. **exp7c 不作为 SOTA diagnostic classifier 提出。**  
   它的定位是 deployment-oriented single-lead abnormality detector，核心证据包括因果链、INT8、MCU 资源占用、真实 AFE 与时间确认策略。

---

## 2. Line A：文献主流方向

| 维度 | 常见文献设定 |
|---|---|
| 数据 | 常用 PTB-XL 等大规模 12 导联库 |
| 任务 | 多类诊断 / rhythm / MI / AAMI 心拍分类 |
| 粒度 | 记录级或心拍级 |
| 指标 | macro/micro AUC、Se/Spec/PPV、AAMI 逐类指标 |
| 部署 | 多数在 PC/server 或只做量化仿真 |
| 划分 | 有些做 patient-aware split，有些沿用 DS1/DS2 类协议 |

**写法要点**：  
不要拿文献的 PTB-XL macro-AUC 直接压我们的单拍二分类 AUC。应承认其任务更完整、数据更大，但指出其多数工作不覆盖 MCU 因果部署链和真实 AFE。

---

## 3. Line B：本项目研究型患者级结果

### 3.1 aggregate 口径

| 模型 | MIT AUC | PTB AUC | 说明 |
|---|---:|---:|---|
| P2A (部署) | 0.9878 | 0.7502 | 历史强研究模型，跨域 PTB 明显下降 |
| exp5 patient-clean | 见 FINAL_RESULTS | 见 FINAL_RESULTS | 患者级研究基线之一 |
| exp6-SGD / deploy 系列 | 见 FINAL_RESULTS | 见 FINAL_RESULTS | 部署链重训路线 |

### 3.2 跨架构对照

| 架构 | MIT Δ(B−A) | PTB Δ(B−A) | 结论 |
|---|---:|---:|---|
| lstm_cnn | +0.025 | −0.001 | 无显著失配 |
| cnn_standard | +0.005 | **−0.181** | PTB 显著失配 |
| resnet1d | +0.005 | −0.103 | 点估计下降，但 95% CI 含 0 |

结论不能写成 universal mismatch；只能写成 architecture/domain-dependent sensitivity。

### 3.3 AAMI 矩阵

产物：

```text
pc_tools/ecg_dl/models/aami_matrix_deploy_patient.json
pc_tools/ecg_dl/models/aami_matrix_deploy_patient.csv
```

关键点：

- P2A 和 exp5 在 MIT+INCART deploy 测试集上仍是最强研究锚点；
- exp7c @0.60 的低 recall 是高特异性操作点所致；
- CNN/ResNet/LSTM 的逐类行为不同，说明 aggregate AUC 会掩盖类别效应；
- 当前 AAMI matrix 使用带增强块的 deploy 测试集，若进主表建议再补 noaug 版本。

---

## 4. Line C：exp7c 部署锚点

| 维度 | exp7c 证据 |
|---|---|
| 定位 | deployment-oriented single-lead abnormality detector |
| 平台 | ESP32-S3 TFLite Micro |
| 格式 | INT8 |
| 大小 | 167,376 B |
| 输入 | 单导联心拍，因果部署链 |
| 报警语义 | θ = 0.60 + 5-beat confirmation |
| MIT AUC | float32 0.8964 / INT8 0.8979（causal-cache noaug 口径） |
| PTB AUC | float32 0.8015 / INT8 0.7880（同上） |
| 真实正常拍置信度 | float32 mean 0.4166；INT8 mean 0.4424 |
| 真机短测 | 46 次推理 0 报警 |
| 后续离线重测 | 63 窗 raw > 0.60 占 22.2%，但 5 拍确认后报警块 0 |

**写作要点**：  
exp7c 的价值不是打败 PTB-XL 多类模型，而是证明单导联异常检测可以在 MCU 上按部署链一致方式运行，并通过真实 AFE 微调和时间确认控制误报。

---

## 5. 推荐论文句式

### 不要写

> Our method outperforms patient-level ECG classifiers reported on PTB-XL.

### 应该写

> Direct comparison with large multi-lead record-level benchmarks is not meaningful because the task granularity, input configuration, label space, and evaluation metric differ. Instead, we position the system along three axes: rigorous single-lead patient-level evaluation, deployment-chain consistency, and real-time MCU feasibility.

### 关于失配

> The cross-architecture experiment does not support a universal performance drop caused by the causal deployment chain. Instead, the effect depends on pathology-related frequency content and architectural inductive bias: PTB myocardial-infarction detection is more sensitive to low-frequency morphology distortion, whereas MIT/INCART arrhythmia beats rely more on higher-frequency QRS features.

### 关于 exp7c

> exp7c is therefore presented not as a state-of-the-art diagnostic classifier but as an embedded abnormality detector whose operating point was tuned for low false-alarm operation under the deployed causal chain.

---

## 6. 下一步

1. 若 AAMI 矩阵要进入主表，补 noaug / non-augmented 版本；
2. 将 exp7c INT8 AAMI 结果与固件时间确认口径并列；
3. related work 中显式声明三线不可直接互比；
4. limitation 中说明域泛化 / adversarial patient adaptation 未在本工作中实现。
