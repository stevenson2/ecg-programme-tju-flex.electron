# ESP32-ECG 心电采集与 AI 异常检测系统

> **ESP32-S3 · ESP-IDF · TFLite Micro + ESP-NN · v3-A INT8 · BLE NUS · Flutter**  
> [English](README.en.md) | 中文

**便携式单导联（Lead II）心电采集 + 板载深度学习逐拍异常检测。** 500 Hz 采样，片上完成滤波、心率检测与异常推理，异常经 BLE 推送手机 App，数据以 ECGR 格式板载录制并支持 WiFi 下载。

> **固件线状态（2026-09-10）**：官方固件为 **ESP-IDF 迁移工程** `experiments/esp_idf_ecg_migration/`（2026-08-28 转正，含 AI/存储/WiFi/BLE/心率/规则组件与录制链）。
> 旧 **Arduino + PlatformIO** 线已归档至 `legacy_arduino/`，**仅历史参考，不再构建/烧录**。
> 板上模型 **v3-A** INT8（167,376 B，`models_ecg_model_v3a_int8_tflite`），固件运行 θ=0.50 + 1-of-5 + 冷却 5（以 `main.cc` 为准）。

## 当前状态（2026-09-18，R18 模型线专项）

- **R18 模型线探索（2026-09-18，负结果轮，保留 v3-A）**：按预注册四阶段（文献→预注册→冒烟→全量）对 v3-A 做架构优化探索，三条假设线全部未过门槛1——①**双头（abnormal+valid 载波有效性头）**两次迭代（λ=1.0/0.5）均 0/3：valid 头推理门控在 **6/6 训练中把语料 pure_motion FP 从 0.88-1.00 压到 0.000**（正是 §114.4 的 0x01 误擎住场景：运动假拍抬 SQI 骗过 D2 门，valid 头在全窗形态层面不被骗），但共享骨干联合训练使 abnormal 头工作点漂移（拍级 AUC 3/3 反升而事件 evF1 3/3 降 -0.06~-0.15；固件 θ=0.50 冻结不可重定阈）→ 路线关闭，**冻结骨干只训 valid 头**列为 R19 首选立项依据；②**cinc2017 穿戴域正常拍扩展**（+4000/+2000 两迭代）干净 test MIT AUC 单调劣化 -0.04~-0.10——"增强换不来部署级"（§107/§110）延伸到"有机域扩展同样不行"（62.8k 参数/6.5k 拍量级）；③**mixup α=0.2** 1/3（MIT 拍级 AUC 3/3 同向 +0.025~+0.041 为方向性正信号，非劣性未立，不采纳）。副产品：cinc2017 数据资产 + SplitGuard 集成（N 类 4862 条/37.8k 拍、泄漏审计全过）、v3-A 3-seed 基线方差（PTB 极差 0.134 披露）。板上模型/固件/App 零改动（TUNING_HISTORY §118）。**后续深入（R19M/R20M，2026-09-19，TH §119）**：沿"valid 头推理门控 6/6 压制运动伪迹"机制继续两轮预注册探索——冻结 v3-A 骨干只训 valid 头（门槛1 浮点逐位免疫、C0 端到端验证），分别配 R 居中正例与三源盲切窗正例（INCART/PTB/cinc2017 连续流），均因 TP 保护项未过而按失败规则关闭。21 次训练坐实双头设计空间图谱：**"门槛1 校准保持"与"valid 头泛化"在共享骨干上互斥**（联合训练→校准漂移；冻结→valid 头退化为形态典型性判据、压制强异常形态与带噪载波）。运动伪迹压制机制本身 21/21 成立；R21 对症候选：KD 锚定联合训练 + 含噪载波正例（valid=1）。

- **P0 protocol layer (2026-09-14)**: protocol/ecg_proto.json single source of truth + generated firmware/App/Web constants + four-end golden tests. BLE v2 HELLO + 10th asrc column, backward compatible. ECGR v2 dual-version + unified tolerant truncation. Metadata de-hardcoded via STATUS. Evidence: TUNING_HISTORY sec.115, protocol/README.md.

- **LEADOFF 报警源上线（2026-09-14，Round-H 上板验证）**：真实电极脱落（拔线实测零报警）根因为"有能量的坏信号"绕过全部四路既有源——新增第五路 `ALARM_SRC_LEADOFF`(0x10) 双判据：**A 静默型**（分析链 3s 滚动 RMS<0.05mV 持续≥2s，工频拾取被梳状滤波 annihilate 后亦归此类）+ **B 噪声型**（10s 窗≥8 合格秒：RMS 脱落带[0.05,0.25] 且 秒min-SQI<0.70 或 [无拍 且 crest<3.2]，**伪迹假拍不阻断**）。VF 被 RMS 天花板+SQI+vf抑制三重隔离；走动等效（载波+motion SNR0dB）零报警。全部阈值语料+板上标定（新语料 seg34..40，7 case×60s，既有 31 case 逐字节不变）。报警解除追加"有拍恢复"条件。已知盲区（连续中带噪声 SQI 饱和）如实披露（TUNING_HISTORY §114）。
- **推理出环 + 160MHz 重新采纳（2026-09-13，Round-F 上板验证）**：AI 推理移至核 1 独立任务（窗口交接队列 + 丢窗遥测），采样节拍改绝对期限——**稳态采样率从 467Hz 恢复到精确 500Hz**（TICK 中位 1.000s），ovr 从每窗 1 次降为偶发停顿（与时钟无关）→ **CPU 160MHz 依据新证据重新采纳**（时钟 33% 下降，invoke 变慢被异步预算吸收）。途中修复两个 bug：xTaskDelayUntil 重锚陷阱（prev 超前被内核视为 tick 回绕、永不延迟）与 VF 检测器真节奏误擎住（VF 单独降级为遥测位，需 AI/规则佐证；此前 M1 的 VF 上板验证实际从未在真节奏下做过）。录制停顿治理（Round-G）：STOP 终结异步化（worker 核 1 + SPIFFS 互斥）+ 保留策略挪至开机维护，停止窗口采样停顿 **0.8-2.2s → 0.5s**；残余为 finalize 写自身的 flash 全局禁 cache 物理约束，根治需采样解耦（timer ISR，Round-H 候选）（TUNING_HISTORY §113）。
- **Round-E（2026-09-13，相位协议+VF 门控）**：相位稠密评估协议（`phase_eval.py`，30 case×1250 窗）显示正常组 FP 相位均匀（~14% 工作点本底）、PTB test 异常记录检出相位鲁棒性已达 95%+——相位增广动机消灭，协议入库存档。固件上线 **VF rms≥0.05mV 下限门控**：平线上物理无意义的 asrc 0x08 标记消除（0/57 tick），VF 域幅度信号（合成探针 rms 0.36mV）不受影响；语料扩至 31 case（+`synthetic_vf_probe`，既有 case MD5 逐字节不变）（TUNING_HISTORY §111）。
- **Round-D（2026-09-13，勘误+负结果+固件门控）**：①勘误——语料 "ludb_normal" 载波经 RECORDS 核实为 **PTB test 异常记录**（s0010_re，真值 label=1，51/51 拍检出），早前"跨库 93% 误报"表述作废，实为正确检出（§109）；②纯噪声硬负例重训 v3-N：pure_motion FP 0.96→0.00 但干净 test 再退化（MIT evF1 -0.235）——增广路线**三连败，保留 v3-A**；③固件缓解上线：AI 密度报警源加 **min-SQI≥0.80 门控**，上板验证 pure_motion 擎住 0.767→0、全部 TP 保持、平线/5min 正常回归不变（§110）。
- **抗噪重训判定（2026-09-13，M3 负结果）**：链整形逐拍噪声增强候选 v3-B 三次迭代均未过双门槛（带噪语料全面更优，但干净 test 事件 F1 退化 -0.15~-0.23）——**保留 v3-A**（TUNING_HISTORY §107，`v3b_iteration_summary.json`）。结论页与图：`research/M4_结论.md`（§108/§109 勘误）。
- **噪声鲁棒性语料与基线（2026-09-12，M2）**：30 case 板上回放语料（MIT-100/106 载波 × 5 噪声类型 × SNR 0/10/20dB + 电极脱落/ADC 量化 + 纯噪声 + 外部库探针，`make_replay_corpus.py` 生成、int16 固件资产、`REPLAY 3..32` 可选）。v3-A 板上闭环基线（板↔PC 相关 0.934）：合成噪声下特异性/敏感度尚可，纯运动伪迹为最大 FP 源（90%）——M3/M4/Round-D 的靶点（TUNING_HISTORY §106，载波身份勘误见 §109）。
- **报警持续性修复（2026-09-12，M1 已上板验收）**：BLE 第 8 列改为固件侧**擎住报警位**（任一生理报警源触发后擎住 ≥30s：AI 密度判据 / 规则停搏-过缓-过速 / 时间停搏(电极脱落等效) / VF-VT）。修复前后对照（MIT-106 持续异常 90s）：报警位连续段 **max 1s → 70s**；正常回放（MIT-100）5 分钟**零误报**；平线段（拔接头模拟）报警 67s 连续 + 自动录制触发。规则通道（rs/vf）结果此前被丢弃、vfDetect 从未调用——均已接线。固件新增 **UART/USB 命令通道**（与 BLE 命令同集 + `MODE`/`REPLAY`/`STATUS`），pc_tools 可免重烧全自动驱动（TUNING_HISTORY §105）。
- **功耗优化 + 首次上板（2026-09-12，M0 已上板验证）**：编译档 -Og→**-O2**；BLE **两档广播**（快 30–60 ms 广播 60 s → 慢 ~1 s 常驻）；App 关 Notify 时连接间隔自动拉长到 200–400 ms；未连接时跳过显示滤波；TICK 新增 `busy=%`/`ovr=` 观测。**CPU 维持 240 MHz**：160 MHz 档上板实测 ovr 与 240 相同（均为 AI 推理在环的结构性超时，每窗 1 次，非余量问题），按验收线回退 240（TUNING_HISTORY §104）；另修复 v3-A 首次上板的 **main 任务栈溢出**（3584→8192，TFLM 推理在 main 任务内联执行）。WiFi AP 仍为开机常开（按需开关需 App 增加 WIFI_ON 入口，未做）。
- **板上模型（2026-09-11）**：工作区已从 exp7c 换成 **v3-A** INT8；N16R8 / SuperMini 双板配置见 `experiments/esp_idf_ecg_migration/FLASH_DUAL_BOARD.md`。R16 设备闭环数字仍是 **exp7c** 烧录结果，不能直接当成 v3-A 指标。
- **设备闭环（R16 / exp7c）**：官方 ESP-IDF 固件已烧录到 ESP32-S3-WROOM-1-N16R8（当时 COM3，app 1,613,824 B，
  哈希校验通过），并完成固件内置 **SIMULATOR / MIT-BIH 回放** 三模式 90 s 板上测试：
  - MIT-BIH 106（VEB 密集，真实异位拍率 49.3%）：设备 raw abnormal **51.2%**、中位置信度 0.77
    → 检出与真值吻合；
  - MIT-BIH 100（窦性，真实异位拍率 1.8%）：中位置信度 0，但 raw abnormal **16.7%**
    （PC 位级参考 11.4%）→ 约 15% 窗级假阳性，属模型工作点问题；
  - SIMULATOR（合成 ECG）：中位 0.99 / raw 61.9% → **对 AI 是 OOD，不能作正常对照**；
  - 设备 vs PC 相关 0.886/0.827、平均绝对差 0.050/0.128 → 板上实现与 PC 部署链一致。
  - 详细报告：`runs/SESSION_REPORT_R16_DEVICE.md`（研究树）。
- **研究线**：Lane B（M4 v1 门控）不变；Lane A 的 RR 形状通道为研究级正信号
  （事后两段式门禁 052 FP=0 / afdb 0.971 / ltafdb 0.916），**AF 头形态学跨域修复路线
  已于 R16 关闭**（A4 FAIL）。下一步：正常段特异性修复 + 设备 AF 正样本采集 + R17 预注册。
- **文档**：开发规则见 `docs/03_Software_Docs/AGENTS.md`（ESP-IDF 时代重写）；
  实验证据见 `docs/03_Software_Docs/TUNING_HISTORY.md` §100；权威数字见
  `docs/FINAL_RESULTS.md`。
- **公开/私密**：本仓库 `main` = 固件 + PC 工具 + App；研究树快照在
  **私有远端（R17 起研究树私有化，公开 `meeti-research` 已删除）**；会话提示词/`PLAN_STATE.json`/论文全文等私密文件永不推送
  （见 `AGENTS.md` §4）。

---

## 特性

| 类别 | 内容 |
|------|------|
| 采集 | 500 Hz 三通道（clean / noisy / filtered），信号源：模拟发生器 / 真实 AFE / MIT-BIH 回放 |
| 滤波 | 双级梳状（50/100 Hz 陷零，-119.2 dB）→ HP → LP 40 Hz |
| 心率 | 能量包络 QRS 检测 v6（LUDB：F1 0.868、Se 96.4%、BPM MAE 4.16） |
| AI | v3-A ResNet-L INT8（167,376 B），TFLite Micro + ESP-NN 推理，逐拍异常检测 |
| 心律 | 停搏 / 过缓 / 过速（规则）、房颤（CV+熵）、VF/VT（DSP 特征+LR） |
| 报警 | 5 s 锁存，BLE/串口 abnormal 标志 + 异常位图 |
| 记录 | ECGR 格式（32B 头 + int16 流 + 1 B/s 异常位图），异常触发自动录制，WiFi REST 下载 |
| 通信 | BLE NUS（TX/RX）+ WiFi HTTP；手机端 Flutter App |

---

## 快速开始

```bash
# —— 官方固件（ESP-IDF v6）——
cd experiments/esp_idf_ecg_migration
idf.py build              # 编译
idf.py -p <PORT> flash    # 烧录
idf.py monitor            # 串口监视

# —— 旧 Arduino 线（仅历史）——
cd legacy_arduino && pio run

# —— PC 绘图 / 手机 App ——
python pc_tools/ecg_plotter.py
cd ecg_app && flutter run
```

> ⚠️ AI 模型训练需 GPU（WSL2）。

---

## 架构

```mermaid
flowchart LR
    A[电极 RA/LA/RL<br/>Lead II] --> B[AFE<br/>AD8232 / 自制]
    B --> C[ADC<br/>500 Hz]
    C --> D[梳状 50/100 Hz<br/>-119.2 dB]
    D --> E[HP + LP 40 Hz]
    E --> F[心率 v6 + 心律/AF/VF]
    E --> G[2:1 抽取<br/>250 Hz]
    G --> H[250 点窗]
    H --> I[Z-score + INT8]
    I --> J[TFLite Micro + ESP-NN<br/>v3-A INT8]
    J --> K[异常概率]
    K --> L[报警锁存 5 s]
    E --> M[ECGR 录制 + 异常位图]
    F --> L
    L --> N[BLE NUS TX + 串口]
    M --> O[WiFi REST 下载]
    L --> P[Flutter App]
    O --> P
```

- **核心分工**：Core 1 = 采集/滤波/通信/存储；Core 0 = AI 推理（250 点窗，AI_STRIDE=250）。
- **数据流**：500Hz → 因果滤波 → 2:1 抽取 → 250 点窗 → 归一化 → INT8 → 推理 → 反量化取异常概率。

---

## 目录结构

```
experiments/esp_idf_ecg_migration/  # 官方 IDF 固件线
  main/                            # 主程序（采样/滤波/心率/报警/记录/BLE/WiFi）
  components/                      # ecg_ai · ecg_core · ecg_ble · ecg_wifi · ecg_storage · ecg_afe
                                   # + esp-tflite-micro · esp-nn（vendored）
legacy_arduino/                    # 旧 Arduino/PlatformIO 线（已归档）
pc_tools/                          # PC 工具：ecg_plotter + ecg_dl（训练/评估/导出）
ecg_app/                           # Flutter 手机 App
web/                               # Web 记录下载前端
docs/                              # 论文与结果文档（权威数字：docs/FINAL_RESULTS.md）
test/                              # 测试代码
papers/                            # 文献
```

---

## AI 模型与指标

**板上模型**：v3-A clean baseline（ResNet-L，~80K 参数），INT8 **167,376 B**，2026-09 上板（接替 exp7c）。论文口径最优操作点 beat θ≈0.35 / patient θ≈0.5；**固件运行 θ=0.50 + 1-of-5 + 冷却 5**（以 `main.cc` 为准，不是 0.60 / 5 拍确认）。

| 口径 | 模型 | MIT-AUC | MIT-R@0.5 | PTB-AUC | PTB-R@0.5 |
|------|------|:---:|:---:|:---:|:---:|
| 患者级清洁/未增强测试 | exp5 | 0.9295 | 0.9264 | 0.7845 | 0.6281 |
| 患者级清洁/未增强测试 | exp6 | 0.8942 | 0.9194 | **0.8232** | 0.7019 |
| 跨域参考（无 PTB 训练） | P2A | **0.9878** | 0.9312 | 0.7502 | 0.2552 |
| 部署链（D3，δ 对齐） | exp6-SGD | 0.9122 | 0.9102 | 0.7697 | 0.7069 |

> 两表口径不同（患者级清洁/未增强 vs 部署链 D3），不可直接比较。

---

## 工具链

- **PC 绘图** `pc_tools/ecg_plotter.py`：实时三通道波形 + AI 异常标签（可打包 ECG-Plotter.exe）。
- **深度学习** `pc_tools/ecg_dl/`：训练 / INT8 导出 / 评估（患者级无泄漏划分 + SplitGuard 守卫）。
- **手机 App** `ecg_app/`：BLE NUS 波形显示、AI 高亮报警、记录列表/回放。
- **跨 Shell 协作**：`scripts/project_paths.json`（唯一路径真值）+ `scripts/project_paths.{py,sh,ps1}` 读取器 + `scripts/cross_shell.py` 调度器；WSL 负责训练/评测，PowerShell 负责固件/烧录/串口/推送，规则见 `docs/03_Software_Docs/AGENTS.md`（内部）与研究树 `docs/PROJECT_STRUCTURE.md`。

---

## 文档

面向用户的说明见各子项目 `README.md`（`pc_tools/ecg_dl/`、`ecg_app/`、`web/`、`experiments/`）。

---

## 许可

[MIT](LICENSE)

## P1-1 A1 Gate Status (2026-09-14)

R18 binary gate failed the relaxed target (3 seeds, deploy chain, patient-level split):
only 1/3 seeds reached test E_A<=0.10, and that seed had Sn_A=0.058; the other two
had E_A 0.18-0.20 with Sn 0.50-0.57. Decision: FAIL_TO_3CLASS. R19 3-class retry
was aborted by host memory pressure and is incomplete. P1-3/P1-4 remain blocked until
A4 passes. Evidence: pc_tools/ecg_dl/models/gate/r18_gate_*.json; TUNING_HISTORY ch.116.

## ECG Plotter Serial Fix (2026-09-14)

Root cause: the ESP-IDF firmware only streamed waveform CSV over BLE, not over the serial console, so ECG Plotter showed no data after connecting. Firmware now accepts WAVE 1 / WAVE 0 and streams the same 10-column v2 CSV at 100 Hz when enabled (default off, so serial-log-only operation is unchanged). Plotter now initializes USB-Serial-JTAG/CH343 DTR-RTS correctly, sends WAVE 1 on connect / WAVE 0 on exit, and adds a "切换模式" button plus MODE full-line commands (sim / afe / replay_normal / replay_abnormal / replay_flat). Reflash the firmware to use this. The prebuilt ECG-Plotter.exe was not rebuilt (PyInstaller absent); run python pc_tools/ecg_plotter.py.
