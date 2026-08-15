# ESP32-ECG 项目交接文档（omo 迁移 + 新 agent 全貌交接）

> **生成日期**：2026-08-13
> **用途**：① 把 omo 配置（skills/agents）迁移到新 agent 环境；② 给新 agent 一份完整交接 prompt。
> **使用**：第一部分按步骤迁移配置；第二部分整体复制给新 agent 作为第一条消息。

---

## 一、omo 配置迁移方法

### 1.1 配置分三层（只有用户级需手动迁移）

| 层 | 位置 | 内容 | 迁移方式 |
|---|---|---|---|
| **项目级** | `<项目>/.opencode/` | agents/ecg-expert.md + skills/{ai-training, pc-tools, platformio-build} | ✅ git 已跟踪，`git clone/pull` 自动带上 |
| **用户级** | `C:\Users\cai\.config\opencode\skills\` | 12 个 skill：clonedeps/codemap/deepwork/docx/literature-search/pdf/pptx/reflect/simplify/skill-creator/verification-planning/worktrees/xlsx | ⚠️ 手动复制 |
| **项目状态** | `<项目>/.omo/` | plans/evidence/drafts/boulder.json + run-continuation/ | 随项目目录走；换电脑时选择性复制 |

> `~/.config/opencode/node_modules` 是依赖，**勿迁移**（新环境自动安装）。

### 1.2 用户级 skills 迁移命令

**方式 A：打包（跨机器，推荐）**
```powershell
# 旧机器打包
Compress-Archive -Path "C:\Users\cai\.config\opencode\skills" -DestinationPath "C:\Users\cai\Desktop\opencode-skills.zip"
# 新机器解压
Expand-Archive -Path "opencode-skills.zip" -DestinationPath "C:\Users\<新用户名>\.config\opencode\"
```

**方式 B：直接复制（同机/局域网）**
```powershell
robocopy "C:\Users\cai\.config\opencode\skills" "\\目标机\C$\Users\<新用户名>\.config\opencode\skills" /E
```

### 1.3 迁移前提醒

- 换电脑前先 `git push`（当前工作区有大量未提交改动，`git status` 可见）；
- `.omo/run-continuation/` 是历史会话延续 json（几十个），不需要历史延续上下文时只带
  `.omo/plans` + `.omo/evidence`，不带 `run-continuation`；
- 迁移后校验：新环境 `/skill` 能看到 3 个 project skill + 12 个 user skill，`ecg-expert`
  agent 可用。

---

## 二、完整交接 prompt（复制给新 agent）

```
# ESP32-ECG 项目交接：全貌 + 当前状态 + 下一步

## 项目根目录
C:\Users\cai\OneDrive\Desktop\Fe programme 25261\ecg-programme-tju-flex.electron-master

## 项目定位（一句话）
ESP32-S3（N16R8）+ AD8232 便携式单导联心电采集系统：500Hz 采样，板上 Pan-Tompkins
心率 + TFLite Micro INT8 逐拍异常检测（AI）+ 心律分析（停搏/过缓过速/AF/VF），
BLE 推流到 Flutter App，SPIFFS 记录 + WiFi AP 下载。产品定位=心律失常检测+显示（消费级，
非临床诊断）。论文核心创新点=训练-部署滤波链失配的系统量化与补偿。

## 三大角色
固件=src/（C++，Arduino+PlatformIO）；模型=TFLite Micro（exp7b 上板）；PC 工具链=
pc_tools/ecg_dl/（Python 训练/评估，WSL2）。

## 权威文档（读这些，不要信其他旧文档）
- docs/FINAL_RESULTS.md —— 论文权威数字（可溯源 JSON）
- TUNING_HISTORY.md —— 证据日志，最新到第四十五章
- PROJECT_SUMMARY.md —— 现状总结
- docs/PROBLEM_SOLUTION_PLAN.md —— v4 问题解决计划（硬件锁死/固件可改/产品优先）
- docs/SOFTWARE_PLAN.md —— 软件侧执行细节
- AGENTS.md —— 项目行为规范（铁律，必须读）

## 当前固件状态（本会话已改，未提交）
1. platformio.ini 恢复 N16R8（board=4d_systems_esp32s3_gen4_r8n16）
2. 显示链 HP 0.05→0.5Hz（filter.cpp，解决基线漂移）
3. AI 链解耦（filter.cpp 新增 applyFilterAI=HP0.05+LP40 独立链，与训练链一致）
4. 显示列去 HP/LP（main.cpp 串口/BLE filtered 列=noisyNoDC 梳状后原始）
5. LOD 导联脱落检测（AD8232 LO+→IO5、LO-→IO6，高=脱落，强制 abnormal=1）
6. AI 诊断命令：串口 'a'=AI 统计+最近置信度，'l'=LOD 电平，'DIAG AI 0/1'=AI 开关
7. AI 系数 fs=250 修正（AI_HP_*，ai_hp_coeffs_fs250.txt）+ exp7b INT8 上板
全部 pio run 编译通过，未提交，未烧录（除用户已烧录的）。

## 已确诊的核心问题：AI 在真实 ECG 上分布偏移（严重误报）
证据（本会话实测）：
- 模拟器正常 ECG：AI 置信度 0.066（不误报）
- 真实 AFE 正常 ECG（sqi 0.861 高质量）：AI 置信度 mean=0.895，94% 集中在 0.8~1.0，
  阈值 0.6 下误报率 59%
- 结论：训练数据（MIT-BIH/INCART/PTB 临床库）与真实采集（AD8232+商用电极+胸前贴片）
  形态分布不一致→域迁移。调阈值无效（置信度是系统性偏移到 0.9，非边界波动）。
- 必须走"模型微调"（TH §40 B 方案）：用真实 ECG 数据微调 exp7b。

## 下一步（微调流程，已定，按此执行）
1. 改固件一行：src/main.cpp ecgRecorderPushSample 记录源 filteredSample→cleanSample
   （记录"去偏置原始"供微调），pio run 编译检查，用户烧录。
2. 用户静息贴电极（'l' 确认 IO5/IO6 LOW），'m'×2 切真实 AFE，REC_START 录 2-3 分钟。
3. 下载记录（WiFi AP 192.168.4.1/api/records/{id}/data），得 250Hz int16 原始。
4. 预处理：250Hz 原始→梳状5抽头+HP0.05+LP40+因果0.5Hz(@250Hz)→R峰切250点窗口→标注正常。
5. 微调：加载 best_resnet_large_exp7b.h5，冻结骨干 lr=1e-5，混合原始训练异常拍(防遗忘)，
   产出 exp7c。
6. 评估：真实正常拍置信度应 <0.5；MIT/PTB AUC 不显著回退（exp7b PTB 0.7829 为锚点）。
7. 导出 INT8→替换 ecg_model_data.h→pio run。

## 环境与命令
- 固件编译：C:\Users\cai\.platformio\penv\Scripts\pio.exe run（PowerShell）
- 训练/评估：wsl -e bash -lc "cd /mnt/c/.../pc_tools/ecg_dl && python3 <脚本>"
  （timeout 训练 900000ms/评估 600000ms；WSL 已装 python3+tensorflow 2.21+scipy+pywt）
- 串口脚本：C:\Users\cai\AppData\Local\Temp\opencode\serial_*.py（Python 用
  C:\Users\cai\.platformio\penv\Scripts\python.exe，含 pyserial；设备 COM4，波特率 460800）
- 设备偶发卡死：串口 0 输出时用 DTR/RTS 复位（serial_reset.py）

## 硬性约束（AGENTS.md 铁律，必须遵守）
1. 固件改动仅 pio run 编译检查，不烧录、不要求用户操作硬件（烧录由用户执行）。
2. 用户手机验证期间禁串口脚本（§9）。
3. 大文件（.h5/.tflite/数据库）不提交；只提交用户明确要求的文件。
4. 数字可溯源（数字→JSON→脚本+参数）；1.000/0/100% 完美数字必须核查（§8 两条事故教训）。
5. 重大决策在 TUNING_HISTORY.md 追加"第四十六章"留痕（背景/证据/方案/结果）。
6. 决策自主执行，不暂停询问；仅无法恢复异常才暂停汇报。

## 关键决策历史（避免重复踩坑）
- 双专家 OR 已否决（TH §8.8）；3-beat/SSL/相位扰动增强/RR融合器 均已证负面（勿重试）。
- 显示链 HP 0.5Hz（基线）与 AI 链 HP 0.05Hz（训练一致）已解耦，勿再混。
- 二次 softmax 已删；AI_TRIGGER_OFFSET=6 群延迟补偿已加。
- exp7b 的 MIT 回退(-0.032)是 0.5Hz 因果 HP 的真实代价（非 bug），已诚实记录。
- 文档已整理：models_problems/solutions/必做清单/下一步工作提示词/wifi_debug_brief 等
  过时文档已删除，SUBMISSION_READY/human_qa_checklist/paper_figure_inventory 已合并为
  docs/paper_submission_status.md。

## 完成后
向用户汇报微调前后置信度对比（0.895→?）、MIT/PTB 防回归、上板决策、变更清单。
不提交代码（除非用户明确要求）。
```

---

*本文档整合 omo 配置迁移 + 新 agent 全貌交接。第一部分用于换环境迁移，第二部分复制给新 agent 建立全局认知并接手微调工作。*

---

## 三、迁移完成记录（2026-08-13，新 agent: DeepSeek Harness）

本部分为迁移执行后的留痕，不再是待办说明。第一部分描述的是"旧 agent (opencode/omo)
时代"的三层配置；迁移完成后，新 agent 的对应物如下：

| 层 | 旧位置 (已清除) | 新位置 (新 agent 配置) |
|---|---|---|
| 项目级 agents | `<项目>/.opencode/agents/ecg-expert.md` | `C:\Users\cai\.dsh\.agent-presets\ecg-expert\` (preset.yml + agent.cordis.yml, persona 为 ESP32-ECG 专家, 下次服务重启后生效) |
| 项目级 skills | `<项目>/.opencode/skills/{ai-training,pc-tools,platformio-build}/` | `<项目>/.dsh/skills/{ai-training,pc-tools,platformio-build}/` (内容已适配新环境命令, 本会话已热加载) |
| 用户级 skills | `C:\Users\cai\.config\opencode\skills\` (13 个) | `C:\Users\cai\.dsh\skills\` (13 个, 原样复制) |
| 项目状态 | `<项目>/.omo/` (plans/evidence/drafts/boulder.json/run-continuation) | 已清除; 权威知识在 docs/FINAL_RESULTS.md、TUNING_HISTORY.md (最新第四十六章)、PROJECT_SUMMARY.md |
| 串口脚本 | `C:\Users\cai\AppData\Local\Temp\opencode\serial_*.py` | `<项目>/pc_tools/serial/` (15 个脚本) |

清除前备份: `C:\Users\cai\OneDrive\Desktop\opencode-config-backup-20260813.zip`
(0.8MB, 491 文件, 不含 node_modules)。git 影响: `.opencode/` 下 4 个跟踪文件在
工作区标记删除 (未提交); `.gitignore` 已移除 `.omo/` 行; `.dsh/skills/` 与
`pc_tools/serial/` 为新增未跟踪文件。

微调流程第 1 步已完成: `src/main.cpp` 步骤3.7 记录源 `filteredSample` → `cleanSample`
(去偏置原始), `pio run` 编译通过 (SUCCESS)。第 2 步起需用户硬件配合
(烧录 + 静息贴电极录制), 详见第二部分与 TUNING_HISTORY.md 第四十六章。
---

## 四、下一步工作提示词（2026-08-13 后，exp7c + HR 重构已完成、git 已提交）

> 复制下面整段给新 agent（DeepSeek Harness）作为第一条消息，即可从当前状态接手。

```
# ESP32-ECG 项目接手提示词（exp7c + HR 重构已完成，git 已提交 b33e53c）

## 权威文档（先读，勿信其他旧文档）
- AGENTS.md —— 项目铁律（§4 只提交用户要求文件、§5 诚实纠正、§7 自主决策+TH留痕、§8 数字审计、§9 用户BLE验证期禁串口）
- docs/FINAL_RESULTS.md —— 论文权威数字（可溯源到 pc_tools/ecg_dl/models/deploy_match/*.json）
- TUNING_HISTORY.md —— 证据日志（最新到第五十五章）
- docs/manuscript_sections_1_4.md —— 论文手稿 §1-§7
- docs/paper_submission_status.md —— 论文提交单一事实来源

## 当前状态（已完成，勿重做）
- exp7c 已上板：真实数据微调（冻结骨干、只训 fc1/out、Adam lr=1e-5、稀疏CE + class_weight{0:4,1:1}、2000异常+600正常原始拍+210真实正常拍）。INT8 模型在 include/ai_inference/ecg_model_data.h（167,376 字节）。
- exp7c 锚点数字（写论文用，可溯源 deploy_match/*.json）：
  - 真实正常拍置信度 0.732→0.417（frac>0.5 从 81%→15%）
  - MIT AUC 0.8769→0.8964（INT8 0.8979）；PTB 0.8033→0.8015（INT8 0.7880，vs exp7b INT8 0.7816）
  - 上板 46 次推理 0 误报
- HR 检测器已重构（src/heartrate/heartrate.cpp）：能量包络 x² + 8-25Hz QRS带通 + 40采样MWI + millis计时RR + 形态学验证关闭(MIN_CONF_FEAT 1000)。模拟器验证 75 BPM=真实75，RR 798-804ms，SQI 0.97。
- 显示链：applyDisplayFilter = HP4 + LP40（filter.cpp/filter.h）。
- git 已提交 b33e53c（100 文件），工作区 clean。

## 待办（按优先级，自主执行不询问）
1.【最高】LUDB 重验 —— 论文 §5.4 表 T12 已脱节：§5.4 仍写旧 Pan-Tompkins 导数检测器的 LUDB 表（Se 72.9% / PPV 82.6% / BPM MAE 3.2），但板上已换成能量包络检测器。把新检测器 port 成 Python，在 LUDB（200记录、1831标注QRS峰、lead ii、500Hz）重跑，更新表 T12 + §6.5(7) 的"72.9% 敏感性"表述。旧 LUDB 验证脚本位置需先定位（pc_tools/ 或 TUNING_HISTORY 相关章节）。
2. 论文同步 —— 把 exp7c 域适配结果 + 336Hz 帧率失配发现写进 manuscript_sections_1_4.md（§4.3 训练/部署链匹配、§5.2 结果、§6.4 一致性代价、§6.5 局限），并更新 paper_submission_status.md（硬件数据采集阻塞已部分解除：真实AFE采集 + exp7c）。
3. esp_timer 硬件定时根治 —— 主循环"尽可能快"约 336Hz（AFE）/516Hz（SIM），非设计 500Hz，导致 50Hz梳状陷波漂移、记录采样率 225.7Hz。用 esp_timer 固定采样节拍到 500Hz，改后重验陷波与记录率。
4. BLE 波形阶梯感诊断 —— 需用户手机 + 串口捕捉（§9 注意与用户协调）。

## 关键根因备忘（避免重踩）
- 主循环非 500Hz（见待办3）。
- 心率检测最严重 bug：MIN_RR 用 ms 数值却和秒比较→除首拍外全拒。已修为 0.480s/2.000s。
- 导数² 放大锐利伪影 8×→改能量包络 x²。
- 5-15Hz 带通把窄 R 峰衰减到 T 波级→改 8-25Hz。
- 形态学宽度检查(40-80采样)按能量包络校准错、卡拍在 8→MIN_CONF_FEAT 1000 关闭。
- 串口会话会复位设备（DTR/RTS）；SIM 是默认模式（无需 'm'）；AFE 需 'm'×N 直到"真实AFE"。
- OneDrive 同步偶发 EIO(1175)：edit 失败就重试，或用 pwsh Add-Content 追加。

## 环境与命令
- 固件编译：pio run（PowerShell；exit code 1 但输出含 SUCCESS 是 PS 管道伪影，看 SUCCESS 文本）。只编译检查，不烧录。
- 训练/评估：wsl -e bash -lc "cd /mnt/c/.../pc_tools/ecg_dl && python3 <脚本>"（timeout 训练 900000ms/评估 600000ms）
- 串口：pc_tools/serial/*.py（COM4，460800；DTR/RTS 复位见 serial_cmd.py）

## 完成后
向用户汇报：LUDB 新表数字、论文同步段落、esp_timer 改动与编译结果、变更清单。只提交用户明确要求的文件。
```



---

## 五、未来 bug 修复与改进计划提示词（2026-08-14 后，BLE 阶梯感未根治）

> 复制下面整段给新 agent 作为第一条消息。上一段（四）的待办 ①②③ 已完成，④ BLE 阶梯感仍未根治，是本段重点。

```
# ESP32-ECG 项目接手提示词（BLE 阶梯感未根治 + 收尾改进）

## 权威文档（先读）
- AGENTS.md —— 铁律（§4 只提交用户要求文件、§5 诚实纠正、§7 自主决策+TH留痕、§8 数字审计、§9 用户BLE验证期禁串口）
- TUNING_HISTORY.md —— 最新到第五十七章（含 BLE 阶梯感诊断全过程）
- docs/HANDOFF.md（本文档）、docs/FINAL_RESULTS.md、docs/paper_submission_status.md

## 当前状态（本会话已完成，勿重做）
- LUDB 重验完成（能量包络检测器 v5）：Se 96.94% / PPV 71.03% / F1 0.820 / BPM MAE 10.17（中位 3.15 / P90 36.2）。脚本 pc_tools/ecg_dl/verify_heartrate_ludb_v5.py，结果 models/ludb_hr_v5_{eval.json,detail.csv}。已更新论文 §5.4 表 T12 + §6.5(7)（敏感性不再是限制指标，改为 Se↔PPV/BPM 权衡）。
- 论文同步完成：exp7c 域适配 + 336Hz 帧率失配已写入 manuscript_sections_1_4.md（§4.3/§5.2/§6.4/§6.5）+ paper_submission_status.md（硬件采集阻塞部分解除）。
- esp_timer 500Hz 已改（src/main.cpp）：esp_timer 周期 2000µs 固定节拍 + AFE 双重读取根治（每帧 8→4 analogRead）。pio run 编译通过，【未烧录验证】。
- 上述改动 + TUNING_HISTORY 56/57 章 + 3 个新文件均【未提交 git】；docs/manuscript_sections_1_4.md 被 gitignore（.gitignore:40），改动仅落盘不进 git。

## 【未解决·最高】BLE 波形阶梯感（重连累积退化）
症状：初次连接波形光滑；每次重连波形越来越阶梯状（一段段平直台阶）；退出 App 重连恢复光滑。

已排除（都试过、固件已烧录、仍退化）：
1. MTU：固件 setMTU(185) + App requestMtu(185)。
2. 连接优先级：App requestConnectionPriority(high) + 200ms 稳定延时 + 失败重试。
3. 外设侧连接参数更新：已移除 ble.cpp onConnect 的 esp_ble_gap_update_conn_params（改为 App 端 central 请求唯一控制）。

吞吐账（关键认知，写代码前先想清）：250Hz notify × ~45B/帧 ≈ 11KB/s。BLE 有效吞吐由连接间隔决定：15ms+DLD(251B)≈16KB/s 可承载；30ms≈6KB/s 必丢帧→阶梯。所以阶梯的本质=连接间隔过长导致 notify 丢帧，不是 MTU。

剩余假设（按可能性排序）：
A. Android BLE 栈连接间隔随重连退化（外设/App 请求都拦不住，退出 App 重建 BLE 栈才恢复）。
B. App 侧流订阅泄漏：ble_service.dart _connectionSub 每次 _connectToDevice 未 cancel 旧订阅即覆盖；_onDataReceived 每次 connect 新增 listener（旧 device 若未被 GC 可能重复收数）。
C. esp_timer 500Hz 下主循环偶发跟不上（BLE notify/串口阻塞时，esp_timer 累积节拍被一次性消费=丢样本 → notify<250Hz）。

下一步（按顺序，先确认再动手）：
1.【确认根因】串口抓 gapCallback 日志（"[BLE] conn params evt: status=… conn_int=…"），对比首次 vs 第 N 次重连的 conn_int，确认是否间隔退化。（§9：在用户不操作手机的空档做短时捕捉）
2.【若间隔退化·App侧】断开彻底清理：disconnect() 后等 connectionState 真正 disconnected 再重连；cancel 旧 _connectionSub 与 onValueReceived 订阅。
3.【若间隔退化·Android侧】requestConnectionPriority 改 BALANCED 档、或加更长稳定延时、或调整 requestMtu 与 requestConnectionPriority 的调用顺序。
4.【根治·降数据率（最稳健）】notify 降到 125Hz（s_bleNotifyDivider 2→4），App 时间轴适配（visibleSamples = timeWindow*250 → *125）。125Hz×45B≈5.5KB/s，即使 30ms 间隔也能承载。代价=波形分辨率减半（QRS ~100ms 从 25 采样降到 12，仍可显示）。
5.【根治·自适应时间轴】App 实测样本到达率，动态适配时间轴（替代硬编码 250Hz）。工作量较大但最通用。
6.【检查 DLE】确认 ESP32 BLE Arduino 2.0.0 默认启用 DLE（数据长度扩展）。若未启用，27B/包吞吐仅 ~2KB/s，任何间隔都扛不住 11KB/s。

## 改进计划（非阻塞，可择机做）
- HR 检测器 BPM 精度：LUDB 新结果暴露能量包络检测器 12% 记录 T 波双计数（BPM MAE 10.17 vs 旧 3.2）。改进方向：形态学宽度验证不直接关闭，而是按能量包络重新校准阈值（当前 MIN_CONF_FEAT=1000 是粗暴禁用），可恢复 PPV/BPM 又不丢 Se。
- esp_timer 运行时验证（需硬件）：烧录后串口实测帧率应收敛 ~500Hz、50Hz 陷波回归、录制采样率回归 250Hz。
- 论文收尾：表 T12 新数字同步进 FINAL_RESULTS.md（当前仅 manuscript 已更新）；T12 的 v4.x 旧行标注固件版本与历史溯源。
- git 提交：工作区累积未提交改动（见上），待用户确认范围后一次性提交。

## 环境与命令
- 固件编译：pio run（PowerShell；看 SUCCESS 文本，exit 1 是 PS 管道伪影）。只编译不烧录。
- App 构建：D:\flutter\bin\flutter.bat build apk --debug && flutter install -d 85a8d7ce（手机 RMX3888）。
- 训练/评估：wsl -e bash -lc "cd /mnt/c/.../pc_tools/ecg_dl && python3 <脚本>"。
- 串口：pc_tools/serial/*.py（COM4，460800；DTR/RTS 复位）。

## 完成后
向用户汇报 BLE 阶梯感根因结论与修复方案、esp_timer 验证结果。只提交用户明确要求的文件。
```


---

## 六、二轮继续结果（2026-08-14，BLE 阶梯感：订阅泄漏修复 + 125Hz 降载）

上一段（五）中"先串口确认根因"未执行（§9 需用户空档）。本会话改为代码审计驱动的确定性修复，已完成：

1. **App 重连订阅泄漏修复**：`ble_service.dart` 新增 `_notifySub`/`_connectionEpoch`/`_teardownCurrentConnection()`；每次重连先 cancel 旧 `connectionState` 与 `onValueReceived` 订阅，旧代次 disconnected 事件直接忽略；`ecg_provider.dart` 重连前也 cancel 旧 `_subscription`。
2. **固件 notify 默认 125Hz**：`src/main.cpp` `s_bleNotifyDivider` 2→4（250→125Hz）。`DIAG NOTIFY 2` 仅供 PC/串口诊断（App 已按 125Hz 编译，联调勿切回）。
3. **App 时间轴同步 125Hz**：`ecg_provider.dart` `kLiveSampleRate=125`；`WaveformDataSource` 新增 `samplesPerSecond`；`ecg_waveform.dart` 横向 dx 按动态采样率计算；回放页仍按记录 `sample_rate`。
4. 证据与决策详见 `TUNING_HISTORY.md` 第五十八章。

**未完成**：本会话执行通道不可用（bash/PowerShell/Python 均返回 win32 terminal inspection unsupported），因此 **未跑 `pio run` / `flutter test` / `git status`**；以上改动为文件级静态复核。

**下一步（新 agent 接手）**：
- 先跑 `pio run`（看 SUCCESS）与 `flutter analyze && flutter test`，修复任何编译/测试问题。
- 请用户烧录固件 + 重编 App 后复测：首次连接→断开→重连×3→退出 App 重连，波形应不再阶梯。
- 若仍阶梯：`pc_tools/serial/serial_monitor.py` 抓 `[BLE] conn params evt:`，对比首次 vs 第 N 次重连 `conn_int`（剩余假设 A=Android 间隔退化 / C=esp_timer 节拍积压）。
- 只提交用户明确要求的文件。


---

## 七、完整项目阅读与防御性修复（2026-08-14 后续）

已完成全项目静态阅读（权威文档 + 固件全模块 + App 主要服务），并额外修复 10 个防御性 bug，详见 `TUNING_HISTORY.md` 第五十九章：

1. `main.cpp` 短命令前缀匹配越界读；
2. `main.cpp` 增加 esp_timer 节拍积压丢弃诊断；
3. `ble.cpp` `sendBLEMessage` 空指针防御；
4. `ecg_recorder.cpp` 同一秒重复 START 覆盖旧记录防御；
5. `ecg_recorder.cpp` STOP 落盘失败仍写幽灵索引修复；
6. WiFi DELETE 后 recorder 计数漂移修复（新增 `ecgRecorderRefreshCount()`）；
7. `ble_service.dart` 发现不到 NUS 特征值时断开半连接再返回失败。
8. `ble_service.dart` BLE 扫描异常优雅失败；连接/服务发现失败自动进入下一轮扫描重试。
9. `ble_service.dart` 连接建立后中途异常也会先 `_teardownCurrentConnection()` 再失败返回。
10. `record_schedule_service.dart` 只有调度三项设置变更才重置录制周期，避免免打扰/音量操作误打断定时录制。


**仍待执行**：`pio run` + `flutter analyze && flutter test`（本会话执行通道不可用）；随后用户烧录/重编复测 BLE 阶梯与录制/DELETE 计数。
