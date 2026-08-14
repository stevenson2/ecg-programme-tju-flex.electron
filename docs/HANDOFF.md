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

