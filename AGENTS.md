# ESP32-ECG 项目指南

## 项目概述
基于 ESP32-S3 的便携式心电采集系统，集成深度学习异常检测。
- 芯片: ESP32-S3-SUPERMINI (ESP32S3FH4R2)
- 框架: PlatformIO + Arduino
- 采样率: 500Hz 三通道 (clean/noisy/filtered); AI 输入经固件 2:1 抽取为 250Hz (250 点 = 1s 窗口)
- AI: TFLite Micro 1D-CNN INT8 推理 (部署链定稿 exp6-SGD, 单模型 163.5KB)
- BLE: Nordic UART Service (NUS)

## 项目结构
- `src/` — 固件源码
  - `main.cpp` — 主程序入口
  - `adc_afe/` — ADC/AFE 采集
  - `ai_inference/` — TFLite Micro 推理
  - `bluetooth/` — BLE 通信
  - `filter/` — 数字滤波 (HP+LP)
  - `heartrate/` — Pan-Tompkins 心率检测
  - `signal_generator/` — 模拟信号发生器
  - `thermal/` — 温度管理
- `pc_tools/` — Python PC 工具 (训练/绘图/调试)
- `ecg_app/` — Flutter 手机 App
- `test/` — 测试代码
- `include/` — 头文件
- `lib/` — 库文件

## 构建命令
- `pio run` — 编译固件
- `pio run -t upload` — 编译并上传
- `pio device monitor -b 115200` — 串口监视器
- `pio test` — 运行测试

## 编码规范
- C++: Arduino 风格，使用 `.cpp`/`.h` 扩展名
- Python: TensorFlow 训练脚本，使用 `argparse` 参数化
- 串口/BLE CSV 格式: `<clean>,<noisy>,<filtered>,<bpm>,<true_bpm>,<sqi>,<motion>,<abnormal_flag>,<confidence>`

## Agent 行为规范

### 1. 修改文件后编译检查
- 修改 C++ 文件后，自动运行 `pio run` 检查编译是否通过。
- 修改 Python 文件后，自动运行 `python -m py_compile <file>` 检查语法错误。
- 若编译/语法检查失败，必须修复后再标记任务完成。

### 2. 终端操作: Agent 全自动执行
- **所有终端操作由 Agent 自行执行**（通过 bash 工具调用 WSL2/PowerShell），
  不再交给用户手动运行。
- WSL2 命令格式：`wsl -e bash -lc "cd /mnt/c/... && <命令>"`
- PowerShell 命令格式：直接在 bash 工具中运行。
- **当前阶段工作绝对不涉及硬件部署**：不执行、也不需要用户执行上传固件
  （`pio run -t upload`）、接传感器、串口监视器等硬件操作；固件侧修改仅做
  编译检查（`pio run`），不烧录。
- **正常流程不暂停询问用户**：决策点由 Agent 自主决策（见第 7 节），阶段完成后
  按计划继续下一步；仅在**无法自动恢复的异常**（训练崩溃且无法定位、环境故障、
  文件时序锁被破坏等）时才暂停并向用户汇报。
- 命令执行需注意：
  - PowerShell 内联 Python 多行命令易因引号转义失败，**优先写脚本文件再执行**
  - WSL 首次调用可能出现 localhost NAT 警告（非致命，忽略）
  - 长时命令需设置足够 timeout（训练 900000ms=15min，评估 600000ms=10min）

### 3. 长时训练任务: Agent 后台执行 + 进度监控
- **长时训练任务由 Agent 自行执行**，不再交给用户终端。
- 执行方式：
  1. 训练前完成代码正确性验证（冒烟测试/编译检查）
  2. 用 bash 工具启动训练（WSL2，设置足够 timeout）
  3. 训练完成后自动读取 `models/train_history.csv` 汇总结果
  4. 自动运行评估脚本、更新文档
- Loss 可视化由 Agent 训练后读取 CSV 生成图表，不再需要用户开终端 B 监控。
- 若训练时间超过 bash 工具 timeout 上限，采用以下策略：
  - 用 `nohup ... &` 后台启动，记录 PID
  - 轮询检查输出文件（train_history.csv 行数 / best_*.h5 修改时间）
  - 训练完成后继续后续步骤
- **训练完成后自动**：
  - 重命名通用名文件（best_resnet_large.h5 → 实验专属名）
  - 运行 eval_patient_split_all.py 评估
  - 更新 TUNING_HISTORY.md / ROADMAP.md / PATIENT_SPLIT_PROGRESS.md
  - git status 确认变更范围
  - 向用户汇总结果（仅汇报，不等待指示；按计划继续下一步）

### 4. Git 状态检查
- 每次做出**大更改之前**，先运行 `git status` 检查当前工作区状态。
- 每次修改**成功生效之后**（编译通过/实验完成/归档完成），再运行 `git status`
  确认变更范围符合预期。
- 只提交用户明确要求的文件；大文件（模型权重 .h5/.tflite 等）不得擅自提交。

### 5. 回答可信度要求
- 回答技术问题/做项目决策时，在必要的情形使用**联网搜索**（webfetch）与
  **文献阅读**来支撑结论，保证内容可信度。
- 发现用户陈述中的错误时，必须**直接指出**，不得掩盖或附和。
- 不确定的信息必须明确标注"不确定"，不得编造数据、引用或指标。

### 6. 参考文献规范
- **给出的参考文献不得编造**。每条引用必须真实存在，作者、年份、期刊、标题
  均须与原文一致。
- **如能下载**，将 PDF 下载至 `C:\Users\cai\OneDrive\Desktop\Fe programme 25261\ecg-programme-tju-flex.electron-master\papers`
  目录，并在引用中标注"已下载"。
- **如不能下载**，给出文章的 DOI（或正式 URL），并进行联网搜索验证其存在。
  若不能搜索到该文章，则**该文章不予采纳**，不得列入引用。
- **arXiv 论文限制**：除非该主题的联网搜索结果 < 2 条，否则一般不采纳 arXiv
  预印本论文（优先采纳正式发表的期刊/会议论文）。若引用 arXiv 论文，须标注
  "预印本，未经同行评审"。
- 引用格式：作者 (年份). 标题. *期刊*, 卷(期), DOI. [已下载/仅DOI]

### 7. 决策自主执行（不询问用户）
- **所有决策由 Agent 自动执行**：实验方案选择、是否重训/重跑、训练配置调整、
  模型取舍、评估口径、文档更新等均由 Agent 依据证据自主决定，无需用户确认；
  阶段完成后自动执行计划中的下一步，不停下来询问用户。
- **重大决策前置检查（必须全部完成）**：
  1. **联网搜索 + 文献阅读**：搜索最新资料与文献支撑结论（遵循第 5、6 节
     规范），不得凭直觉拍板；
  2. **git status 检查**：确认当前工作区状态与变更范围，避免在脏状态下做决策。
- **更改文档留痕（每次重大决策必须执行）**：在根目录 `TUNING_HISTORY.md`
  末尾追加详细章节（沿用"第 N 章"章节惯例），记录：决策背景、前置检查证据
  （文献结论 + git 状态）、执行方案、执行结果与后续影响（指标变化、模型取舍等）。
- 重大决策示例：是否重训/重跑某模型、调整训练超参与回调、更换评估口径、
  改变数据处理流程、影响论文结论的模型取舍。
- 里程碑结果可在执行间隙向用户汇总，但不阻塞流程、不等待指示。
