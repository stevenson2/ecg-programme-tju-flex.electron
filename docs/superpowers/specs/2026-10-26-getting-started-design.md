# Getting Started 文档设计（2026-10-26）

**English title:** Getting Started docs design (2026-10-26)

**状态：** 产品讨论已拍板。本文件只定稿要写什么。实现留到用户看完本说明之后。

**本 PR 的边界：** 只新增本文件。不创建 `GETTING_STARTED.md`、`GETTING_STARTED.en.md`，不改任何 README，不改脚本，不提交权重或固件二进制。`.gitignore` 把整个 `docs/` 标成内部研究文档、默认不公开；本说明是这条规则下单独入库的一个文件，不因此把 `docs/` 里的其它材料放进来。

## 目标

让 GitHub 上的外部访客和同学尽快把研究原型跑起来：环境、数据、预处理、训练一次、导出 INT8、烧录 SuperMini。

这是研究原型，不是医疗器械。文档开头和结尾都要写明：未认证，不用于临床诊断。

## 读者

外部访客和同学。不是只给仓库维护者看的内部笔记。

默认读者会克隆仓库、能开终端、手头有一块 SuperMini。不假设他们读过实验记录，也不假设他们要复现论文表格。

## 快乐路径的范围

一条完整路径，按这个顺序：

环境 → 下载数据 → 预处理 → 训练一次 → 导出 INT8 → 烧录 SuperMini。

两条线在文档里都要走完，但它们在仓库里**不是同一条数据流**（见下文「导出和烧录怎么接上」）。读者做完两侧，才算走完快乐路径。只做烧录的人走旁路。

## 数据集

- **正文（写细）：** MIT-BIH。这是最小可运行路径：48 条记录，下载进仓库内的原始目录，预处理成 250 Hz 心拍 npz。
- **附录（写轻）：** 研究语料 INCART、PTB、PTB-XL，以及资料库根布局（Windows `C:\ecg_data`，WSL `/mnt/c/ecg_data`，代码经 `ecg_refs`）。附录只回答「这是什么、文件放哪、跑哪个预处理脚本」。不写实验史，不写论文数字。

MIT-BIH **不**放进 `C:\ecg_data`。不要把两条路径写成同一个目录。

## 文档放哪

| 文件 | 做什么 | 不做什么 |
|---|---|---|
| 新建根目录 `GETTING_STARTED.md` | 中文上手指南 | 不写产品沿革，不写论文复现 |
| 新建根目录 `GETTING_STARTED.en.md` | 英文版，章节与中文逐节对齐 | 不另起一套结构 |
| 根目录 `README.md` / `README.en.md` | 各加一行链接 | 不把教程贴进根 README |
| `pc_tools/ecg_dl/README.md` | 继续只做脚本索引 | 不改成教程 |

根 README 的那一行放在「2. 构建与使用 / 2. Build & Use」的命令块之前。中文用：

> 把环境、MIT-BIH、一次训练、INT8 导出和 SuperMini 烧录串起来，见 [GETTING_STARTED.md](GETTING_STARTED.md)（[English](GETTING_STARTED.en.md)）。

英文用：

> To walk environment → MIT-BIH → one training run → INT8 export → SuperMini flash, see [GETTING_STARTED.en.md](GETTING_STARTED.en.md) ([中文](GETTING_STARTED.md)).

除此一行，不动根 README 的产品叙述、特性表、验证表和历史归档。

`pc_tools/ecg_dl/README.md` 最多在「快速定位」表加一行，指向仓库根的 `GETTING_STARTED.md`，并写明本文仍是脚本索引。除此之外不改这个文件。

## 英文版章节标题（与中文逐节对齐）

两份指南使用同一套编号。英文标题用下表，不要重排。

| # | `GETTING_STARTED.md` | `GETTING_STARTED.en.md` |
|---|---|---|
| — | 上手指南 | Getting Started |
| 0 | 免责声明 | Disclaimer |
| 1 | 这条文档是做什么的 | What this guide is for |
| 2 | 前置条件 | Prerequisites |
| 3 | 数据：MIT-BIH | Data: MIT-BIH |
| 4 | 训练一次 | Train once |
| 5 | 导出 INT8 | Export INT8 |
| 6 | 烧录 SuperMini 并做串口冒烟 | Flash the SuperMini and smoke-test over serial |
| 7 | 旁路：跳过训练，烧录已有模型 | Side path: skip training and flash the existing model |
| 8 | 附录：INCART、PTB、PTB-XL 与资料库布局 | Appendix: INCART, PTB, PTB-XL, and the data-root layout |
| 9 | 常见坑 | Pitfalls |
| 10 | 这条路径不是什么 | What this path is not |

## 讨论里的简称，按仓库实况落成下面这些命令

实现时按本节写，不要再猜一遍。

1. 脚本索引里的 `preprocess.py` 实际文件是 `pc_tools/ecg_dl/data/preprocess.py`。包根下没有同名脚本。指南写 `python data/preprocess.py`，工作目录是 `pc_tools/ecg_dl`。
2. 仓库里没有一条更短的启动器只做 `train.py --resnet-large --patient-split`。`run_exp4_patient_clean.sh`、`run_exp5_patient_clean.sh`、`run_exp6_*.sh`、`run_exp7*.sh`、`run_clean_ab_*.sh` 都额外带了 INCART/PTB 和/或部署链参数，那是实验配方。指南不要把读者送去那些脚本。
3. 讨论里的串口冒烟「`STATUS` / `MODE replay`」：`STATUS` 照写。固件没有单独的 `MODE replay`。冒烟命令写成 `MODE replay_normal`。可以再提一句 `MODE replay_abnormal` 是异常回放段，不是必做第二步。
4. `export.py` 不带 `--h5` 时默认读 `models/best_model.h5`。本次训练不会生成这个文件名。指南必须带上 `--h5`。
5. `idf.py` 烧的是 `experiments/esp_idf_ecg_migration/main/model.cc` 里已经编进去的 v3-A，不会自动读取刚导出的 C 数组。

## 快乐路径各节要写的内容

工作目录，除烧录外，都是 `pc_tools/ecg_dl`。

### 0. 免责声明

第一屏就写，文末再写一次。口径与根 README 一致：研究原型，未经过医疗器械认证，不用于临床诊断。英文：research prototype, not a certified medical device, not for clinical diagnosis.

### 1. 这条文档是做什么的

用一两段话说明：这是「先跑起来」的路径，不是论文指标复现，也不保证与板上 v3-A 的训练配方相同。指向第 10 节，避免读者在第 4 节就把开发集准确率当成结果。

### 2. 前置条件

- Python 3.10+。
- 在 `pc_tools/ecg_dl` 执行 `pip install -r requirements.txt`。依赖文件已包含 TensorFlow（`tensorflow>=2.12,<3`）、numpy、scipy、scikit-learn、wfdb、matplotlib、pyserial。GPU 可选：有 CUDA 时 TensorFlow 会用，没有也能把这条路径跑完，只是慢。不要另写一套 GPU 安装教程。
- 可以用虚拟环境，一句话带过即可。
- ESP-IDF v6。固件树按 v6.0.1 开发。先加载 IDF 环境再调用 `idf.py`。不要把某台机器上的 `C:\Users\...` 或 `IDF_PYTHON_ENV_PATH` 写进指南。
- 串口驱动：主机要能枚举出这块 SuperMini 的端口。文档用 `<PORT>`，不要写死 COM 号。波特率 460800。

### 3. 数据：MIT-BIH

```bash
python download_full.py
python data/preprocess.py
```

- `download_full.py` 从 PhysioNet mitdb 1.0.0 下载 48 条记录的 `.hea` / `.dat` / `.atr`，落到 `pc_tools/ecg_dl/data/raw/mit-bih-arrhythmia-database/`。已存在的文件会跳过。该目录被 gitignore。
- 若 PhysioNet 因使用协议拒绝下载：让读者在浏览器接受 MIT-BIH 的数据使用协议后重跑。不要写 cookie、凭据或绕过协议的步骤。
- `data/preprocess.py` 把 360 Hz 重采样到 250 Hz，按 R 峰切成 250 点心拍，写成 npz。默认输出 `pc_tools/ecg_dl/data/processed/mit_bih_processed.npz`（`config.py` 的 `PROCESSED_DIR`）。设置了 `ECG_PROCESSED_DIR` 时写到那个目录。`*.npz` 被 gitignore。
- 不要把 `--test`（只处理 3 条记录）写成正文路径。48 条才是这条最小可运行路径。

### 4. 训练一次

```bash
python train.py --resnet-large --patient-split
```

写明产物在 `pc_tools/ecg_dl/models/`：

| 文件 | 是什么 |
|---|---|
| `best_resnet_large.h5` | 该次训练的 checkpoint |
| `final_resnet_l.h5` | 训练结束时保存的最终模型 |
| `train_history.csv` | 该次训练的曲线记录 |

这些 `.h5` 被 gitignore。指南不要写成「克隆之后仓库里已经有它们」。

指南里的训练命令就是上面这一行，不追加其它旗标。`--patient-split` 在只有 MIT-BIH 时可以跑：INCART 目录不存在时，患者映射是空的，不会因此要求读者先准备 INCART。脚本默认 200 个 epoch，早停看 `val_loss`、patience 20，所以 200 是上限，不是必须坐满的轮数。指南要写明这次会跑很久，并且不要为了缩短时间改命令或加上 `--epochs`。

这一节要明确：命令的目的是把训练入口跑通一次。终端里打印的测试数字是这次划分上的开发期数字，不是盲测，也不是板上 v3-A 的指标。

### 5. 导出 INT8

```bash
python export.py --h5 models/best_resnet_large.h5
```

`--pipeline` 可以，但同样要带这个 `--h5`。不要引导读者去跑 `export_v3a_int8.py`、`export_exp6_sgd.py`、`export_exp7c.py` 或其他实验导出脚本。

默认产物（`config.py` 的 `TFLITE_CONFIG`）：

- `pc_tools/ecg_dl/models/ecg_model.tflite`
- `pc_tools/ecg_dl/models/ecg_model_data.h`

两者都被 gitignore。这一步的完成标准是这两个文件出现在本机，不是板上模型被换掉。

INT8 校准会再读一次 `mit_bih_processed.npz`。导出时的 `ECG_PROCESSED_DIR` 必须和预处理时一致。

`export.py` 用 `load_model` 加载 h5，没有 `compile=False`，也没有登记 `FocalLoss`。而 `train.py --resnet-large` 默认按 `config.py` 的 `focal_loss.enabled = true` 用 `FocalLoss` 编译。指南仍然只指向 `export.py`，不改道到 `export_v3a_int8.py`。写指南之前，先确认这条命令能读入刚训出的 `best_resnet_large.h5`。若 Keras 报未知损失 `FocalLoss`，先单独修 `export.py` 的加载（量化不需要把训练损失编译回来，`compile=False` 即可），修好再写指南。那个修复不放进本设计 PR，也不塞进纯文档 PR。

### 6. 烧录 SuperMini 并做串口冒烟

板型：SuperMini ESP32-S3FH4R2，4MB flash / 2MB PSRAM。只写这一条构建：

```bash
cd experiments/esp_idf_ecg_migration
idf.py -B build_supermini -D SDKCONFIG=sdkconfig.supermini \
      -D SDKCONFIG_DEFAULTS=sdkconfig.defaults.supermini build
idf.py -B build_supermini -p <PORT> flash
```

串口冒烟用同一构建目录，波特率 460800：

```bash
idf.py -B build_supermini -p <PORT> monitor
```

然后发送：

- `STATUS`：回复应以 `STATUS` 开头。
- `MODE replay_normal`：回复应为 `MODE ok ...`。再发一次 `STATUS`，确认模式已经切换。

冒烟只说明固件在应答。不说明电极接触、波形保真或检测性能。

烧录用 `idf.py flash`（应用分区）。不要把 `erase_flash` 或整片擦除写成步骤。需要保留板上记录时，指向根 README 已有的数据保留说明，不要在本指南里展开。

### 7. 旁路：跳过训练，烧录已有模型

条件写清楚：新克隆**已经带有**板上 v3-A，位置是固件源码而不是训练权重。

- 已在仓库中：`experiments/esp_idf_ecg_migration/main/model.cc` 的符号 `models_ecg_model_v3a_int8_tflite`，长度 167376，声明在同目录 `model.h`。`main.cc` 用的就是这个数组。
- **不**在新克隆中：`*.h5` 和 `*.tflite` 被 gitignore。`models/best_resnet_large_clean_baseline_v3.h5` 和 `models/deploy_match/ecg_model_v3a_int8.tflite` 都不会出现。不要写「如果 h5 已经在仓库里就跳过训练」。

旁路步骤：只做第 2 节里的 ESP-IDF 与串口前置，然后做第 6 节。不需要 Python 训练依赖，不需要下载 MIT-BIH。文中说明这条旁路烧的就是板上那份 v3-A。

### 导出和烧录怎么接上

第 5 节和第 6 节都要做，但指南必须用一段话写明它们的关系，避免读者以为板上跑的是自己刚训的权重：

- 导出写到 `pc_tools/ecg_dl/models/`。
- 烧录编译的是 `main/model.cc`。
- 所以快乐路径烧上去的仍是仓库里的 v3-A。刚导出的 `ecg_model_data.h` 不会自动进固件。
- 用新数组替换 `model.cc` 不在本设计内。那会覆盖已知可用的板上模型，也不属于这次文档改动（文档改动不提交新的固件字节）。

## 附录与边界

### 8. 附录要覆盖的三套语料

路径以 `scripts/project_paths.json` 的 `ecg_refs_*` 为准。代码使用 `from ecg_refs import PTBXL_DIR, ECG_DATABASE_DIR, INCART_DIR`（`pc_tools/ecg_dl/ecg_refs.py`）。不要在指南里鼓励手写绝对路径。

| 资料 | Windows | WSL | 代码里的目录 |
|---|---|---|---|
| 根 | `C:\ecg_data` | `/mnt/c/ecg_data` | `ecg_refs_root` |
| PTB-XL | `C:\ecg_data\ptbxl` | `/mnt/c/ecg_data/ptbxl` | `PTBXL_DIR` |
| INCART | `C:\ecg_data\incart\files` | `/mnt/c/ecg_data/incart/files` | `INCART_DIR`（实现是 `ecg_refs_incart / 'files'`，文件不要只放到 `incart\` 根上） |
| PTB（ptbdb） | `C:\ecg_data\ecg_database` | `/mnt/c/ecg_data/ecg_database` | `ECG_DATABASE_DIR` |

同根下还有 `figs/`、`lib/`。指南点名即可，快乐路径和附录脚本都不用它们。

每套语料只写这三件事：

- **INCART**（St. Petersburg 12 导联心律失常库）：记录比 MIT-BIH 长，附录用它补一套导联 II 的 250 Hz 心拍。脚本 `pc_tools/ecg_dl/data/preprocess_incart.py`，输出 `incart_processed.npz`（位于 `PROCESSED_DIR`）。目录里还没有 `.dat` 时，该脚本会自己向 PhysioNet 拉取；指南用一句话说明这一点，不要再写一篇下载教程。
- **PTB（ptbdb）**：诊断形态语料（健康对照记录相对患者记录），不是像 MIT-BIH 那样带逐拍心律失常标注的库。脚本 `data/preprocess_ptb.py`，输出 `ptb_processed.npz`。指南不写它的下载步骤，只写文件应放在 `ECG_DATABASE_DIR`。
- **PTB-XL**：大规模 12 导联语料，文件放在 `PTBXL_DIR`。三个脚本各用一句话，不要展开：
  - `data/preprocess_ptbxl.py`：心拍级 250 Hz，正常对诊断异常，输出 `ptbxl_processed.npz`。
  - `data/preprocess_ptbxl_records.py`：记录级 100 Hz、五个 SCP 超类，输出 `ptbxl_records_100hz.npz`。这不是心拍快乐路径。
  - `data/preprocess_ptbxl_rhythm.py`：只保留节律诊断记录的心拍，输出 `ptbxl_rhythm_processed.npz`。

附录不要引用调参轮次、AUC、泄漏审计结论或「该不该把这些库加进训练」的实验记录。那些留在脚本索引和内部记录里。

### 9. 常见坑

指南用独立一节写这四条，每条说明现象和该怎么做，不展开成排障手册。

- **OneDrive / 9p 慢 IO。** 预处理和大 npz 不要放在 OneDrive 或 9p 挂载上跑。`ECG_PROCESSED_DIR` 会覆盖 `PROCESSED_DIR`（见 `config.py`）。克隆若在这类盘上，预处理和训练前把该变量指到本地盘。不要把某台开发机的 `/home/devcontainers/ecg_data` 写成读者必用的路径。
- **GPU 显存。** ResNet-Large 可能把显存吃满。吃满时用 CPU 把「跑起来」做完是可接受的。CPU 或中途缩短的运行印出来的数字不要拿去报告。
- **SuperMini 4MB 与 N16R8。** 快乐路径只烧 SuperMini：`sdkconfig.supermini`、`sdkconfig.defaults.supermini`、`partitions_ecg_supermini.csv`、构建目录 `build_supermini`。N16R8 是 16MB / 8MB（`sdkconfig.n16r8`、`partitions_ecg.csv`、`build_n16r8`），默认配置还会链进额外的回放语料，不能拿去烧 SuperMini。SuperMini 配置若不是 4MB 或分区表不是 `partitions_ecg_supermini.csv`，CMake 会直接失败。另一块板的细节指向 `experiments/esp_idf_ecg_migration/FLASH_DUAL_BOARD.md`，但不要把该文件里的个人安装路径抄进指南。
- **开发集数字不是盲测。** `train.py` 打印的是它刚刚划出来的那份数据上的数字。当前开发集已经反复用过。不要把这些数字写成留出测试，也不要写成跨设备泛化。

### 10. 这条路径不是什么

- **不是论文指标复现。** 患者级主结果仍以 `pc_tools/ecg_dl/README.md` 指向的 `eval_patient_split_all.py`、`eval_patient_split_noaug.py` 为准。指南不带领读者跑这些脚本。
- **不是板上 v3-A 的原配方。** 板上模型来自 `train_clean_baseline_v3.py` 的干净基线，权重文件名是 `models/best_resnet_large_clean_baseline_v3.h5`（不在克隆里），INT8 导出脚本是 `export_v3a_int8.py`，产物名是 `models/deploy_match/ecg_model_v3a_int8.tflite`（也不在克隆里）。编进固件的是 `model.cc` 里那 167376 字节。`train.py --resnet-large --patient-split` 是另一条入口，写出的是 `best_resnet_large.h5` / `final_resnet_l.h5`。
- **不是临床验证。** 串口冒烟只证明固件应答。

## 非目标

- 不重写根 README 的产品叙述。
- 不合并、不关闭、不改写草稿 PR #1、#2、#4。
- 不在文档改动里发布预编译固件、`.bin`、`.tflite`、`.h5`，也不替换 `model.cc`。
- 不把 Flutter App、`ecg_plotter.py`、固定输入板测构建写进这条路径。
- 不在 `pc_tools/ecg_dl/README.md` 里展开教程。

## 实现说明（下一轮，不在本 PR）

用户看完本说明并确认之前，不写 `GETTING_STARTED.md` 与 `GETTING_STARTED.en.md`，也不写实现计划。

确认之后，实现只动这些文件：

1. 新建 `GETTING_STARTED.md`、`GETTING_STARTED.en.md`，章节按上表，命令按「仓库实况」一节。
2. `README.md`、`README.en.md` 各加规定的那一行。
3. 可选：`pc_tools/ecg_dl/README.md` 的快速定位表加一行。没有别的改动。

读者侧的完成标准：

- 全文路径：本机有 `mit_bih_processed.npz`、`best_resnet_large.h5`、`final_resnet_l.h5`、`ecg_model.tflite`、`ecg_model_data.h`，并且 SuperMini 烧录成功，`STATUS` 与 `MODE replay_normal` 得到上面规定的回复。
- 旁路：不做数据与训练，烧录和同样的串口回复即算完成。
- 两份语言的章节编号一致。免责声明出现在开头和结尾。
