# Getting Started Docs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship visitor-facing `GETTING_STARTED.md` / `GETTING_STARTED.en.md` plus one-line root README links so a GitHub visitor can run environment → MIT-BIH → one train → INT8 export → SuperMini flash (or the skip-train side path).

**Architecture:** Docs-only change grounded in `docs/superpowers/specs/2026-10-26-getting-started-design.md` (PR #6). One pre-doc code fix for `export.py` FocalLoss load (separate small PR). Guides live at repo root; root READMEs get a single link line before §2 command blocks; `pc_tools/ecg_dl/README.md` stays a script index (optional one-line pointer).

**Tech Stack:** Markdown; Python / TensorFlow training stack under `pc_tools/ecg_dl`; ESP-IDF v6 SuperMini flash under `experiments/esp_idf_ecg_migration`.

## Global Constraints

- Audience: GitHub visitors / external students; not maintainers-only notes.
- Happy path order: env → download MIT-BIH → preprocess → `train.py --resnet-large --patient-split` → `export.py --h5 models/best_resnet_large.h5` → SuperMini flash + serial smoke.
- Working directory for train/export/preprocess: `pc_tools/ecg_dl` (except flash, which is `experiments/esp_idf_ecg_migration`).
- Disclaimer (research prototype, not certified, not for clinical diagnosis) at start and end of both guides; match root README tone.
- CN/EN section numbers must match the design table (0–10). English titles are fixed by the design; do not reorder.
- Export ≠ auto-replace firmware: flash still uses in-tree `main/model.cc` v3-A (167,376 bytes). Say this explicitly between §5 and §6.
- Do not send readers to `run_exp*.sh`, `export_v3a_int8.py`, `export_exp6_sgd.py`, `export_exp7c.py`, or `--test` preprocess.
- Serial smoke: `STATUS` and `MODE replay_normal` (not bare `MODE replay`). Baud 460800. Port placeholder `<PORT>`.
- No product narrative rewrite of root READMEs beyond the one link line. No Flutter / `ecg_plotter.py` / fixed-input board-test in this path. No shipping `.h5` / `.tflite` / `.bin` / new `model.cc`.
- `docs/` is gitignored except intentionally force-added tracked exceptions (this plan and the design spec). Force-add plan/spec paths when committing.
- Spec source of truth: `docs/superpowers/specs/2026-10-26-getting-started-design.md` on branch `cursor/getting-started-design-d640` / [PR #6](https://github.com/stevenson2/ecg-programme-tju-flex.electron/pull/6).

## File Structure

| File | Responsibility |
|------|----------------|
| `pc_tools/ecg_dl/export.py` | Pre-doc fix: load H5 with `compile=False` so FocalLoss-trained `best_resnet_large.h5` can export |
| `GETTING_STARTED.md` (new, repo root) | Chinese visitor guide, sections 0–10 |
| `GETTING_STARTED.en.md` (new, repo root) | English guide, same section numbers/titles as design table |
| `README.md` | Insert one Chinese link line before §2 command block |
| `README.en.md` | Insert one English link line before §2 command block |
| `pc_tools/ecg_dl/README.md` | Optional: one row in 快速定位 table pointing to root guide |

---

### Task 0: Fix `export.py` H5 load for FocalLoss (separate small PR)

**Files:**
- Modify: `pc_tools/ecg_dl/export.py` — `convert_to_tflite` load line only
- Test: local smoke after a train artifact exists, or unit-style load of any FocalLoss-compiled H5 if available; if no H5 on the machine, verify by code review + a one-liner that mocks `load_model` kwargs

**Interfaces:**
- Consumes: H5 written by `train.py --resnet-large` (compiled with `FocalLoss` when `config.py` `focal_loss.enabled` is true)
- Produces: `tf.keras.models.load_model(h5_path, compile=False)` so quantization does not need the training loss registered

- [ ] **Step 1: Confirm the failure mode in code**

Open `pc_tools/ecg_dl/export.py` and locate:

```python
model = tf.keras.models.load_model(h5_path)
```

inside `convert_to_tflite`. Note there is no `compile=False` and no `custom_objects` for `FocalLoss`.

- [ ] **Step 2: Apply the minimal fix**

Replace that single call with:

```python
model = tf.keras.models.load_model(h5_path, compile=False)
```

Do not change CLI flags, default paths, quantization, or C-array export. Do not touch `export_v3a_int8.py` or other exporters in this task.

- [ ] **Step 3: Verify**

If a `models/best_resnet_large.h5` (or any FocalLoss-trained H5) is available under `pc_tools/ecg_dl`:

```bash
cd pc_tools/ecg_dl
python -c "import tensorflow as tf; m=tf.keras.models.load_model('models/best_resnet_large.h5', compile=False); print(type(m).__name__, 'ok')"
```

Expected: prints a model class name and `ok` (no `Unknown loss function: FocalLoss`).

If no H5 is present, grep to confirm the only `load_model` in `convert_to_tflite` now passes `compile=False`, and skip runtime.

- [ ] **Step 4: Commit on its own branch / PR**

```bash
git checkout -b fix/export-compile-false
git add pc_tools/ecg_dl/export.py
git commit -m "fix(ecg_dl): load H5 with compile=False in export.py

Train --resnet-large compiles with FocalLoss; export only needs weights
for INT8 conversion, so skip recompilation."
```

Open a PR separate from the docs PR. Do not mix this with GETTING_STARTED files.

---

### Task 1: Create `GETTING_STARTED.md` (Chinese)

**Files:**
- Create: `GETTING_STARTED.md` (repo root)
- Reference: `docs/superpowers/specs/2026-10-26-getting-started-design.md`

**Interfaces:**
- Consumes: pinned commands and section outline from the design spec
- Produces: complete Chinese guide with sections titled exactly as the CN column of the design table

- [ ] **Step 1: Create the file with title, disclaimer, and §1–§2**

Write `GETTING_STARTED.md` starting with:

```markdown
# 上手指南

## 0. 免责声明

本仓库是**研究原型**，**未经过医疗器械认证**，**不用于临床诊断**。下文路径只帮助你在本机把训练与烧录跑通，不构成任何医疗有效性声明。

## 1. 这条文档是做什么的

这份指南把「环境 → 下载 MIT-BIH → 预处理 → 训练一次 → 导出 INT8 → 烧录 SuperMini」串成一条最小可跑通路径，面向 GitHub 上的外部访客和同学。

它**不是**论文指标复现，也**不保证**与板上 v3-A 的训练配方相同。终端里打印的测试数字是本次划分上的开发期数字——详见第 10 节，不要在第 4 节就把它们当成最终结果。

若你只想烧录仓库里已有的板上模型、跳过数据和训练，直接看第 7 节旁路。

## 2. 前置条件

- **Python 3.10+**。在 `pc_tools/ecg_dl` 执行：

```bash
cd pc_tools/ecg_dl
python -m venv .venv   # 可选
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

  `requirements.txt` 已包含 TensorFlow（`tensorflow>=2.12,<3`）、numpy、scipy、scikit-learn、wfdb、matplotlib、pyserial。有 CUDA 时 TensorFlow 会用 GPU；没有也能跑完这条路径，只是更慢。本指南不另写 GPU 安装教程。

- **ESP-IDF v6**（固件按 v6.0.1 开发）。先加载 IDF 环境再调用 `idf.py`。不要把某台机器上的用户目录或 `IDF_PYTHON_ENV_PATH` 写进命令。

- **串口**：主机要能枚举出这块 SuperMini 的端口。下文一律写 `<PORT>`（例如 Linux 上的 `/dev/ttyACM0`，Windows 上的 `COM5`），波特率 **460800**。
```

- [ ] **Step 2: Add §3 Data MIT-BIH and §4 Train once**

Append:

```markdown
## 3. 数据：MIT-BIH

工作目录：`pc_tools/ecg_dl`。

```bash
python download_full.py
python data/preprocess.py
```

- `download_full.py` 从 PhysioNet mitdb 1.0.0 下载 48 条记录的 `.hea` / `.dat` / `.atr`，落到 `pc_tools/ecg_dl/data/raw/mit-bih-arrhythmia-database/`。已存在的文件会跳过。该目录被 gitignore。
- 若 PhysioNet 因使用协议拒绝下载：在浏览器打开 PhysioNet，接受 MIT-BIH 的数据使用协议后重跑上面的下载命令。不要用 cookie、凭据或任何绕过协议的做法。
- `data/preprocess.py`（文件在 `pc_tools/ecg_dl/data/preprocess.py`）把 360 Hz 重采样到 250 Hz，按 R 峰切成 250 点心拍，默认写出 `pc_tools/ecg_dl/data/processed/mit_bih_processed.npz`。若设置了环境变量 `ECG_PROCESSED_DIR`，则写到该目录。`*.npz` 被 gitignore。
- **不要**把 `--test`（只处理 3 条记录）当成正文路径；48 条才是这条最小可运行路径。
- MIT-BIH **不**放进附录里的 `C:\ecg_data` 资料库根；两条路径不要混成同一个目录。

## 4. 训练一次

仍在 `pc_tools/ecg_dl`：

```bash
python train.py --resnet-large --patient-split
```

只使用这一行，不要追加其它旗标，也不要改去跑 `run_exp*.sh` / `run_clean_ab_*.sh`（那些是带 INCART/PTB 或部署链参数的实验配方）。

产物在 `pc_tools/ecg_dl/models/`（`.h5` 被 gitignore，克隆后仓库里**没有**它们）：

| 文件 | 是什么 |
|---|---|
| `best_resnet_large.h5` | 该次训练的 checkpoint |
| `final_resnet_l.h5` | 训练结束时保存的最终模型 |
| `train_history.csv` | 该次训练的曲线记录 |

`--patient-split` 在只有 MIT-BIH 时可以跑：INCART 目录不存在时患者映射为空，不会因此要求你先准备 INCART。脚本默认最多 200 个 epoch，早停看 `val_loss`、patience 20，所以 200 是上限。这次会跑很久——**不要**为了缩短时间改命令或加上 `--epochs`。

本节目的是把训练入口跑通一次。终端打印的测试数字是这次划分上的开发期数字，**不是**盲测，也**不是**板上 v3-A 的指标。
```

- [ ] **Step 3: Add §5 Export, export↔flash bridge, §6 Flash, §7 Side path**

Append:

```markdown
## 5. 导出 INT8

仍在 `pc_tools/ecg_dl`：

```bash
python export.py --h5 models/best_resnet_large.h5
```

`--pipeline` 也可以，但同样必须带这个 `--h5`。不带 `--h5` 时默认读 `models/best_model.h5`，而本次训练**不会**写出那个文件名。不要改去跑 `export_v3a_int8.py`、`export_exp6_sgd.py`、`export_exp7c.py` 或其他实验导出脚本。

默认产物（见 `config.py` 的 `TFLITE_CONFIG`）：

- `pc_tools/ecg_dl/models/ecg_model.tflite`
- `pc_tools/ecg_dl/models/ecg_model_data.h`

两者都被 gitignore。这一步的完成标准是这两个文件出现在本机，**不是**板上模型被换掉。

INT8 校准会再读一次 `mit_bih_processed.npz`。导出时的 `ECG_PROCESSED_DIR` 必须和预处理时一致。

### 导出和烧录怎么接上

第 5 节和第 6 节都要做，但它们在仓库里**不是同一条数据流**：

- 导出写到 `pc_tools/ecg_dl/models/`。
- 烧录编译的是 `experiments/esp_idf_ecg_migration/main/model.cc`。
- 因此快乐路径烧上去的仍是仓库里的 **v3-A**。刚导出的 `ecg_model_data.h` **不会**自动进固件。
- 用新数组替换 `model.cc` 不在本指南范围内（那会覆盖已知可用的板上模型，也不属于这次文档改动）。

## 6. 烧录 SuperMini 并做串口冒烟

板型：SuperMini ESP32-S3FH4R2（4MB flash / 2MB PSRAM）。只使用下面这一条构建：

```bash
cd experiments/esp_idf_ecg_migration
idf.py -B build_supermini -D SDKCONFIG=sdkconfig.supermini \
      -D SDKCONFIG_DEFAULTS=sdkconfig.defaults.supermini build
idf.py -B build_supermini -p <PORT> flash
```

串口冒烟（同一构建目录，波特率 460800）：

```bash
idf.py -B build_supermini -p <PORT> monitor
```

然后发送：

- `STATUS`：回复应以 `STATUS` 开头。
- `MODE replay_normal`：回复应为 `MODE ok ...`。再发一次 `STATUS`，确认模式已经切换。

（可选了解）`MODE replay_abnormal` 是异常回放段，不是必做第二步。固件没有单独的 `MODE replay`。

冒烟只说明固件在应答，不说明电极接触、波形保真或检测性能。

烧录用 `idf.py flash`（应用分区）。不要把 `erase_flash` 或整片擦除写成步骤。需要保留板上记录时，看根 README 已有的数据保留说明。

## 7. 旁路：跳过训练，烧录已有模型

新克隆**已经带有**板上 v3-A，位置是固件源码，不是训练权重：

- **在仓库中：** `experiments/esp_idf_ecg_migration/main/model.cc` 的符号 `models_ecg_model_v3a_int8_tflite`，长度 **167376**，声明在同目录 `model.h`。`main.cc` 用的就是这个数组。
- **不在新克隆中：** `*.h5` 和 `*.tflite` 被 gitignore。不要写「如果 h5 已经在仓库里就跳过训练」。

旁路步骤：只做第 2 节里的 ESP-IDF 与串口前置，然后做第 6 节。不需要 Python 训练依赖，不需要下载 MIT-BIH。这条旁路烧的就是板上那份 v3-A。
```

- [ ] **Step 4: Add §8 Appendix, §9 Pitfalls, §10 What this is not, closing disclaimer**

Append:

```markdown
## 8. 附录：INCART、PTB、PTB-XL 与资料库布局

研究语料走资料库根（**不是** MIT-BIH 的 `data/raw/...`）。路径以 `scripts/project_paths.json` 的 `ecg_refs_*` 为准；代码使用：

```python
from ecg_refs import PTBXL_DIR, ECG_DATABASE_DIR, INCART_DIR
```

（定义在 `pc_tools/ecg_dl/ecg_refs.py`。）不要手写绝对路径进脚本。

| 资料 | Windows | WSL | 代码里的目录 |
|---|---|---|---|
| 根 | `C:\ecg_data` | `/mnt/c/ecg_data` | `ecg_refs_root` |
| PTB-XL | `C:\ecg_data\ptbxl` | `/mnt/c/ecg_data/ptbxl` | `PTBXL_DIR` |
| INCART | `C:\ecg_data\incart\files` | `/mnt/c/ecg_data/incart/files` | `INCART_DIR`（实现是 `ecg_refs_incart / 'files'`，文件不要只放到 `incart\` 根上） |
| PTB（ptbdb） | `C:\ecg_data\ecg_database` | `/mnt/c/ecg_data/ecg_database` | `ECG_DATABASE_DIR` |

同根下还有 `figs/`、`lib/`；快乐路径和附录脚本都不用它们。

- **INCART**（St. Petersburg 12 导联心律失常库）：记录比 MIT-BIH 长，用于补一套导联 II 的 250 Hz 心拍。脚本 `pc_tools/ecg_dl/data/preprocess_incart.py`，输出 `incart_processed.npz`（位于 `PROCESSED_DIR`）。目录里还没有 `.dat` 时，该脚本会自己向 PhysioNet 拉取。
- **PTB（ptbdb）**：诊断形态语料（健康对照相对患者记录），不是像 MIT-BIH 那样带逐拍心律失常标注的库。脚本 `data/preprocess_ptb.py`，输出 `ptb_processed.npz`。本附录不写下载步骤，只要求文件放在 `ECG_DATABASE_DIR`。
- **PTB-XL**：大规模 12 导联语料，文件放在 `PTBXL_DIR`。三个脚本各用一句话：
  - `data/preprocess_ptbxl.py`：心拍级 250 Hz，正常对诊断异常 → `ptbxl_processed.npz`
  - `data/preprocess_ptbxl_records.py`：记录级 100 Hz、五个 SCP 超类 → `ptbxl_records_100hz.npz`（不是心拍快乐路径）
  - `data/preprocess_ptbxl_rhythm.py`：只保留节律诊断记录的心拍 → `ptbxl_rhythm_processed.npz`

附录不引用调参轮次、AUC、泄漏审计结论或「该不该把这些库加进训练」的实验记录。

## 9. 常见坑

- **OneDrive / 9p 慢 IO。** 预处理和大 npz 不要放在 OneDrive 或 9p 挂载上跑。`ECG_PROCESSED_DIR` 会覆盖 `PROCESSED_DIR`（见 `config.py`）。克隆若在这类盘上，预处理和训练前把该变量指到本地盘。不要把某台开发机的专用路径当成读者必用路径。
- **GPU 显存。** ResNet-Large 可能把显存吃满。吃满时用 CPU 把「跑起来」做完是可接受的。CPU 或中途缩短的运行印出来的数字不要拿去报告。
- **SuperMini 4MB 与 N16R8。** 快乐路径只烧 SuperMini：`sdkconfig.supermini`、`sdkconfig.defaults.supermini`、`partitions_ecg_supermini.csv`、构建目录 `build_supermini`。N16R8（`sdkconfig.n16r8`、`partitions_ecg.csv`、`build_n16r8`）默认还会链进额外回放语料，不能拿去烧 SuperMini。另一块板的细节见 `experiments/esp_idf_ecg_migration/FLASH_DUAL_BOARD.md`，但不要把该文件里的个人安装路径抄进命令。
- **开发集数字不是盲测。** `train.py` 打印的是它刚刚划出来的那份数据上的数字。当前开发集已经反复用过。不要把这些数字写成留出测试，也不要写成跨设备泛化。

## 10. 这条路径不是什么

- **不是论文指标复现。** 患者级主结果仍以 `pc_tools/ecg_dl/README.md` 指向的 `eval_patient_split_all.py`、`eval_patient_split_noaug.py` 为准。本指南不带领你跑这些脚本。
- **不是板上 v3-A 的原配方。** 板上模型来自 `train_clean_baseline_v3.py` 的干净基线，权重文件名是 `models/best_resnet_large_clean_baseline_v3.h5`（不在克隆里），INT8 导出脚本是 `export_v3a_int8.py`，产物名是 `models/deploy_match/ecg_model_v3a_int8.tflite`（也不在克隆里）。编进固件的是 `model.cc` 里那 167376 字节。`train.py --resnet-large --patient-split` 是另一条入口，写出的是 `best_resnet_large.h5` / `final_resnet_l.h5`。
- **不是临床验证。** 串口冒烟只证明固件应答。

## 免责声明（再申明）

本仓库是研究原型，未经过医疗器械认证，不用于临床诊断。
```

- [ ] **Step 5: Spec checklist for the Chinese file**

Verify all of the following are present; fix any miss before commit:

1. Sections 0–10 with exact CN titles from the design table
2. Commands: `python data/preprocess.py`, `python train.py --resnet-large --patient-split`, `python export.py --h5 models/best_resnet_large.h5`, SuperMini `idf.py` build/flash/monitor as in design
3. Serial: `STATUS` + `MODE replay_normal`
4. Export↔flash bridge paragraph
5. Side path names `models_ecg_model_v3a_int8_tflite` / 167376
6. Disclaimer at start and end
7. No `run_exp*.sh`, no bare `MODE replay`, no `--test` as main path, no Flutter/plotter

- [ ] **Step 6: Commit Chinese guide**

```bash
git add GETTING_STARTED.md
git commit -m "docs: add Chinese GETTING_STARTED visitor guide"
```

---

### Task 2: Create `GETTING_STARTED.en.md` (English, section-aligned)

**Files:**
- Create: `GETTING_STARTED.en.md` (repo root)
- Mirror: `GETTING_STARTED.md` from Task 1

**Interfaces:**
- Consumes: same facts/commands as Task 1
- Produces: English guide whose `#` and section titles match the EN column of the design table exactly

- [ ] **Step 1: Write the English file with fixed titles**

Create `GETTING_STARTED.en.md` with this skeleton and full prose (commands identical to CN; only narrative language differs):

```markdown
# Getting Started

## 0. Disclaimer

This repository is a **research prototype**. It is **not a certified medical device** and is **not for clinical diagnosis**. The path below only helps you run training and flashing on your machine; it makes no medical-validity claim.

## 1. What this guide is for

This guide chains environment → download MIT-BIH → preprocess → train once → export INT8 → flash the SuperMini into one minimal runnable path for GitHub visitors and fellow students.

It is **not** a paper-metric reproduction, and it does **not** claim to match the on-board v3-A training recipe. Numbers printed in the terminal are development-set figures for this split — see §10; do not treat §4 output as a final result.

If you only want to flash the in-tree on-board model and skip data/training, jump to §7.

## 2. Prerequisites

- **Python 3.10+**. From `pc_tools/ecg_dl`:

```bash
cd pc_tools/ecg_dl
python -m venv .venv   # optional
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

  `requirements.txt` already includes TensorFlow (`tensorflow>=2.12,<3`), numpy, scipy, scikit-learn, wfdb, matplotlib, and pyserial. TensorFlow will use a GPU when CUDA is available; without one you can still finish this path, just more slowly. This guide does not add a separate GPU install tutorial.

- **ESP-IDF v6** (firmware developed against v6.0.1). Load the IDF environment before calling `idf.py`. Do not paste machine-specific user paths or `IDF_PYTHON_ENV_PATH` into the commands.

- **Serial**: your host must enumerate this SuperMini’s port. Below, always write `<PORT>` (e.g. `/dev/ttyACM0` on Linux, `COM5` on Windows). Baud rate **460800**.

## 3. Data: MIT-BIH

Working directory: `pc_tools/ecg_dl`.

```bash
python download_full.py
python data/preprocess.py
```

- `download_full.py` downloads all 48 mitdb 1.0.0 records (`.hea` / `.dat` / `.atr`) from PhysioNet into `pc_tools/ecg_dl/data/raw/mit-bih-arrhythmia-database/`. Existing files are skipped. That directory is gitignored.
- If PhysioNet refuses the download because of the data-use agreement: accept the MIT-BIH agreement in a browser, then re-run the download. Do not use cookies, credentials, or any agreement bypass.
- `data/preprocess.py` (file at `pc_tools/ecg_dl/data/preprocess.py`) resamples 360 Hz → 250 Hz, cuts 250-sample beats on R-peaks, and writes `pc_tools/ecg_dl/data/processed/mit_bih_processed.npz` by default. If `ECG_PROCESSED_DIR` is set, it writes there instead. `*.npz` files are gitignored.
- Do **not** treat `--test` (3 records only) as the main path; all 48 records are the minimal runnable path.
- MIT-BIH does **not** go under the appendix data root `C:\ecg_data`; keep the two layouts separate.

## 4. Train once

Still in `pc_tools/ecg_dl`:

```bash
python train.py --resnet-large --patient-split
```

Use exactly this line. Do not add other flags, and do not redirect readers to `run_exp*.sh` / `run_clean_ab_*.sh` (those are experiment recipes with INCART/PTB and/or deploy-chain args).

Artifacts land in `pc_tools/ecg_dl/models/` (`.h5` files are gitignored — a fresh clone does **not** contain them):

| File | What it is |
|---|---|
| `best_resnet_large.h5` | checkpoint from this run |
| `final_resnet_l.h5` | model saved at end of training |
| `train_history.csv` | training curve log |

`--patient-split` works with MIT-BIH alone: if the INCART directory is missing, the patient map is empty and you are not required to prepare INCART first. Default is up to 200 epochs with early stopping on `val_loss` (patience 20), so 200 is a ceiling. This run takes a long time — do **not** shorten it with `--epochs` or by changing the command.

The goal of this section is to exercise the training entry once. Printed test numbers are development-set figures for this split — **not** a blind test and **not** on-board v3-A metrics.

## 5. Export INT8

Still in `pc_tools/ecg_dl`:

```bash
python export.py --h5 models/best_resnet_large.h5
```

`--pipeline` is fine too, but it must carry the same `--h5`. Without `--h5`, export defaults to `models/best_model.h5`, which this training run does **not** write. Do not send readers to `export_v3a_int8.py`, `export_exp6_sgd.py`, `export_exp7c.py`, or other experiment exporters.

Default outputs (`TFLITE_CONFIG` in `config.py`):

- `pc_tools/ecg_dl/models/ecg_model.tflite`
- `pc_tools/ecg_dl/models/ecg_model_data.h`

Both are gitignored. Done means those two files exist on disk — **not** that the on-board model was replaced.

INT8 calibration re-reads `mit_bih_processed.npz`. `ECG_PROCESSED_DIR` at export time must match preprocess.

### How export connects to flash

§5 and §6 are both part of the happy path, but they are **not** the same data flow in the repo:

- Export writes under `pc_tools/ecg_dl/models/`.
- Flash compiles `experiments/esp_idf_ecg_migration/main/model.cc`.
- So the happy-path flash still runs in-tree **v3-A**. The just-exported `ecg_model_data.h` does **not** enter the firmware automatically.
- Replacing `model.cc` with a new array is out of scope for this guide.

## 6. Flash the SuperMini and smoke-test over serial

Board: SuperMini ESP32-S3FH4R2 (4MB flash / 2MB PSRAM). Use only this build:

```bash
cd experiments/esp_idf_ecg_migration
idf.py -B build_supermini -D SDKCONFIG=sdkconfig.supermini \
      -D SDKCONFIG_DEFAULTS=sdkconfig.defaults.supermini build
idf.py -B build_supermini -p <PORT> flash
```

Serial smoke (same build dir, baud 460800):

```bash
idf.py -B build_supermini -p <PORT> monitor
```

Then send:

- `STATUS`: the reply should start with `STATUS`.
- `MODE replay_normal`: the reply should be `MODE ok ...`. Send `STATUS` again and confirm the mode switched.

(`MODE replay_abnormal` is the abnormal replay segment — optional, not a required second step. Firmware has no bare `MODE replay`.)

Smoke only proves the firmware answers. It does not prove electrode contact, waveform fidelity, or detection performance.

Flash with `idf.py flash` (app partition). Do not list `erase_flash` or full-chip erase as steps. To keep on-device records, follow the data-preservation notes already in the root README.

## 7. Side path: skip training and flash the existing model

A fresh clone **already contains** on-board v3-A as firmware source, not as training weights:

- **In the tree:** `experiments/esp_idf_ecg_migration/main/model.cc` symbol `models_ecg_model_v3a_int8_tflite`, length **167376**, declared in `model.h` next to it. `main.cc` uses that array.
- **Not in a fresh clone:** `*.h5` and `*.tflite` are gitignored. Do not say “skip training if the h5 is already in the repo.”

Side-path steps: only the ESP-IDF + serial parts of §2, then §6. No Python training deps, no MIT-BIH download. This path flashes the in-tree v3-A.

## 8. Appendix: INCART, PTB, PTB-XL, and the data-root layout

Research corpora use the shared data root (**not** MIT-BIH’s `data/raw/...`). Paths follow `ecg_refs_*` in `scripts/project_paths.json`; code uses:

```python
from ecg_refs import PTBXL_DIR, ECG_DATABASE_DIR, INCART_DIR
```

(from `pc_tools/ecg_dl/ecg_refs.py`). Do not hard-code absolute paths into scripts.

| Corpus | Windows | WSL | Code directory |
|---|---|---|---|
| Root | `C:\ecg_data` | `/mnt/c/ecg_data` | `ecg_refs_root` |
| PTB-XL | `C:\ecg_data\ptbxl` | `/mnt/c/ecg_data/ptbxl` | `PTBXL_DIR` |
| INCART | `C:\ecg_data\incart\files` | `/mnt/c/ecg_data/incart/files` | `INCART_DIR` (implemented as `ecg_refs_incart / 'files'` — do not leave files only under `incart\`) |
| PTB (ptbdb) | `C:\ecg_data\ecg_database` | `/mnt/c/ecg_data/ecg_database` | `ECG_DATABASE_DIR` |

The same root also has `figs/` and `lib/`; neither the happy path nor the appendix scripts use them.

- **INCART** (St. Petersburg 12-lead arrhythmia): longer records than MIT-BIH; used here for another Lead-II 250 Hz beat set. Script `pc_tools/ecg_dl/data/preprocess_incart.py` → `incart_processed.npz` under `PROCESSED_DIR`. If `.dat` files are missing, that script fetches from PhysioNet itself.
- **PTB (ptbdb)**: diagnostic morphology corpus (controls vs patients), not beat-annotated like MIT-BIH. Script `data/preprocess_ptb.py` → `ptb_processed.npz`. This appendix does not document download steps; place files under `ECG_DATABASE_DIR`.
- **PTB-XL**: large 12-lead corpus under `PTBXL_DIR`. One line each:
  - `data/preprocess_ptbxl.py` — beat-level 250 Hz, normal vs diagnostic abnormal → `ptbxl_processed.npz`
  - `data/preprocess_ptbxl_records.py` — record-level 100 Hz, five SCP superclasses → `ptbxl_records_100hz.npz` (not the beat happy path)
  - `data/preprocess_ptbxl_rhythm.py` — beats from rhythm-diagnosis records only → `ptbxl_rhythm_processed.npz`

The appendix does not cite tuning rounds, AUCs, leakage-audit conclusions, or whether these corpora should enter training.

## 9. Pitfalls

- **Slow OneDrive / 9p I/O.** Do not run preprocess or large npz workloads on OneDrive or 9p mounts. `ECG_PROCESSED_DIR` overrides `PROCESSED_DIR` (see `config.py`). If the clone lives on such a disk, point that variable at a local disk before preprocess/train. Do not treat any one developer machine path as required for readers.
- **GPU memory.** ResNet-Large may exhaust VRAM. Finishing “get it running” on CPU is acceptable. Do not report numbers from CPU-only or early-stopped shortcut runs.
- **SuperMini 4MB vs N16R8.** Happy path flashes SuperMini only: `sdkconfig.supermini`, `sdkconfig.defaults.supermini`, `partitions_ecg_supermini.csv`, build dir `build_supermini`. N16R8 (`sdkconfig.n16r8`, `partitions_ecg.csv`, `build_n16r8`) also links extra replay corpora by default and must not be used on SuperMini. Other-board detail: `experiments/esp_idf_ecg_migration/FLASH_DUAL_BOARD.md` — do not copy personal install paths from that file into this guide.
- **Dev-set numbers are not a blind test.** `train.py` prints metrics on the split it just made. The current development set has been reused many times. Do not present those numbers as a held-out test or as cross-device generalization.

## 10. What this path is not

- **Not paper-metric reproduction.** Patient-level primary results still follow `eval_patient_split_all.py` / `eval_patient_split_noaug.py` as pointed from `pc_tools/ecg_dl/README.md`. This guide does not walk those scripts.
- **Not the original on-board v3-A recipe.** The on-board model comes from `train_clean_baseline_v3.py` (weight name `models/best_resnet_large_clean_baseline_v3.h5`, not in a clone); INT8 export script `export_v3a_int8.py`; artifact `models/deploy_match/ecg_model_v3a_int8.tflite` (also not in a clone). Firmware embeds the 167376-byte array in `model.cc`. `train.py --resnet-large --patient-split` is a different entry that writes `best_resnet_large.h5` / `final_resnet_l.h5`.
- **Not clinical validation.** Serial smoke only proves the firmware answers.

## Disclaimer (again)

This repository is a research prototype, not a certified medical device, and not for clinical diagnosis.
```

- [ ] **Step 2: Alignment check**

Diff section headings against the design table EN column — must be exact. Diff every fenced command block against `GETTING_STARTED.md` — must be identical (including `--h5`, `replay_normal`, SuperMini flags).

- [ ] **Step 3: Commit English guide**

```bash
git add GETTING_STARTED.en.md
git commit -m "docs: add English GETTING_STARTED visitor guide"
```

---

### Task 3: One-line links in root READMEs

**Files:**
- Modify: `README.md` — insert before the §2 command fence
- Modify: `README.en.md` — insert before the §2 command fence

**Interfaces:**
- Consumes: `GETTING_STARTED.md` / `GETTING_STARTED.en.md` from Tasks 1–2
- Produces: single prescribed link line each; no other README edits

- [ ] **Step 1: Patch Chinese README**

In `README.md`, find the heading `## 2. 构建与使用` and the following blank line before the opening ` ```bash `. Insert exactly this paragraph between the heading and the fence:

```markdown
把环境、MIT-BIH、一次训练、INT8 导出和 SuperMini 烧录串起来，见 [GETTING_STARTED.md](GETTING_STARTED.md)（[English](GETTING_STARTED.en.md)）。
```

Do not change the product narrative, feature table, verification table, or archive sections.

- [ ] **Step 2: Patch English README**

In `README.en.md`, find `## 2. Build & Use` and insert before its command fence:

```markdown
To walk environment → MIT-BIH → one training run → INT8 export → SuperMini flash, see [GETTING_STARTED.en.md](GETTING_STARTED.en.md) ([中文](GETTING_STARTED.md)).
```

- [ ] **Step 3: Commit README links**

```bash
git add README.md README.en.md
git commit -m "docs: link GETTING_STARTED from root READMEs"
```

---

### Task 4 (optional): One row in `pc_tools/ecg_dl/README.md` 快速定位

**Files:**
- Modify: `pc_tools/ecg_dl/README.md` — 快速定位 table only

**Interfaces:**
- Consumes: root `GETTING_STARTED.md`
- Produces: one new table row; file remains a script index

- [ ] **Step 1: Add the row**

In the `## 快速定位（按任务）` table, add one row (place near the top, after the header separator):

```markdown
| 访客最小可跑通（环境→训练→导出→烧录） | 仓库根 [GETTING_STARTED.md](../../GETTING_STARTED.md)（本文仍是脚本索引） |
```

No other edits to this file.

- [ ] **Step 2: Commit or skip**

If including this optional task:

```bash
git add pc_tools/ecg_dl/README.md
git commit -m "docs(ecg_dl): point 快速定位 at root GETTING_STARTED"
```

If skipping, note “Task 4 skipped” in the PR body and leave the file untouched.

---

### Task 5: Open docs PR and final spec coverage review

**Files:**
- All files from Tasks 1–4 (and confirm Task 0 PR is separate / already merged if export was broken)
- Do **not** include `docs/superpowers/specs/...` changes unless rebasing onto PR #6

- [ ] **Step 1: Branch and PR**

Prefer branching from `main` after PR #6 (design) is merged, or include the design file if still open and the user asked to stack. Suggested:

```bash
git checkout -b docs/getting-started-guides
# ensure Tasks 1–4 commits are on this branch
gh pr create --title "docs: GETTING_STARTED visitor guides" --body "$(cat <<'EOF'
Implements the approved design in docs/superpowers/specs/2026-10-26-getting-started-design.md.

- Add GETTING_STARTED.md / GETTING_STARTED.en.md (sections 0–10 aligned)
- One-line links in README.md / README.en.md
- Optional ecg_dl 快速定位 row (if included)

Depends on / related: export.py compile=False fix PR (Task 0), design PR #6.
EOF
)"
```

- [ ] **Step 2: Final checklist (self-review)**

| Spec requirement | Task that covers it |
|---|---|
| CN + EN guides, sections 0–10 aligned | Tasks 1–2 |
| Root README one-line links (exact copy) | Task 3 |
| Optional ecg_dl index row | Task 4 |
| Pinned commands (preprocess path, train flags, `--h5`, `replay_normal`) | Tasks 1–2 |
| Export ≠ replace `model.cc` | Tasks 1–2 bridge subsection |
| Side path / 167376 / gitignored weights | Tasks 1–2 §7 |
| Appendix INCART/PTB/PTB-XL + `ecg_refs` | Tasks 1–2 §8 |
| Pitfalls + “what this is not” | Tasks 1–2 §9–10 |
| Disclaimer start+end | Tasks 1–2 |
| FocalLoss export load | Task 0 (separate PR) |
| No Flutter/plotter/binaries/`model.cc` rewrite | Global constraints |

- [ ] **Step 3: Done criteria for the reader (document in PR)**

- Full path: local `mit_bih_processed.npz`, `best_resnet_large.h5`, `final_resnet_l.h5`, `ecg_model.tflite`, `ecg_model_data.h`, SuperMini flash OK, `STATUS` + `MODE replay_normal` replies as specified.
- Side path: flash + same serial replies without data/train.
- CN/EN section numbers match; disclaimer at both ends.

---

## Self-Review (plan author)

1. **Spec coverage:** All design “实现说明” bullets map to Tasks 0–4; pinned command section and export↔flash bridge are explicit in Tasks 1–2; FocalLoss caveat is Task 0, not buried in the docs PR.
2. **Placeholders:** None — section bodies and exact README lines are inlined.
3. **Consistency:** Train/export/flash command strings are identical across CN and EN tasks; symbol name `models_ecg_model_v3a_int8_tflite` / length 167376 match the design.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-10-26-getting-started.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
