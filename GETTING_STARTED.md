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
