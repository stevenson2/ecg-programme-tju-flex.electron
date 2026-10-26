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
