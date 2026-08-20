# Supplementary Material Plan

> This document catalogs the supplementary materials planned to accompany the manuscript submission to *Sensors* (MDPI).  
> **Status:** Inventory and planning. Most items are in preparation or pending data collection.  
> Files will be organized in `docs/submission_package/supplementary/` at submission time.

---

## S1. Patient Split Specifications

| File | Content | Status |
|------|---------|--------|
| `S1_patient_splits.json` | Train/validation/test patient ID lists for MIT-BIH (47 patients), INCART (32 patients), merged MIT+INCART (79 patients), and PTB (290 patients), under seed=42 60/20/20 permutation semantics. Includes the explicit train∩test intersection verification (empty for all splits). | **Available** — to extract from `pc_tools/ecg_dl/` split logs |
| `S1_split_metadata.csv` | Per-record metadata: dataset, patient ID, partition (train/val/test), number of beats, augmentation factor | **To generate** |

## S2. AAMI Class-Wise Breakdown

| File | Content | Status |
|------|---------|--------|
| `S2_aami_per_class.json` | Per-AAMI-superclass metrics (N, SVEB, VEB, F, Q) for each reported model (exp4, exp5, exp6 patient-level clean, P2A), including per-class recall, precision, F1, and confusion matrix counts at θ=0.5. | **Available** — to extract from `patient_split_eval.json` per-class fields |
| `S2_fusion_beat_analysis.csv` | Detailed breakdown of the fusion beat (F) recall ceiling (0.442) with per-record counts and misclassification patterns | **To generate** |

## S3. Deployment-Chain Mismatch Analysis

| File | Content | Status |
|------|---------|--------|
| `S3_deploy_chain_comparison.json` | Side-by-side metrics for the same model evaluated through training chain (D0, filtfilt) vs. deployment chain (D3, causal biquad + comb + 2:1 decimation), with ΔAUC, ΔRecall, ΔPrecision per domain. Source: `deploy_match/retrain_exp6_sgd_eval.json` vs. `patient_split_eval.json` exp6(患者级清洁). | **Available** — from existing JSONs |
| `S3_filter_transfer_functions.pdf` | Bode plots comparing the frequency responses of the training chain (filtfilt zero-phase) and deployment chain (causal biquad + comb) | **To generate** (MATLAB/Python) |

## S4. Electrode Characterization Data

| File | Content | Status |
|------|---------|--------|
| `S4_electrode_char.csv` | Measured values from the T1–T5 electrode characterization protocol (conductivity, tensile, peel adhesion, skin-contact impedance, ECG RMS noise) with Ag/AgCl comparator. Source: `docs/hardware/electrode_char_protocol.md` and `docs/hardware/electrode_char_checklist.md`. | **Pending** — protocol exists; measurements pending execution |
| `S4_electrode_fabrication.pdf` | Photographic documentation of the electrode fabrication process (solution blending, casting, drying, cutting) and final films | **To capture** |

## S5. Human-Subject Protocol Documents

| File | Content | Status |
|------|---------|--------|
| `S5_human_subject_protocol.md` | Full protocol document including action segment timing, electrode placement diagram, exclusion criteria, stop rules, and analysis plan. Source: `docs/hardware/human_subject_protocol.md`. | **Available** — protocol written |
| `S5_consent_form_template.pdf` | Blank informed consent form template (Chinese, from `docs/hardware/consent_form_zh.md`) — redacted of advisor/student contact placeholders if not yet filled. | **Available** — template written; fill contact info before use |
| `S5_data_collection_sheet.csv` | Per-subject data collection log: subject code, date, electrode order, segment timestamps, any deviations or adverse events | **To create** |

## S6. On-Device Benchmarking Results

| File | Content | Status |
|------|---------|--------|
| `S6_ondevice_bench.json` | Measured latency (mean, P95, P99), power (idle/AI/BLE states), INT8 consistency (|Δp|, decision agreement, |ΔAUC|), and thermal profile over 10 min continuous inference. Source: `docs/hardware/ondevice_bench_protocol.md`. | **Pending** — protocol exists; measurements pending execution |

## S7. System Architecture and Firmware

| File | Content | Status |
|------|---------|--------|
| `S7_firmware_signal_chain.pdf` | Block diagram of the firmware signal chain: comb filter → biquad HP/LP/notch → 2:1 decimation → ring buffer → TFLite Micro inference, with annotated data rates and buffer sizes | **To generate** (from manuscript Figure F4 source) |
| `S7_ble_frame_format.md` | Specification of the 9-column CSV frame format, BLE NUS UUIDs, notify batching behavior, and serial command interface | **Available** — from README.md and `src/bluetooth/` |

## S8. Training Configuration and Hyperparameters

| File | Content | Status |
|------|---------|--------|
| `S8_training_configs.json` | Full training configurations for all reported models: architecture, optimizer, learning rate schedule, loss function (focal loss γ=1.0, α=0.75), augmentation parameters, early stopping, batch size, epochs | **Available** — from `TUNING_HISTORY.md` and training scripts |
| `S8_training_curves.pdf` | Training and validation loss/AUC curves for exp4, exp5, exp6 patient-clean runs, generated from `train_history_*.csv` | **To generate** (matplotlib) |

---

## Assembly Checklist

- [ ] Create `docs/submission_package/supplementary/` folder
- [ ] Copy or generate each S1–S8 file
- [ ] Verify all JSON/CSV files are valid and internally consistent with `docs/FINAL_RESULTS.md`
- [ ] For pending items (S4, S5, S6): either complete data collection before submission, or note in the cover letter that supplementary characterization data will be provided upon completion
- [ ] Compress supplementary folder as a single `.zip` archive for MDPI submission system upload (if required)
- [ ] Reference supplementary materials in the manuscript where appropriate (e.g., "see Supplementary Figure S2.1")

**Note:** MDPI *Sensors* accepts supplementary files in PDF, ZIP, and common image/table formats. Check the current author guidelines for file type and size limits.
