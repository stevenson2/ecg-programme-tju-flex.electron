# Data Availability Statement

> Per MDPI *Sensors* policy, all manuscripts must include a Data Availability Statement.  
> **MDPI template category:** "Data are contained within the article and supplementary materials. The original contributions presented in the study are included in the article/Supplementary Material; further inquiries can be directed to the corresponding authors."

---

## Data Sources — Public Datasets

The following public datasets were used in this study. All are distributed through PhysioNet (Goldberger et al., *Circulation*, 2000; DOI: 10.1161/01.CIR.101.23.e215) and are freely available for research purposes:

| Dataset | Access | Used For |
|---------|--------|----------|
| **MIT-BIH Arrhythmia Database** | https://physionet.org/content/mitdb/ | Arrhythmia beat classification (48 records, 47 patients) |
| **St. Petersburg INCART 12-lead Arrhythmia Database** | https://physionet.org/content/incartdb/ | Arrhythmia beat classification (75 records, 32 patients) |
| **PTB Diagnostic ECG Database** | https://physionet.org/content/ptbdb/ | Myocardial infarction detection (533 records, 290 patients) |

## Code and Processed Data

The full project codebase — including firmware source code (`src/`), Python training and evaluation tools (`pc_tools/ecg_dl/`), electrode characterization protocols (`docs/hardware/`), PC plotting tools (`pc_tools/`), and the Flutter smartphone application (`ecg_app/`) — is available from the project repository:

> **Repository:** [URL to be inserted — the project is hosted in a private repository; a public release or Zenodo archive will be created upon acceptance.]

## Model Weights and Large Files

Trained model weights (`.h5`, `.tflite`, `.cc` C array exports) and evaluation result files (`.json`, `.csv`) are **not committed to the public repository** due to GitHub file-size limits. These files are available from the corresponding author upon reasonable request:

| File | Description | Size |
|------|-------------|------|
| `best_resnet_large_exp6_patient_clean.h5` | Best patient-level clean model (ResNet-L, ~80K params) | ~2.7 MB |
| `best_resnet_large_exp6_sgd.h5` | Deployment-chain retrained model (SGD) | ~2.7 MB |
| `final_resnet_l_p2a_backup.h5` | Best cross-domain model (P2A) | ~2.7 MB |
| `patient_split_eval.json` | Full evaluation results (14 models, all metrics) | ~100 KB |
| `retrain_exp6_sgd_eval.json` | Deployment-chain evaluation results | ~5 KB |
| `*.tflite` | INT8-quantized TFLite models for on-device deployment | ~25 KB each |
| `train_history_*.csv` | Training history logs | ~50 KB each |

## Figures and Analysis Scripts

All figures in the manuscript are generated from data in `docs/FINAL_RESULTS.md` and the evaluation JSON files listed above. Plotting and analysis scripts reside in `pc_tools/ecg_dl/` and are included in the repository.

## Human-Subject Data

No human-subject data have been collected at the time of writing (ethics approval pending). Upon completion of the study, anonymized ECG waveform recordings will be made available as supplementary material accompanying the publication, in accordance with the informed consent terms (subject codes S01–S10; only sex and age recorded; raw waveforms do not contain personally identifiable information).

---

**Notes for pre-submission:**
1. Insert the actual repository URL (GitHub public release or Zenodo DOI) once decided.
2. Confirm whether model weights should be uploaded to Zenodo with a DOI or offered "on request."
3. If MDPI *Sensors* requires a specific data availability template wording, adjust accordingly.
