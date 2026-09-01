# ESP32-ECG — Portable ECG Acquisition & On-device AI Anomaly Detection

> **ESP32-S3 · ESP-IDF · TFLite Micro + ESP-NN · exp7c INT8 · BLE NUS · Flutter**  
> English | [中文](README.md)

**Portable single-lead (Lead II) ECG acquisition with on-device deep-learning beat-level anomaly detection.** 500 Hz sampling; filtering, heart-rate detection and anomaly inference run on the chip; alarms are pushed to a Flutter app over BLE, and recordings are stored on-board in the ECGR format with WiFi download support.

> **Firmware status (2026-09-01)**: the **official firmware is the ESP-IDF migration project** `experiments/esp_idf_ecg_migration/` (promoted 2026-08-28; AI/storage/WiFi/BLE/heart-rate/rule components plus the recorder chain). The legacy **Arduino + PlatformIO** line is archived under `legacy_arduino/` for reference only.

---

## Features

| Area | Details |
|------|---------|
| Acquisition | 500 Hz, 3-channel (clean / noisy / filtered); sources: simulator, real AFE, MIT-BIH replay |
| Filtering | Two-stage comb (50/100 Hz nulling, -119.2 dB) → HP → LP 40 Hz |
| Heart rate | Energy-envelope QRS detector v6 (LUDB: F1 0.868, Se 96.4%, BPM MAE 4.16) |
| AI | exp7c ResNet-L INT8 (167,376 B); TFLite Micro + ESP-NN inference; beat-level anomaly detection |
| Rhythm | Asystole / bradycardia / tachycardia (rules), AF (CV+entropy), VF/VT (DSP features + LR) |
| Alarm | 5 s latch; `abnormal` flag + per-second bitmap over BLE/serial |
| Recording | ECGR format (32 B header + int16 stream + 1 B/s abnormal bitmap); auto-record on anomaly; WiFi REST download |
| Communication | BLE NUS (TX/RX) + WiFi HTTP; Flutter app on mobile |

---

## Quick Start

```bash
# —— Official firmware (ESP-IDF v6) ——
cd experiments/esp_idf_ecg_migration
idf.py build              # build
idf.py -p <PORT> flash    # flash
idf.py monitor            # serial monitor

# —— Legacy Arduino line (reference only) ——
cd legacy_arduino && pio run

# —— PC plotting / mobile app ——
python pc_tools/ecg_plotter.py
cd ecg_app && flutter run
```

> ⚠️ AI model training requires a GPU (WSL2).

---

## Architecture

```mermaid
flowchart LR
    A[Electrodes RA/LA/RL<br/>Lead II] --> B[AFE<br/>AD8232 / custom]
    B --> C[ADC<br/>500 Hz]
    C --> D[Comb 50/100 Hz<br/>-119.2 dB]
    D --> E[HP + LP 40 Hz]
    E --> F[HR v6 + Rhythm/AF/VF]
    E --> G[2:1 decimate<br/>250 Hz]
    G --> H[250-pt window]
    H --> I[Z-score + INT8]
    I --> J[TFLite Micro + ESP-NN<br/>exp7c INT8]
    J --> K[Anomaly probability]
    K --> L[Alarm latch 5 s]
    E --> M[ECGR record + bitmap]
    F --> L
    L --> N[BLE NUS TX + serial]
    M --> O[WiFi REST download]
    L --> P[Flutter App]
    O --> P
```

- **Core assignment**: Core 1 = acquisition/filtering/comms/storage; Core 0 = AI inference (250-pt window, AI_STRIDE=250).
- **Pipeline**: 500 Hz → causal filtering → 2:1 decimation → 250-pt window → normalization → INT8 → inference → dequantized anomaly probability.

---

## Directory Layout

```
experiments/esp_idf_ecg_migration/  # official ESP-IDF firmware line
  main/                            # main (sampling/filter/HR/alarm/record/BLE/WiFi)
  components/                      # ecg_ai · ecg_core · ecg_ble · ecg_wifi · ecg_storage · ecg_afe
                                   # + esp-tflite-micro · esp-nn (vendored)
legacy_arduino/                    # legacy Arduino/PlatformIO line (archived)
pc_tools/                          # PC tools: ecg_plotter + ecg_dl (train/eval/export)
ecg_app/                           # Flutter mobile app
web/                               # web recording-download frontend
docs/                              # papers & results (authoritative numbers: docs/FINAL_RESULTS.md)
test/                              # tests
papers/                            # literature
```

---

## AI Model & Metrics

**On-board model**: exp7c (ResNet-L, ~80K params), INT8 **167,376 B**, on-device since 2026-08-14. Paper operating point: beat θ≈0.35 / patient θ≈0.5; **firmware runs θ=0.60 + 5-beat confirmation**.

| Cadence | Model | MIT-AUC | MIT-R@0.5 | PTB-AUC | PTB-R@0.5 |
|------|------|:---:|:---:|:---:|:---:|
| Patient-level clean / un-augmented | exp5 | 0.9295 | 0.9264 | 0.7845 | 0.6281 |
| Patient-level clean / un-augmented | exp6 | 0.8942 | 0.9194 | **0.8232** | 0.7019 |
| Cross-domain (no PTB training) | P2A | **0.9878** | 0.9312 | 0.7502 | 0.2552 |
| Deploy chain (D3, δ-aligned) | exp6-SGD | 0.9122 | 0.9102 | 0.7697 | 0.7069 |

> The two cadences (patient-level clean/un-augmented vs deploy chain D3) are not directly comparable.

---

## Toolchain

- **PC plotting** `pc_tools/ecg_plotter.py`: real-time 3-channel waveform + AI anomaly labels (packaged as ECG-Plotter.exe).
- **Deep learning** `pc_tools/ecg_dl/`: training / INT8 export / evaluation (patient-level leakage-free split + SplitGuard).
- **Mobile app** `ecg_app/`: BLE NUS waveform display, AI-highlighted alarms, recording list/playback.

---

## Docs

User-facing details are in the per-module `README.md` files (`pc_tools/ecg_dl/`, `ecg_app/`, `web/`, `experiments/`).

---

## License

[MIT](LICENSE)
