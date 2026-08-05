---
name: ai-training
description: Train, evaluate, and export TFLite Micro INT8 1D-CNN models for ECG anomaly detection
license: MIT
compatibility: opencode
metadata:
  framework: tensorflow
  model: 1d-cnn
---

## What I do
- Download MIT-BIH data: `python pc_tools/ecg_dl/download_mitdb.py`
- Preprocess and train: `python pc_tools/ecg_dl/train.py`
- Evaluate model: `python pc_tools/ecg_dl/evaluate.py`
- Export TFLite INT8: `python pc_tools/ecg_dl/export_tflite.py`
- Convert to C header: `xxd -i model.tflite > model_data.h`

## When to use me
Use when you need to retrain the AI model, improve accuracy, or export a new TFLite model for deployment on ESP32.
