---
name: platformio-build
description: Build the ESP32-S3 (N16R8) ECG firmware with PlatformIO for compile-check. Never uploads firmware — hardware operations belong to the user.
---

# ESP32-ECG 固件编译检查

## 硬性约束（AGENTS.md 铁律）
- **只编译检查，不烧录**：禁止 `pio run -t upload`；烧录由用户执行。
- 修改任何 C++ 文件后必须运行 `pio run`，编译失败必须修复后才能标记任务完成。

## 命令（PowerShell，项目根目录运行）
- 编译检查：`C:\Users\cai\.platformio\penv\Scripts\pio.exe run`
- 环境：`esp32-s3-n16r8`（board=4d_systems_esp32s3_gen4_r8n16，platformio.ini 已配）
- 干净构建：`pio run -t clean && pio run`

## 已知要点
- 输出目录 `.pio/build/esp32-s3-n16r8/`；增量编译只重编改动的翻译单元。
- AI 模型头文件：`include/ai_inference/ecg_model_data.h`（exp7b INT8, 163.5KB）。
- 修改 filter.cpp/filter.h 后注意显示链与 AI 链已解耦（HP 0.5 vs HP 0.05+LP40），勿混。
- 串口监视/命令属硬件操作阶段，本阶段不执行。
