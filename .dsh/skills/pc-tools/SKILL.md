---
name: pc-tools
description: Python PC tools for ECG serial capture, plotting and signal debugging on the ESP32-ECG project (PowerShell + WSL2).
---

# PC 工具与串口脚本

## 串口脚本（已迁移到 pc_tools/serial/）
- 解释器：`C:\Users\cai\.platformio\penv\Scripts\python.exe`（含 pyserial）
- 设备：COM4，波特率 460800
- `serial_reset.py`：DTR/RTS 复位（设备偶发卡死、串口 0 输出时使用）
- 常用诊断：`serial_ecg_diag2.py`、`serial_ai_dist.py`（AI 置信度分布）、`serial_baseline.py`

## 硬性约束
- **用户手机验证 BLE 期间禁止运行任何串口脚本**（实测会干扰 BLE 连接）。
- 需要真机数据时先征得用户同意，或等用户不操作设备的空档执行。

## 训练/评估脚本（pc_tools/ecg_dl/，WSL2 运行）
- 统一入口：`wsl -e bash -lc "cd /mnt/c/.../pc_tools/ecg_dl && python3 <脚本>"`
- 训练 timeout ≥ 900000ms，评估 ≥ 600000ms；WSL 已装 python3+tensorflow 2.21+scipy+pywt
- 修改 Python 文件后用 `python -m py_compile` 做语法检查。

## 串口/BLE CSV 格式
`<clean>,<noisy>,<filtered>,<bpm>,<true_bpm>,<sqi>,<motion>,<abnormal_flag>,<confidence>`
