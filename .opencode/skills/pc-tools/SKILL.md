---
name: pc-tools
description: Use Python PC tools for real-time ECG plotting, data capture, and signal debugging
license: MIT
compatibility: opencode
metadata:
  language: python
  tools: ecg_plotter, capture_debug
---

## What I do
- Real-time ECG plotting: `python pc_tools/ecg_plotter.py --port COMx`
- Capture debug data: `python pc_tools/capture_debug.py --port COMx --output capture.csv`
- Find ESP32 port: `python pc_tools/find_port.py`
- Verify filter coefficients: `python pc_tools/verify_filter_coeffs.py`
- Simulate heart rate: `python pc_tools/hr_sim_verify.py`

## When to use me
Use when you need to visualize ECG signals, capture data for analysis, or debug signal processing on the PC side.
