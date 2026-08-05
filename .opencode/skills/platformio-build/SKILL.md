---
name: platformio-build
description: Build, upload, and monitor ESP32-S3 firmware using PlatformIO CLI
license: MIT
compatibility: opencode
metadata:
  framework: platformio
  board: esp32-s3-supermini
---

## What I do
- Build firmware: `pio run`
- Upload to board: `pio run -t upload`
- Monitor serial output: `pio device monitor -b 115200`
- Clean build: `pio run -t clean`
- Build with environment: `pio run -e esp32-s3-supermini`

## When to use me
Use when you need to compile, flash, or debug the ESP32 firmware. I handle PlatformIO build system commands and serial monitoring.
