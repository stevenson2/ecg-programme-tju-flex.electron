#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""setup_esp_dual_board.py -- prepare dual-board ESP-IDF build configs.

Creates:
  - sdkconfig.n16r8       (current 16MB / octal PSRAM config)
  - sdkconfig.supermini   (4MB / quad PSRAM config)
  - partitions_ecg_supermini.csv
  - sdkconfig.defaults.supermini
in both the repo experiments project and the C:/esp build copy.
Also writes flash helper batch files for both boards.
"""
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2] / "experiments" / "esp_idf_ecg_migration"
ESP = Path("/mnt/c/esp/esp_idf_ecg_migration")
DIRS = [REPO, ESP] if ESP.exists() else [REPO]

SUPERMINI_PARTITIONS = """# ESP-IDF Partition Table - ESP32-S3 SuperMini (ESP32S3FH4R2)
# 4MB Quad Flash / 2MB Quad PSRAM
# Name,   Type, SubType, Offset,  Size, Flags
nvs,      data, nvs,     0x9000,   24K,
phy_init, data, phy,     0xf000,   4K,
factory,  app,  factory, 0x10000,  2816K,
storage,  data, spiffs,  0x2d0000,  960K,
"""

SUPERMINI_DEFAULTS = """CONFIG_FREERTOS_HZ=1000
CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y
CONFIG_ESPTOOLPY_FLASHSIZE_4MB=y
CONFIG_ESPTOOLPY_FLASHSIZE="4MB"
CONFIG_ESP32S3_SPIRAM_SUPPORT=y
CONFIG_SPIRAM_MODE_QUAD=y
CONFIG_SPIRAM_SPEED_80M=y
CONFIG_SPIRAM_USE_MALLOC=y

CONFIG_PARTITION_TABLE_CUSTOM=y
CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="partitions_ecg_supermini.csv"

CONFIG_BT_ENABLED=y
CONFIG_BT_NIMBLE_ENABLED=y
CONFIG_BT_BLUEDROID_ENABLED=n
CONFIG_BT_NIMBLE_MAX_CONNECTIONS=1
"""

FLASH_N16R8 = """@echo off
call %~dp0..\\..\\v6.0.1\\esp-idf\\export.bat
set IDF_PYTHON_ENV_PATH=C:\\Users\\cai\\.espressif\\python_env\\idf6.0_py3.13_env
set IDF_TOOLS_PATH=C:\\Users\\cai\\.espressif
set IDF_PATH=C:\\esp\\v6.0.1\\esp-idf
set ESP_IDF_VERSION=6.0
cd /d C:\\esp\\esp_idf_ecg_migration
idf.py -B build_n16r8 -D SDKCONFIG=C:/esp/esp_idf_ecg_migration/sdkconfig.n16r8 build flash
"""

FLASH_SUPERMINI = """@echo off
call %~dp0..\\..\\v6.0.1\\esp-idf\\export.bat
set IDF_PYTHON_ENV_PATH=C:\\Users\\cai\\.espressif\\python_env\\idf6.0_py3.13_env
set IDF_TOOLS_PATH=C:\\Users\\cai\\.espressif
set IDF_PATH=C:\\esp\\v6.0.1\\esp-idf
set ESP_IDF_VERSION=6.0
cd /d C:\\esp\\esp_idf_ecg_migration
idf.py -B build_supermini -D SDKCONFIG=C:/esp/esp_idf_ecg_migration/sdkconfig.supermini build flash
"""


def patch_supermini(sdk_txt: str) -> str:
    sdk_txt = sdk_txt.replace("CONFIG_ESPTOOLPY_FLASHSIZE_16MB=y",
                              "# CONFIG_ESPTOOLPY_FLASHSIZE_16MB is not set")
    sdk_txt = sdk_txt.replace("# CONFIG_ESPTOOLPY_FLASHSIZE_4MB is not set",
                              "CONFIG_ESPTOOLPY_FLASHSIZE_4MB=y")
    sdk_txt = sdk_txt.replace('CONFIG_ESPTOOLPY_FLASHSIZE="16MB"',
                              'CONFIG_ESPTOOLPY_FLASHSIZE="4MB"')
    sdk_txt = sdk_txt.replace("CONFIG_SPIRAM_MODE_OCT=y",
                              "# CONFIG_SPIRAM_MODE_OCT is not set")
    sdk_txt = sdk_txt.replace("# CONFIG_SPIRAM_MODE_QUAD is not set",
                              "CONFIG_SPIRAM_MODE_QUAD=y")
    sdk_txt = sdk_txt.replace('CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="partitions_ecg.csv"',
                              'CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="partitions_ecg_supermini.csv"')
    return sdk_txt


def main():
    for d in DIRS:
        if not isinstance(d, Path):
            continue
        d.mkdir(parents=True, exist_ok=True)
        sdk_current = d / "sdkconfig"
        sdk_n16 = d / "sdkconfig.n16r8"
        if sdk_current.exists() and not sdk_n16.exists():
            shutil.copy2(sdk_current, sdk_n16)
        sdk_super = d / "sdkconfig.supermini"
        if sdk_n16.exists():
            txt = sdk_n16.read_text(encoding="utf-8", errors="replace")
            sdk_super.write_text(patch_supermini(txt), encoding="utf-8")
        (d / "partitions_ecg_supermini.csv").write_text(SUPERMINI_PARTITIONS, encoding="utf-8")
        (d / "sdkconfig.defaults.supermini").write_text(SUPERMINI_DEFAULTS, encoding="utf-8")
        print(f"[DUAL] prepared {d}", flush=True)
    # Flash helpers in C:/esp only.
    if ESP.exists():
        # Write helper scripts next to the project (C:/esp).
        (ESP / "flash_n16r8.bat").write_text(FLASH_N16R8, encoding="utf-8")
        (ESP / "flash_supermini.bat").write_text(FLASH_SUPERMINI, encoding="utf-8")
        print("[DUAL] wrote flash_n16r8.bat / flash_supermini.bat in C:/esp/esp_idf_ecg_migration", flush=True)


if __name__ == "__main__":
    main()
