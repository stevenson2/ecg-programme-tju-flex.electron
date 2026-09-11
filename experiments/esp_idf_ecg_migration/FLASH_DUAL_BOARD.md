# ESP-IDF 双板烧录说明（v3-A 模型）

本说明用于通过 ESP-IDF 工具链把 **v3-A clean baseline INT8 模型** 烧录到：

- **N16R8 模组**：ESP32-S3-WROOM-1-N16R8，16MB Flash / 8MB Octal PSRAM
- **SuperMini 开发板**：ESP32S3FH4R2，4MB Flash / 2MB Quad PSRAM

## 1. 模型文件

已生成：

- `C:\esp\esp_idf_ecg_migration\main\model.cc`
- `C:\esp\esp_idf_ecg_migration\main\model.h`
- `main.cc` 已引用 `models_ecg_model_v3a_int8_tflite`

模型源 tflite：

- `pc_tools/ecg_dl/models/deploy_match/ecg_model_v3a_int8.tflite`

## 2. 板级配置

| 配置 | N16R8 | SuperMini |
|---|---|---|
| sdkconfig | `sdkconfig.n16r8` | `sdkconfig.supermini` |
| flash | 16MB | 4MB |
| PSRAM | Octal 80M | Quad 80M |
| 分区表 | `partitions_ecg.csv` | `partitions_ecg_supermini.csv` |
| 构建目录 | `build_n16r8` | `build_supermini` |

## 3. 烧录命令

### N16R8

```bat
call C:\esp\v6.0.1\esp-idf\export.bat
set IDF_PYTHON_ENV_PATH=C:\Users\cai\.espressif\python_env\idf6.0_py3.13_env
set IDF_TOOLS_PATH=C:\Users\cai\.espressif
set IDF_PATH=C:\esp\v6.0.1\esp-idf
set ESP_IDF_VERSION=6.0

cd /d C:\esp\esp_idf_ecg_migration
idf.py -B build_n16r8 -D SDKCONFIG=C:/esp/esp_idf_ecg_migration/sdkconfig.n16r8 build flash -p COM17
```

### SuperMini

```bat
call C:\esp\v6.0.1\esp-idf\export.bat
set IDF_PYTHON_ENV_PATH=C:\Users\cai\.espressif\python_env\idf6.0_py3.13_env
set IDF_TOOLS_PATH=C:\Users\cai\.espressif
set IDF_PATH=C:\esp\v6.0.1\esp-idf
set ESP_IDF_VERSION=6.0

cd /d C:\esp\esp_idf_ecg_migration
idf.py -B build_supermini -D SDKCONFIG=C:/esp/esp_idf_ecg_migration/sdkconfig.supermini build flash -p COMx
```

`COMx` 换成实际串口。

## 4. 快捷脚本

- `C:\esp\esp_idf_ecg_migration\flash_n16r8.bat`
- `C:\esp\esp_idf_ecg_migration\flash_supermini.bat`

## 5. 当前状态

- 已用 `build_n16r8` + v3-A 模型烧录到 COM17 并验证写入成功。
- 已用 `build_supermini` + v3-A 模型烧录到 COM15 并验证写入成功。
- SuperMini 构建信息：
  - app 大小 `0x185160` bytes，分区 `0x2c0000`，剩余 45%。

## 6. 注意

- 串口波特率 **460800**（`sdkconfig.n16r8` / `sdkconfig.supermini` / `sdkconfig.defaults` 已对齐；`pc_tools` 与 Arduino 线同速）。`idf.py monitor` 与 Python 脚本不要再用 115200。
- 烧录前用 `esptool --port COMx chip_id` / `flash_id` 确认板子与端口；COM 号会变，不要死记 COM17/COM15。
- 如果 `export.bat` 报 py3.12 环境缺失，请用本说明中的 `IDF_PYTHON_ENV_PATH` 指向 `idf6.0_py3.13_env`，或直接使用 `C:\Users\cai\.espressif\python_env\idf6.0_py3.13_env\Scripts\python.exe C:\esp\v6.0.1\esp-idf\tools\idf.py`。
- 编译时建议用 `cmake --build <build_dir> -- -j2` 避免 GCC 并行编译偶发段错误。
