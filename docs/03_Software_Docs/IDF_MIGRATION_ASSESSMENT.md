# ESP-IDF 固件迁移评估与当前进度（2026-08-24）

## 1. 路线评估

### 路线 A：整个固件迁移到 ESP-IDF（推荐）
- 优点：esp-tflite-micro / ESP-NN 本身就是 ESP-IDF component，无需混合框架；
  ITCM/PSRAM、FreeRTOS、esp_timer、esp_wifi、NimBLE 和 SPIFFS/LittleFS 均为原生 API。
- 缺点：原 Arduino 固件的 BLE（NimBLE-Arduino）、WiFi（WiFi.h/WebServer.h）、
  SPIFFS、ADC 和 `Serial/String` 需要重写为 ESP-IDF API，工作量明显。
- 结论：作为长期方向，**路线 A 正确**，也是本次已开始落地的方式。

### 路线 B：Arduino 工程中封装 esp-tflite-micro + ESP-NN
- 优点：可保留现有 BLE/WiFi/SPIFFS/主循环，仅替换 TFLite 推理后端。
- 缺点：esp-tflite-micro 是 IDF component，依赖 IDF 的 CMake/组件机制；
  在 PlatformIO+Arduino 中直接编译其全部源码需要手工维护 include 路径和
  `TF_LITE_STATIC_MEMORY` 等宏，且与现有 `tanakamasayuki/TensorFlowLite_ESP32`
  可能符号冲突。可行性未做最小验证，风险高。
- 结论：不推荐作为主线；只有时间极紧且 AI 加速是唯一目标时才值得做最小验证。

**决策：选择路线 A，继续完善 `experiments/esp_idf_ecg_migration`。**

## 2. 已完成（可编译/可运行）

| 模块 | 状态 |
|---|---|
| `components/ecg_ai` | ✅ ESP-IDF AI 推理组件：INT8 exp7c、PSRAM arena、2:1 抽取、因果 0.5Hz HP、Z-score、INT8 填充、反量化、1-of-N/K-of-N/冷却、结果队列/回调 |
| `components/ecg_storage` | ✅ SPIFFS 录制组件（POSIX/esp_spiffs），ECGR 格式保持，PSRAM 缓冲 |
| `components/ecg_wifi` | ✅ ESP-IDF SoftAP + HTTP server：记录列表/下载/删除 |
| `components/ecg_ble` | ✅ ESP-IDF NimBLE NUS：TX Notify / RX Write，设备名 ESP32-ECG |
| `components/ecg_core` | ✅ 移植 filter / heartrate / rhythm_safety / af_detect / vf_detect / ecg_simulator / ecg_replay，保留原算法参数 |
| `experiments/esp_idf_ecg_migration` | ✅ `idf.py build` 通过（ESP32-S3，`-j2` ninja 构建成功） |
| PC-ESP32 输出一致性 | ✅ 200 拍固定集：mean\|Δp\|=0.000625，max=0.0273，\|ΔAUC\|=0.00015 |

## 3. 尚未完成（路线 A 后续）

1. **BLE**：将 `src/bluetooth/ble.cpp` 从 NimBLE-Arduino 迁移到 ESP-IDF 原生 NimBLE 或 Bluedroid。
2. **WiFi AP + HTTP 下载**：将 `src/wifi/ecg_wifi.cpp` 从 `WiFi.h/WebServer.h` 迁移到
   `esp_wifi` + `esp_http_server`。
3. **SPIFFS 录制**：将 `src/storage/ecg_recorder.cpp` 从 Arduino SPIFFS 迁移到
   `esp_spiffs`/`esp_littlefs`，或保留 FATFS。
4. **ADC/AFE 与 LOD**：将 `src/adc_afe/afe_hal.cpp` 从 `analogRead` 迁移到
   `esp_adc/adc_oneshot` + GPIO。
5. **串口命令**：将 Arduino `Serial` 命令处理迁移到 `esp_console` 或 UART driver。
6. **回放/报警联调**：在迁移后的 IDF 主循环中接上完整采样、心率、AF/VF、心律安全、
   存储、BLE、WiFi 逻辑，并完成非真实 AFE 回放验收。

## 4. 证据产物

- 一致性 PC 输出：`experiments/esp_tflm_bench/consistency_pc.json`
- 板上日志：`C:\esp\esp_tflm_bench\monitor_component2.log`
- 一致性结果：`experiments/esp_tflm_bench/consistency_result.json`
- AI 组件：`components/ecg_ai/`（同时复制于 `experiments/esp_tflm_bench/components/ecg_ai`）
- IDF 核心工程：`experiments/esp_idf_ecg_migration/`
- 快速构建（短路径）：`C:\esp\esp_idf_ecg_migration`（需把 EXTRA_COMPONENT_DIRS 指向仓库 components）
