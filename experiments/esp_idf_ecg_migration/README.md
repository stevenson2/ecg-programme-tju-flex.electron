# ESP-IDF ECG Core Migration（实验工程）

本工程验证从 Arduino 固件向 ESP-IDF 迁移的第一步：

- `components/ecg_core`：滤波、心率、心律安全、AF、VF、模拟/回放源码（从原 Arduino 移植，算法参数未动）。
- `components/ecg_ai`：esp-tflite-micro + ESP-NN 的 exp7c INT8 推理组件。
- `components/ecg_storage`：SPIFFS ECGR 录制。
- `components/ecg_wifi`：SoftAP + HTTP 记录列表/下载。
- `components/ecg_ble`：NimBLE NUS 服务。
- `main.cc`：模拟器 + AI 流式 + 心率/规则链 + Storage/WiFi/BLE 初始化。

## 构建（短路径）

```bat
cd C:\esp\esp_idf_ecg_migration
idf.py set-target esp32s3
idf.py build   // 如内存紧张用 build 目录内 ninja -j2
```

仓库内路径因 Windows 路径长度限制可能导致 mbedtls/ESP-TFLM 编译依赖文件路径超长，
建议使用 `C:\esp\esp_idf_ecg_migration` 短路径构建。该短路径的 CMakeLists.txt
已将 `EXTRA_COMPONENT_DIRS` 指向仓库 `components` 目录。
