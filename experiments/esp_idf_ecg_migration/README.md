# ESP-IDF ECG Core Migration（实验工程）

> **验证状态（2026-10-23，SuperMini 4MB 板）**：4MB 正式版构建+部署通过
> （应用 1,622,960 B，factory 槽余约 43.7%）；AI_VERIFY 两段回放 92 窗与 PC 参考
> 逐字节一致（23,000 INT8 输入字节，输入/输出/raw/confirmed 全对齐）；保留式启动
> 未改动任何 ECG 记录（SPIFFS 摘要不变），NVS 仅无线 PHY 校准键更新。以上限于
> 数字回放与数字链，不含真实 AFE 采集、整机功耗、完整报警时序或诊断性能。
> 详细证据保存在研究侧验证报告中，本 README 只保留结论与范围。

本工程验证从 Arduino 固件向 ESP-IDF 迁移的第一步：

- `components/ecg_core`：滤波、心率、心律安全、AF、VF、模拟/回放源码（从原 Arduino 移植，算法参数未动）。
- `components/ecg_ai`：esp-tflite-micro + ESP-NN 的 INT8 推理组件（当前板上模型 **v3-A**）。
- `components/ecg_storage`：SPIFFS ECGR 录制。
- `components/ecg_wifi`：SoftAP + HTTP 记录列表/下载。
- `components/ecg_ble`：NimBLE NUS 服务。
- `main.cc`：模拟器 + AI 流式 + 心率/规则链 + Storage/WiFi/BLE 初始化。

## 构建（明确选择板型）

```bat
idf.py -B build_supermini -D SDKCONFIG=sdkconfig.supermini -D SDKCONFIG_DEFAULTS=sdkconfig.defaults.supermini build
```

在本产品仓固件目录运行；先加载 ESP-IDF 环境。可以用 `-B` 指向独立短路径，
不用复制另一套源码。SuperMini 为 4MB Flash / 2MB Quad PSRAM，沿用
`partitions_ecg_supermini.csv`（应用 0x10000/2816KiB，storage 0x2d0000/960KiB）。
16MB 默认配置会启用额外的 1,530,000 字节回放语料，不能用于此板。
选用 `sdkconfig.supermini`（含带后缀的副本）时，CMake 自动选 SuperMini defaults；
若显式提供的 defaults 造成 16MB 或错误分区配置，构建将报错。

固定输入板测可增加 `-D ECG_FIXED_INPUT_BOARD_TEST=ON`：使用产品分析链和内置回放，
拒绝进入 AFE，跳过 SPIFFS/录制与无线初始化。默认 OFF；这项测试不验证无线与存储。
保留设备数据时只写已核实分区范围内的应用，不执行整片擦除或通用全量 flash 命令。

正式固件启动采用保留策略：SPIFFS 挂载失败不格式化，启动不清理无效/过量记录，
不重建索引；自动录制默认关闭。需要录制时显式使用 `REC_START` 或 `REC_AUTO 1`。
NVS 初始化失败时无线功能报错返回，不自动擦除 NVS；正常 PHY 校准仍可能更新其校准键。

正式镜像可用 USB/UART 的 `AI_VERIFY 0` / `AI_VERIFY 1` 做逐窗核对：
重置 AI 与滤波状态，分别输入现有回放段前 20,000 点（40s @500Hz），通过真实异步推理，
输出 `AI_TRACE,sample_index,250字节INT8十六进制,q0,q1,raw,confirmed`。
输入结束后暂停喂入、保留查询，待末个在途结果完成后发送 `AI_VERIFY STOP` 返回模拟器。
此诊断默认关闭，不初始化 AFE；运行时拒绝切入 AFE 或改变其他采集/录制配置。
它验证数字链的一致性，不证明回放标签、输出医学含义或真实采集保真。
