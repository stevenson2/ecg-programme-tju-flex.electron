# PLAN：BLE + 发热 降功耗方案（SuperMini / ESP32-S3）— 2026-10-20

> 分支 `feat/power-wifi-ondemand-ble-stream`。**未刷机**，所有电流/温度数字除"实测"外均为**估算**（来源：ESP32-S3 datasheet 典型值量级 + 常见 SOT-23 LDO 热阻），必须按第 6 节协议实测确认。

## 0. 结论先行（按收益排序）
| # | 动作 | 状态 | 预估节省 (估算) |
|---|---|---|---|
| 1 | **Wi‑Fi SoftAP 开机常开 → 按需开启** | ✅ 已实现（宏可回退） | **~80–120 mA**（SoftAP 不能 modem-sleep，PS_NONE 下 RX 常开）。这是目前最大热源 |
| 2 | **模型数组放 flash（const）** | ✅ 已实现 | 不省电流，但 DRAM 静态占用 97.6% → 48.6%（释放 ~163 KB）——为后续 PM/BLE 缓冲留余量 |
| 3 | Wi‑Fi AP 最大发射功率 20 dBm → 8.5 dBm | ✅ 已实现 | 上传期间峰值 ~100 mA 量级下降 |
| 4 | BLE 流开关 STREAM 1/0（关流=1 Hz 摘要+报警即发）+ 关流时慢连接间隔(200–400 ms, latency 2) | ✅ 已实现（默认仍开流，兼容冻结 App） | 关流时 BLE 平均 ~3–8 mA → ~0.5–1 mA |
| 5 | CPU 240 MHz 常跑、无 PM → 开 `CONFIG_PM_ENABLE` + DFS(80–160 MHz) + tickless idle + light sleep | 📝 仅计划（需改采样架构） | ~20–30 mA |
| 6 | ADC oneshot 在主循环 500 Hz 轮询 → 改 ADC continuous(DMA) / GPTimer 驱动 | 📝 仅计划（第 5 项前提） | 让 CPU 可进 light sleep |
| 7 | 二进制 int16 delta 帧 + 每 notify 打包 ~0.5 s | 📝 仅计划（App 需改解析） | 开流时空口字节数 ↓ ~5–8×，notify 次数 62.5/s → 2–4/s |
| 8 | 2M PHY 只用于批量同步 | 📝 仅计划 | 同步时空口时间减半 |
| 9 | 供电/LDO：从 5V USB 供电时 LDO 发热（见第 4 节） | 📝 硬件建议 | 热量本身，不是电流 |

## 1. 现状审计（代码事实）
- Wi‑Fi：`app_main` 中 `ecgWifiInit(); ecgWifiStart();` → **SoftAP 开机常开**，`esp_wifi_set_ps(WIFI_PS_NONE)`，`MAX_WIFI_TX_POWER=20 dBm`。只停不 deinit。
- BLE：NimBLE、1 连接、四个角色全开（CENTRAL/OBSERVER 无用）。TX 0 dBm（已降过）。广播：快档 30–60 ms 60 s → 慢档 ~1 s（已做）。只要 TX 订阅就以 125 帧/s ASCII CSV 流，每 2 帧一 notify（62.5 notify/s），连接间隔 15–30 ms。
- CPU：`CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ=240`，`# CONFIG_PM_ENABLE is not set`，FreeRTOS 1000 Hz tick。
- 主循环：`vTaskDelayUntil(…,2)` 500 Hz 节拍，**不是忙等**（✓）；但每 2 ms 唤醒一次 + `afeHalReadSample` 做 oneshot × oversample 次，阻止任何深/浅睡。
- 未发现 LED 驱动代码；SuperMini 板载电源 LED（若有）常亮 ~1–3 mA，可剪/去电阻。
- **发现并修复的问题**：`model.cc` 的模型数组未加 `const`，链接到 `.data` → 167 KB 进 DRAM（开机还要从 flash 拷贝）。DIRAM 静态 97.6%（仅剩 8 KB），这也是 Wi‑Fi + BLE + TFLM arena 抢堆的根源。
- 修复的构建阻断：`ecg_ai.cpp` 对 `volatile` 用 `++`（C++20 -Werror=volatile），R 峰提交 add1b8c 从未编译过。

## 2. 已实现（本分支）
### 2.1 Wi‑Fi 按需（`ecg_wifi.*` + `main.cc`）
- 开机**不开射频**：`ECG_WIFI_AUTOSTART_AT_BOOT=0`（设 1 恢复旧行为）。`ecgWifiInit` 仍只做 NVS/netif/事件循环。
- 开启途径：BLE/串口 `WIFI_ON` 或 `UPLOAD`；可选定时同步 `ECG_WIFI_SYNC_INTERVAL_S>0`（每 N 秒若有记录且不在录制则开 AP，默认 0 关闭）。
- 关闭途径：`WIFI_OFF` / `UPLOAD_DONE`（App 传完立即发）；**空闲超时** `ECG_WIFI_IDLE_TIMEOUT_S=120`（无 HTTP 请求且无已关联手机）；**硬上限** `ECG_WIFI_MAX_ON_S=900`。自动关时经 BLE 回 `WIFI_OFF auto`。下载每个 chunk 都刷新活动时间，长文件不会被切断。
- 关时 `esp_wifi_stop()` + `esp_wifi_deinit()` + `esp_netif_destroy_default_wifi()`（`ECG_WIFI_DEINIT_ON_STOP=1`，设 0 恢复只 stop）。
- AP 最大发射功率 `ECG_WIFI_MAX_TX_QDBM=34`（8.5 dBm）。
- 新命令：`WIFI?`（状态+空闲秒）、`UPLOAD`、`UPLOAD_DONE`。
### 2.2 BLE（`ble.*` + `main.cc`）
- `STREAM 1/0`、`STREAM?`；`ECG_BLE_STREAM_DEFAULT=1`（默认开流兼容冻结 App；App 支持后改 0）。关流时每秒 1 帧（同 10 列 CSV 格式，旧 App 照样能解析，只是波形稀）+ 报警位变化立即补发。
- `bleSetStreamActive()`：关流或未订阅 → 慢连接参数；开流且订阅 → 快参数。
- 慢参数 latency 0 → `ECG_BLE_IDLE_LATENCY=2`（监督超时 5 s > (1+2)×400 ms×2 = 2.4 s，合规）。
- TX 功率宏 `ECG_BLE_TX_PWR_LVL`（默认 0 dBm 不变；贴身+手机 1 m 内可试 `ESP_PWR_LVL_N6`）。
### 2.3 内存
- `model.cc`：`#include "model.h"` + `alignas(16) const` → 模型放 `.rodata`(flash)。

## 3. 计划（未实现）
1. **PM + light sleep**（收益第二大）：`CONFIG_PM_ENABLE=y`、`CONFIG_FREERTOS_USE_TICKLESS_IDLE=y`、`esp_pm_configure({max 160, min 80, light_sleep_enable=true})`；`CONFIG_FREERTOS_HZ` 1000→100/250（主循环改成事件驱动后）；`CONFIG_BT_CTRL_MODEM_SLEEP=y` + 32 kHz 低速时钟源（SuperMini 无外部 32k 晶振 → 用内部 RC，BLE 需放宽 sleep clock accuracy 500 ppm，需实测连接稳定性）。
   - 前提：采样不能靠 CPU 每 2 ms 醒来。改 `adc_continuous`（DMA，500 Hz×oversample 由硬件定时），每 ~50 ms 一个 DMA 帧回调 → 主任务批处理 25 个样本。AI 推理已异步到核 1，不受影响。
   - 风险：DFS 下 ADC/UART 波特率需锁（`ESP_PM_APB_FREQ_MAX` lock 在 UART 活动/WAVE 时持有）。
2. **二进制帧**：`[0xEC][seq u8][n u8][base int16][delta int8×(n-1)][flags/hr/sqi/asrc 4B][crc8]`，0.5 s 打包一 notify（≤244 B，MTU 247）；HELLO 能力位协商，旧 App 保持 CSV。
3. **离线报警队列 + 批量同步**：断连期间报警事件入 NVS/SPIFFS 环形队列，连上后先发；记录同步用 2M PHY（`ble_gap_set_prefered_le_phy`），仅同步期间。
4. 关掉 NimBLE CENTRAL/OBSERVER 角色（省 flash/RAM 少量）。
5. 广播：慢档 1 s 已合理；可在“佩戴中且曾配对”时用 2 s，并缩短广播包（只放 UUID）。

## 4. SuperMini 发热：LDO 估算（估算，需实测）
- SuperMini 常见为 SOT‑23‑5 3.3 V LDO（ME6211/同类），θJA ≈ 200–250 °C/W（估算，手册量级）。
- USB 5 V 供电，当前负载估 150 mA（Wi‑Fi AP 常开 + BLE 流 + 240 MHz）：P_LDO = (5.0−3.3)×0.15 ≈ **0.26 W → 温升 50–65 °C**，叠加 ESP32‑S3 芯片自身 3.3×0.15≈0.5 W，在小板无散热时 **85 °C+ 完全吻合**。
- 降到 ~40 mA（Wi‑Fi 关、关流、未开 PM）：P_LDO ≈ 0.07 W → 温升 ~15 °C；开 PM 后 ~15–20 mA → 几乎不热。
- 锂电 4.2 V 直供 LDO：压差 0.9 V，同电流下 LDO 发热约为 USB 供电的一半。**测温时要区分 USB 供电和电池供电**。
- 硬件建议：产品板换小封装 Buck（如 TPS62840 类，效率 >85%）或至少 DFN 封装 LDO + 铺铜；去掉电源 LED。

## 5. 预估电流预算（全部为估算）
| 场景 | 现状 | 本分支(默认) | 本分支 + STREAM 0 | + PM/DMA(计划) |
|---|---|---|---|---|
| 佩戴、App 未开 | ~140–180 mA（AP 常开） | ~45–60 mA | ~42–55 mA | **~10–20 mA** |
| App 看波形 | ~150–190 mA | ~50–65 mA | — | ~20–30 mA |
| 上传中(Wi‑Fi) | ~180–250 mA | ~150–200 mA（8.5 dBm） | — | 同左（短时） |

## 6. 实测验证协议（刷机前后各测一次，同一块板）
**设备**：USB 电流表（如 Power Profiler Kit II / UM34C，≥1 kHz 采样更好）、热电偶或红外测温枪贴 LDO 与 S3 屏蔽罩、室温记录。
**固定条件**：同一块 SuperMini、同一电源（分别测 USB 5 V 与 3.7 V 锂电）、板子平放无风、手机距离 0.5 m。
**步骤**（每段 15 min，记录平均电流 + 第 15 min 的 LDO/芯片温度）：
1. A：旧固件（main 分支），BLE 连接 App 看波形。
2. B：新固件，BLE 连接 App 看波形（STREAM 1）。
3. C：新固件，App 后台/STREAM 0。
4. D：新固件，未连接（慢广播）。
5. E：新固件，`UPLOAD` → 下载一条记录 → 不发 UPLOAD_DONE，验证 120 s 后 `WIFI_OFF auto`，并看 `heap_caps_get_free_size` 在 WIFI_ON/OFF 前后恢复（deinit 生效）。
6. 回归：报警延迟（回放 abnormal 段）在 STREAM 0 下仍立即推送；HELLO 正常；BLE 断连重连 10 次无异常。
**通过线（建议）**：D 场景 ≤60 mA 且 LDO ≤55 °C（USB 供电，室温 25 °C）；E 场景自动关成功率 10/10。

## 7. 回退开关一览
`ECG_WIFI_AUTOSTART_AT_BOOT=1`、`ECG_WIFI_DEINIT_ON_STOP=0`、`ECG_WIFI_IDLE_TIMEOUT_S=0`、`ECG_WIFI_MAX_ON_S=0`、`ECG_WIFI_MAX_TX_QDBM=78`、`ECG_BLE_STREAM_DEFAULT=1`、`ECG_BLE_IDLE_LATENCY=0`、`ECG_BLE_TX_PWR_LVL=ESP_PWR_LVL_N0`。
