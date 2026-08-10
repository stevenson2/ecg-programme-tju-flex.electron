# WiFi SoftAP beacon 不可见 — 专项排查交接文档 (2026-08-10)

> 本文件为 ESP32-S3 WiFi AP beacon 手机/电脑均不可见问题的完整排查背景与交接说明。
> 供新会话/新 AI 直接引用,避免重复排查。问题解决后在本文件追加结论并同步
> TUNING_HISTORY.md(下一章)。

## 症状
- 固件开 WiFi AP(SSID ESP32-ECG-3E8C,密码 12345678,WebServer:80):
  **手机和电脑 WiFi 列表均搜不到 beacon**;手机"添加网络"手动连接失败(找不到网络)。
- BLE 完全正常(App 可扫描/连接/收数据);串口/传感器/存储/报警全部正常。
- 平台:ESP32-S3-WROOM-1-N16R8(16MB/8MB PSRAM),PlatformIO board
  4d_systems_esp32s3_gen4_r8n16,arduino-esp32 **2.0.17**,platform-espressif32 6.13.0。

## 已完成排查(8 轮最小固件二分 + 诊断,勿重复)
| 测试 | 结果 |
|---|---|
| 纯最小 AP | ✅ 可连接 |
| + WiFi.macAddress() | ✅ 可连接 |
| + SPIFFS.begin | ✅ 可连接 |
| + WebServer(begin 在 softAP **前**) | ❌ 不可连接 |
| + WebServer(softAP 后 begin,正式固件同序) | ✅ 可连接 |
| + AI 模拟负载(Core0, prio1, 910ms busy/1s) | ✅ 可连接 |
| + 主循环 500Hz 模拟负载 | ✅ 可连接 |
| + BLE 广播(50-100ms) | ✅ 可连接 |
| + BLE 连接 + 高频 Notify(125Hz) | ✅ 可连接 |
| + BLE+AI+主循环 全组合 | ✅ 可连接 |
| + flash 读取负载(163KB XIP) | ✅ 可连接 |
| 运行时(满载 10s 后)启动 AP | ✅ 可连接 |

**结论:所有可模拟差异均不干扰 AP,正式固件全模块组合下仍不可见。**

诊断事实:
- STA 扫描:34-41 个网络(接收正常);AP 启动后 heap 剩 82KB(内存充足)
- softAP 返回 true;无 esp_wifi 驱动错误日志(CORE_DEBUG_LEVEL=3 下)
- sdkconfig:WiFi 任务/BT 控制器/BT 主机栈**全部绑定 Core 0**
  (CONFIG_ESP32_WIFI_TASK_PINNED_TO_CORE_0 / CONFIG_BT_CTRL_PINNED_TO_CORE_0 /
  CONFIG_BT_BLUEDROID_PINNED_TO_CORE_0 = y;CONFIG_ESP32_WIFI_SW_COEXIST_ENABLE=y)
- 已修复:setTxPower 移至 AP 启动后(原触发 "Neither AP or STA has been started"
  警告);移除 STA 扫描诊断;WebServer.begin 顺序确认正确

## 开源社区线索(深挖方向)
1. **espressif/esp-idf #13508** "SoftAP Not Detected on ESP32-S3, Works on ESP32":
   多例;换板解决(个体硬件)、焊接天线解决、特定环境/负载复现;一例 **LEDC 40MHz
   时钟输出干扰 WiFi AP**(禁用后恢复)
2. **Reddit r/esp32 "ESP32-S3 N16R8 SoftAP not showing SSID"**:同款板/模块,现象一致
3. **espressif/arduino-esp32 #9463 / WiFiManager #1467/#1488**:S3/S2 AP 不可见,
   core 版本行为差异
4. **esp32.com 论坛**多帖同类
5. 官方 RF Coexistence 文档:SoftAP TX Beacon + BLE 广播 = Y(支持),Connecting/
   Connected = C1(有约束);**建议 WiFi 与 BT 任务分核**

## 候选方案(按优先级)
- A:升级 arduino-esp32 core 3.x 重测(注意 TensorFlowLite_ESP32 兼容)
- B:正式固件"减模块"二分(禁真实 AI 推理、禁记录器 init 各一次)
- C:WiFi/BT 任务分核(sdkconfig:CONFIG_ESP_WIFI_TASK_CORE_ID /
  CONFIG_BT_*_PINNED_TO_CORE)
- D:STA 模式连接真实路由器(验证发射全链路,需用户路由器密码)
- E:LEDC/外设干扰排查(#13508 案例)
- F:模型权重载入 PSRAM(消除 flash XIP 竞争)
- G:换板验证(个体硬件;esp-idf #13508 多例换板解决)

## 当前固件状态(勿误改已完成功能)
- main.cpp:BLE 250Hz 单帧 9 列、停搏检测(3s<20mV)、REC_SCHEDULE、REC_AUTO、
  报警锁存 5s、initBLE 正常
- src/wifi/ecg_wifi.cpp:softAP(ch6/不隐藏/maxconn4)+ softAPConfig(192.168.4.1)
  + setTxPower(AP 后)+ WebServer 4 端点 + WIFI_ON/OFF
- platformio.ini:分区 v2(11M+4M SPIFFS)、monitor 460800、USB CDC
- 烧录:pio run -t upload(设备 COM4 在线);串口命令 WIFI_ON/WIFI_OFF/REC_STATUS
- git:主线 88e1854,全部推送 gitee;TH 至三十七章

## 研究结论更新 (2026-08-10 第二轮, 详见 TUNING_HISTORY.md 三十八章)

**社区证据修正**: #13508 本身就是"SoftAP 在 S3 不可见"同症状 open issue (In Progress, 无官方修复);
LEDC 40MHz 案例是其 comment 13 (GPIO21/XTAL_CLK, 禁用即恢复, 无官方解释); comment 14 = 天线焊接修复。
最强软件路径证据 = WiFiManager PR#1865 (2026-05, arduino 2.0.17 世代 S3 实测): 快速模式切换/预加载扫描/
STA disable 时序/channel 1 负载不稳 → 修复序列 WIFI_OFF→500ms→WIFI_AP→500ms→softAP(ch6)→500ms。
官方共存文档建议 WiFi 任务与 BT 任务分核 (本站全绑 Core 0); 分核可行: platformio.ini 加
build_flags `-DCONFIG_ESP32_WIFI_TASK_PINNED_TO_CORE_1=1` 即可 (wifi_task_core_id 是运行时 init 字段)。

**源码审计新差异 (8 轮二分未覆盖)**:
1. `WiFi.setTxPower(WIFI_POWER_19_5dBm)` (softAP 后) — 唯一未单独测试的 WiFi 配置调用;
2. BLE notify 实际 250Hz, 二分仅测过 125Hz;
3. 固件无 LEDC/PSRAM/ADC2 使用 → 候选 E/F/ADC 论排除;
4. NVS 擦除 (pio run -t erase, README 分区 v2 后本就要求) 是零代码高价值第一步。

**诊断固件已就绪 (pio run 通过)**: 新增 DIAG 命令 (串口+BLE 双通道):
`DIAG TXP <0|34|60|78>` / `DIAG CH <1|6|11>` / `DIAG SEQ <0|1>` / `DIAG NOTIFY <2|4>` / `DIAG AI <0|1>` / `DIAG`(状态)。
默认值 = 正式固件原行为。实验矩阵 0-10 步见 TH 三十八章 §2。

## 执行结果 (2026-08-10 第三轮→结论反转, 详见 TUNING_HISTORY.md 三十八章 §3)

**🎯 结论反转: AP 功能完全正常, 此前全部"不可见"判定为测量假象** (2026-08-10 末轮)

主动连接测试 (netsh wlan connect + profile) 证明:
- PC 网卡成功关联 ESP32-ECG-3E8C (BSSID 匹配, WPA2, ch6), 信号 96% / RSSI -17dBm
- DHCP 正常 (PC 获 192.168.4.2), HTTP GET /api/records → **200 OK**
- 网卡连接 2.4GHz 后扫描恢复: 42 网络, **ESP32-ECG 立即可见**

**两层测量假象**:
1. **PC 网卡扫描盲区**: Realtek 8852CE "Preferred Band=5G first" — 连接 5GHz 时不报告
   2.4GHz 网络 → 所有 netsh HIDDEN (矩阵/minap/分核) 均为假象
2. **串口复位**: "打开-关闭"串口触发 USB-Serial-JTAG 复位 → 命令设置丢失、AP 状态不确定
   → 早期矩阵无效

**用户症状的可能解释**: 关闭串口/monitor 会复位设备, AP (命令启动) 随之停止 → 手机搜不到。
**请按正确流程复测**: 保持 pio device monitor 打开 → 发 WIFI_ON → 手机看 WiFi 列表。
PC 端"搜不到"可用 5G first 扫描盲区解释 (用户电脑同理)。

**诊断工具 (保留在固件, 默认行为与原固件一致)**: DIAG TXP/CH/SEQ/NOTIFY/AI/STA/STAOFF。
PC 端: ESP32-ECG-3E8C profile 已建 (手动模式), netsh wlan connect name=ESP32-ECG-3E8C 可直接连。

## 约束
- 不回退已完成功能(BLE 报警链路/存储/云端)
- 固件改动 pio run 必须通过;烧录/命令先告知用户
- 用户配合有限,优先一次性设计诊断,减少烧录轮次
- 遵守 AGENTS.md(§9 用户手机验证期间禁串口脚本;重大决策 TH 留痕)
