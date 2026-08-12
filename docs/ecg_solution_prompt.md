# ESP32-ECG 项目解决方案 Prompt(交接文档, 2026-08-12)

> 用途: 新会话/新 AI 直接接手继续工作, 不重复已完成排查。
> 遵守根目录 AGENTS.md 全部规范(决策自主/联网支撑/TH留痕/§9 手机验证期禁串口/用户禁止切换电脑 WiFi——会断对话)。

## 一、项目与设备当前状态

- 硬件: ESP32-S3-WROOM-1-N16R8, PlatformIO board `4d_systems_esp32s3_gen4_r8n16`, arduino-esp32 2.0.17
- 设备: USB COM4 在线, **固件 bc0d865 已烧录**(含全部修复, AP 上电自动启动, DIAG 响应正常)
- 仓库: 18 个提交全部推送 gitee, 工作区干净(仅 untracked papers/exe)
- 环境: pio = `C:\Users\cai\.platformio\penv\Scripts\pio.exe`; flutter = `D:\flutter\bin\flutter.bat`; WSL 训练环境(python3 + tensorflow 2.21 + pywt 已装); 串口脚本在 `C:\Users\cai\AppData\Local\Temp\opencode\serial_*.py`(serial_once / serial_hold / serial_seq2 / serial_ecg_diag2)

## 二、已完成修复(勿重复, 含验证状态)

| 提交 | 修复 | 验证 |
|---|---|---|
| c89418d/a2e1d05 | WiFi beacon 排查闭环: AP 正常, "不可见"= PC 网卡 5G-first 扫描盲区 + 串口复位假象 (TH §38) | ✅ |
| d9b9aa0 | AP 上电自动启动; App BLE 命令缺 `\n` 根因(定时录制命令从未送达)→ App 补 \n + 固件 100ms 超时提交防御 | ✅ 已烧录 |
| 783a8aa | App 连接后 requestConnectionPriority(high)(flutter_blue_plus 1.36.8 命名参数 API) | 含在 APK |
| 9021f93 | APK 构建修复: Gradle 补腾讯镜像 download.flutter.io 仓库 → APK 47.9MB 构建成功 | ✅ |
| da930cc | AI 输入链**窗口级零相位 0.5Hz 高通**(训练链 filtfilt 一致; 因果 IIR 扭曲 QRS: 峰度 0.80→-1.21, RMS 差 34%) | ✅ |
| 3c8aaed | **A 方案**: AI 阈值 0.35→0.60 + 多拍确认 2→5(真实 ECG 分布偏移误报抑制); **BLE 主动请求连接参数 15-22.5ms**(折中 WiFi 共存) | ✅ 真机: AI 异常率 12-55%→0.35% |
| a58f5d7 | recorder 保留策略删除 bug: f.name() 纯文件名致 SPIFFS.remove 静默失败 → removeRecordFile 补全 /ecgdata/ 路径 | ✅ 真机: 堆积 32→10 条, 删除成功 |
| bc0d865 | **WebServer 下载 Content-Length 重复 bug**(send String 重载追加第二个)→ 改 streamFile; BLE GAP 连接参数协商日志 | ✅ **下载成功(用户确认)** |

**真实 ECG 排查完整结论**(TH §40): 波形视觉确认正常窦性 ECG; 深 S 波(S≈R) + 高大 T 波为胸前/锁骨导联固有形态(换位无法消除); AI 误报根因 = 因果 IIR 相位失真 + 模型分布偏移; A 方案已压制误报(截图确认"AI 正常"); 根治需模型微调(B 方案, 用户暂缓)。

## 三、未解决问题(按优先级)

### 1. BLE 波形阶梯感(未解决, 用户反馈"蓝牙仍不光滑")
- **现象**: App 波形呈阶梯状/点状连接(截图 `ECG-figs/cellphone ECG.jpg` 已分析); 串口 100Hz 平滑; 250Hz 数据批量到达特征
- **已做**: 固件 onConnect `esp_ble_gap_update_conn_params`(15-22.5ms, 3c8aaed)+ GAP 回调打印协商结果(bc0d865); App requestConnectionPriority(783a8aa, 需确认用户装的是 47.9MB 最新 APK)
- **卡点**: 用户未看到串口 `[BLE] conn params evt:` 日志(可能被 100Hz CSV 刷屏淹没)
- **下一步**:
  a. 用过滤串口查看: `python serial_hold.py COM4 60 "" "conn params"`(serial_hold 支持第 4 参过滤; 或临时提高过滤)
  b. 确认 App 版本: 是否 `ecg_app\build\app\outputs\flutter-apk\app-release.apk`(47.9MB, 含 783a8aa); 旧版则 requestConnectionPriority 未生效
  c. 若 conn_int≈15-22(协商成功)仍阶梯 → 问题在 App 绘制/数据流(ecg_provider._addSample 每 15 样本 notifyListeners 批量重绘? 检查 BLE 包到达节奏)
  d. 若 conn_int=30+(Android 拒绝外设请求)→ App 端 requestConnectionPriority 是唯一途径, 需新版 APK + 验证调用成功(可加日志/或 flutter_blue_plus 的 connectionState 显示)
  e. 备选: MTU 检查(帧 ~50B, 默认 MTU 23 分 3 包/帧, 250Hz=750 包/s; 大 MTU 一帧一包更稳)——App 可 requestMtu(247)

### 2. 本地 App 查看心电图功能(用户新需求)
- **现状**: `ecg_app/lib/pages/playback_page.dart` **已存在**(回放 250Hz 波形, PlaybackProvider); `record_list_page.dart` 下载成功(存到 `downloadDir/ecg_records/<id>.ecgr`)但**只提示文件路径, 未跳转回放**
- **方案**: record_list_page 下载成功后跳转 PlaybackPage(传 .ecgr 文件路径); 或记录列表加"本地回放"按钮; 用 `ecg_app/lib/services/ecg_record_codec.dart` 解码 .ecgr(三端共用格式)
- 需跑 `flutter test`(现有 164 测试)后构建 APK(注意: APK 构建需 `flutter precache --android` 已补齐引擎; Gradle 腾讯镜像已配置)

### 3. 定时录制链路验证(用户反馈"尚未录制")
- **步骤**: 用户 App 重连蓝牙 → 前台等 2 分钟(App 调度 1min 间隔/20s 时长, 发 REC_START/REC_STOP)→ 串口查 `REC_STATUS` 确认 count 增加 → 手机连 AP 看 `http://192.168.4.1/api/records` 出现新记录
- **注意**: App 命令链路已修(\n + 固件超时防御), 但需 App 前台 + BLE 连接; 固件 REC_SCHEDULE(串口版)也可用但设备复位后丢失

### 4. 心率残余高估(86 vs 实测 64-80, ~10%)
- **分析**: T 波/次峰偶发误检混入 RR 缓冲(深 S/高 T 形态); heartrate.cpp v4.2 已有 5-15Hz QRS 带通 + 200ms 不应期 + 自适应阈值
- **下一步**: 抓串口 CSV 的 bpm 列分布(看是否有间歇性 2× 值); 若偶发误检 → 调 `MIN_PEAK_RATIO`(1.2→1.3-1.5)或不应期; 换电极位置后已从 110→86, 残余差距先数据观察

### 5. 模型微调 B(用户暂缓, 根治 AI 分布偏移)
- **方向**: 收集真实 ECG 数据(用户静止贴电极录 2 分钟, 标注正常)→ WSL 微调(加载 `models/best_resnet_large_exp6_sgd.h5`, 冻结骨干低 lr)→ 评估 → 导出 INT8(`ecg_model_exp6_sgd_int8.tflite`)→ 更新 `include/ai_inference/ecg_model_data.h` → 上板; 可回退 A 方案激进参数
- 遵循 `ai-training` skill 流程; 参考 TH §40

## 四、关键文件索引

- 固件: `src/main.cpp`(AI 输入/模式切换/AP 自动启动), `src/bluetooth/ble.cpp`(GAP 日志/连接参数/命令 \n 超时), `src/wifi/ecg_wifi.cpp`(streamFile 下载), `src/ai_inference/ai_inference.cpp`(窗口零相位), `src/storage/ecg_recorder.cpp`(removeRecordFile), `include/ai_inference/tflite_settings.h`(A 方案参数), `src/filter/filter.cpp`(aiApplyFilterWindow)
- App: `ecg_app/lib/services/ble_service.dart`(sendCommand \n / requestConnectionPriority), `record_api.dart`(下载, baseUrl=192.168.4.1), `record_list_page.dart`(下载→回放对接点), `playback_page.dart`(回放, 已存在), `ecg_record_codec.dart`(ecgr 解码)
- 文档: `TUNING_HISTORY.md` §38-40(完整证据链), `docs/wifi_debug_brief.md`

## 五、验证与操作约束

- 固件改动后 `pio run` 必须通过; 烧录/串口命令先告知用户(用户自己烧录或我烧均可, 设备 COM4)
- **用户禁止切换电脑 WiFi**(会中断用户与本 AI 的对话)——云端/下载验证一律用用户手机
- 用户手机验证期间禁串口脚本(§9); 一次性命令(serial_once/hold)可短用
- 长时命令 timeout: 训练 900s, 评估 600s
- 决策留痕: TUNING_HISTORY.md 追加章节(下一章=四十一)

## 六、建议下一步顺序

1. BLE 阶梯: 过滤串口看 conn_int + 确认 App 版本 → 定位协商是否成功
2. 定时录制验证(用户配合 2 分钟)
3. 本地回放对接(playback_page 接入下载流程)+ flutter test + 构建 APK
4. 心率数据观察
5. 模型微调 B(用户决定时机)
