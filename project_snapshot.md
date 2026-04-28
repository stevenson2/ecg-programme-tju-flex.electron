# ESP32-ECG-Ver1 - 项目快照

## 基础信息

| 项目 | 说明 |
|------|------|
| **项目名称** | ESP32-ECG-Ver1 |
| **开发板** | ESP32-S3-DevKitM-1 |
| **框架** | PlatformIO + Arduino |
| **采样率** | 250Hz (每 4ms 一个样本) |
| **PC 工具** | Python (pySerial + matplotlib) |
| **手机 App** | Flutter (flutter_blue_plus + provider) |

---

## 系统架构与数据流

```
┌──────────────────────────────────────────────────────────┐
│                    ESP32-S3 (C++)                         │
│                                                          │
│  ecg_simulator.cpp       filter.cpp         ble.cpp      │
│  ┌──────────────┐    ┌──────────────┐   ┌───────────┐   │
│  │ 生成P-QRS-T   │───→│ HP 0.5Hz     │   │ BLE NUS   │   │
│  │ + 7种噪声     │    │ LP 40Hz      │──→│ TX Notify │   │
│  │ 叠加直流1.65V │    │ Notch 50Hz   │   │ CSV格式   │   │
│  └──────────────┘    └──────────────┘   └───────────┘   │
│        │                    │                │           │
│        ▼                    ▼                ▼           │
│  串口输出 CSV (3列)    ←─ 滤波器输出 ──  BLE发送        │
└───────────────────────┬─────────────────────────────────┘
                        │
           ┌────────────┴────────────┐
           ▼                         ▼
     pc_tools/ecg_plotter.py    ecg_app (Flutter)
     (Python实时绘图)           (手机端心电监测)
```

**串口/BLE CSV 数据格式（每行）：**
```
<clean_voltage>,<noisy_voltage>,<filtered_voltage>\r\n
```
示例：`0.253,-0.187,0.241`

---

## 已完成功能

### 1. ESP32 端 (C++)

**`src/signal_generator/ecg_simulator.cpp`** — 临床级心电信号生成
- P-QRS-T 波用 5 个高斯函数叠加模拟，周期 200 点 @250Hz = 75bpm
- 直流偏置 1.65V，心电峰峰值 ~1.3V (0.95~2.28V)
- 7 种噪声按 IEC60601-2-51 场景三分布：
  - 工频 50Hz+100Hz (43%) | 基线漂移 (27%) | 肌电 (17%)
  - 运动伪影 (7%) | 电极尖峰 (5%) | 系统白噪声 (3%) | EMI (1%)
- 总 RMS ≈ 0.20V，SNR ≈ 16dB

**`src/filter/filter.cpp`** — 三级级联数字滤波器
- 第1级：二阶 Butterworth 高通 0.5Hz（去基线漂移）
- 第2级：二阶 Butterworth 低通 40Hz（去肌电）
- 第3级：二阶 50Hz 陷波器 Q=30（精准陷零）
- 直接 II 型转置结构，float 运算，状态变量 6 个

**`src/bluetooth/ble.cpp`** — BLE NUS UART 透传
- Nordic UART Service (NUS) 标准 UUID
- 设备名 "ESP32-ECG"，TX Notify 发送 CSV，RX Write 接收指令
- 支持 'r' 重置滤波器，'s' 重置信号发生器

**`src/main.cpp`** — 主循环
- 每 4ms 采样：生成 → 滤波 → BLE发送 → 串口输出
- LED 翻转指示运行状态

### 2. PC 端 (Python)

**`pc_tools/ecg_plotter.py`**
- 实时三通道波形显示（绿=纯净, 红=带噪, 蓝=滤波后）
- 快捷键：←→缩放时间轴，↑↓缩放Y轴，1/2/3切换通道，空格暂停，R复位，Q退出

### 3. 手机端 (Flutter)

**`ecg_app/`** — 心电监测 App
- BLE 扫描连接 "ESP32-ECG" 设备
- 三通道实时波形绘制（CustomPainter 直接渲染）
- 环形缓冲区 500 点（2秒），每 20 点刷新 UI
- R 波检测 + 心率计算（滑动平均）
- 信息面板显示瞬时值和统计数据

---

## 核心配置与常量

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 250 Hz | 每样本间隔 4ms |
| 心拍周期 | 200 点 | 对应 75bpm |
| BLE 设备名 | ESP32-ECG | - |
| NUS Service UUID | 6E400001-B5A3-F393-E0A9-E50E24DCCA9E | - |
| NUS TX UUID | 6E400002-B5A3-F393-E0A9-E50E24DCCA9E | Notify |
| NUS RX UUID | 6E400003-B5A3-F393-E0A9-E50E24DCCA9E | Write |
| 滤波器类型 | IIR Biquad 级联 | HP 0.5→LP 40→Notch 50 |
| 缓冲区大小 | 500 点 | 2秒数据 |
| R波阈值 | 0.3 V | 滤波后信号 |
| 串口波特率 | 115200 | - |

---

## 待办事项 / 已知问题

（请根据你的实际对话内容补充，以下是常见方向）
- [ ] 验证实际 ADC 采样时的滤波器性能
- [ ] 是否需要在 Flutter 端做二次滤波？
- [ ] 50Hz 陷波器 Q=30 是否足够应对实际电网波动？
- [ ] 连续长时间运行的内存稳定性
- [ ] 实际硬件（模拟前端）接入后的标定

---

## 关键技术要点

1. **滤波器系数计算**：在 MATLAB/Python 中设计后硬编码到 filter.cpp，非运行时计算
2. **R波检测**：基于固定阈值 0.3V + 最小间隔 200ms（50点），心率滑动平均 α=0.7
3. **噪声模拟真实性**：7种噪声的能量分布严格参考 IEC60601-2-51 家用场景
4. **UI 性能优化**：Flutter 端每 20 点 notify 一次，避免每秒 250 次重绘
