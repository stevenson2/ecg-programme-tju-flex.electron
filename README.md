# ESP32-ECG 心电采集系统

> **ESP32-S3-SUPERMINI (ESP32S3FH4R2) | PlatformIO + Arduino | 250Hz 采样 | BLE NUS 透传**

## 目录

1. [项目简介](#项目简介)
2. [系统架构](#系统架构)
3. [快速开始](#快速开始)
4. [项目结构](#项目结构)
5. [开发板说明](#开发板说明)
6. [PC 端工具](#pc-端工具)
7. [手机端 App](#手机端-app)
8. [温度与功耗诊断](#温度与功耗诊断)
9. [核心技术要点](#核心技术要点)
10. [核心配置](#核心配置)

---

## 项目简介

基于 ESP32-S3 的便携式心电采集系统，通过 BLE 向手机 App 实时传输三通道心电波形数据，具备临床级信号模拟、三级数字滤波和板上心率检测功能。

**硬件平台**：ESP32-S3-SUPERMINI (ESP32S3FH4R2)  
**开发框架**：PlatformIO + Arduino  
**采样率**：250Hz (每 4ms 一个样本)  
**PC 工具**：Python (pySerial + matplotlib)  
**手机 App**：Flutter (flutter_blue_plus + provider)

---

## 系统架构

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
<clean_voltage>,<noisy_voltage>,<filtered_voltage>,<bpm>,<true_bpm>,<sqi>,<motion>
```
示例：`0.253,-0.187,0.241,75,75,0.87,0`

---

## 快速开始

### 1. 编译烧录

```bash
# 编译
pio run

# 烧录到 SUPERMINI
pio run -t upload

# 打开串口监视器
pio device monitor -p COM9 -b 115200
```

### 2. 使用 PC 绘图仪

```bash
python pc_tools/ecg_plotter.py
```

### 3. 使用手机 App

1. 进入 `ecg_app/` 目录
2. 执行 `flutter pub get`
3. 执行 `flutter run`
4. 扫描并连接 "ESP32-ECG" 设备

### 4. 使用工具箱

双击 `ESP32-ECG toolbox.bat` 启动菜单界面，可选择：
- [1] PC Plotter — 三通道波形绘图
- [2] Serial Monitor — 串口数据监视
- [3] Compile & Upload — 编译烧录固件

### 5. 串口指令

| 指令 | 功能 |
|------|------|
| `r` / `R` | 重置数字滤波器 |
| `s` / `S` | 重置信号发生器 |
| `m` / `M` | 切换 模拟/真实AFE 输入模式 |
| `t` / `T` | 打印温度状态详情 |
| `c` / `C` | 打印 CPU 当前频率 |

---

## 项目结构

```
ecg-programme-tju-flex.electron/
├── platformio.ini              # PlatformIO 构建配置
├── README.md                   # 本文件
├── LICENSE                     # 项目许可证
│
├── src/                        # ESP32 固件源码
│   ├── main.cpp                # 主程序入口
│   ├── bluetooth/
│   │   └── ble.cpp             # BLE NUS UART 透传模块
│   ├── filter/
│   │   └── filter.cpp          # 三级数字滤波器
│   ├── heartrate/
│   │   └── heartrate.cpp       # 心率检测 (简化 Pan-Tompkins)
│   ├── signal_generator/
│   │   └── ecg_simulator.cpp   # 临床级心电信号生成器
│   ├── adc_afe/
│   │   └── afe_hal.cpp         # ADC 采集硬件抽象层
│   └── thermal/
│       ├── thermal.h           # 温度监测接口
│       └── thermal.cpp         # 温度监测实现
│
├── include/                    # 头文件 (对应 src/ 各模块)
│
├── pc_tools/                   # PC 端 Python 工具
│   ├── ecg_plotter.py          # 实时三通道心电波形绘图仪
│   ├── capture_debug.py        # 数据捕获调试工具
│   └── find_port.py            # ESP32 串口自动探测
│
├── ecg_app/                    # Flutter 手机端 App
│   └── lib/
│       ├── main.dart           # App 入口 + 主界面
│       ├── models/ecg_data.dart # 数据模型
│       ├── providers/ecg_provider.dart # 状态管理
│       ├── services/ble_service.dart # BLE 连接与数据接收
│       └── widgets/
│           ├── ecg_waveform.dart # CustomPainter 波形绘制
│           └── info_panel.dart   # 心率/连接状态信息面板
│
├── ecg_app_backup/             # Flutter App 备份 (迁移前)
│
├── test/                       # 测试与诊断固件
│   ├── power_diagnostic_test.cpp # 功耗诊断测试
│   └── adc_test.cpp              # ADC 测试
│
├── ecg_toolbox.ps1             # PowerShell 工具箱
└── ESP32-ECG toolbox.bat       # 工具箱入口
```

---

## 开发板说明

### ESP32-S3-SUPERMINI vs ESP32-S3-DevKitM-1

| 参数 | DevKitM-1 (原) | SUPERMINI (现) |
|------|---------------|----------------|
| **芯片** | ESP32-S3 | **ESP32S3FH4R2** |
| **Flash** | 8MB | **4MB** |
| **PSRAM** | 8MB Octal | **2MB Octal** |
| **USB 接口** | 外置 USB-UART | **内置 USB-Serial-JTAG** |
| **LED** | GPIO48 单色 (拉低亮) | **GPIO48 RGB 共阳极 (LOW=亮)** |
| **尺寸** | 54×28mm | **22.5×18mm** |

### 引脚映射

| 功能 | 引脚 | 说明 |
|------|------|------|
| BOOT 按键 (输入模式切换) | GPIO0 | 内部上拉，按下切换输入源 |
| AFE ADC 采集 | GPIO4 | ADC1_CH3 |
| RGB LED (蓝色) | GPIO48 | 共阳极，LOW=亮 |
| UART TX | GPIO1 | 串口输出 (115200) |
| UART RX | GPIO2 | 串口输入 |

### 烧录说明

**自动烧录**：连接 USB-C → `pio run -t upload`

**手动烧录**（自动失败时）：
1. 按住 **BOOT** 按钮
2. 短按 **RESET** 按钮
3. 松开 **BOOT** 按钮
4. 执行 `pio run -t upload`
5. 按 RESET 重启

---

## PC 端工具

### ecg_plotter.py

实时三通道波形显示（绿=纯净, 红=带噪, 蓝=滤波后）。

**快捷键**：
| 按键 | 功能 |
|------|------|
| ← → | 缩放时间轴 |
| ↑ ↓ | 缩放 Y 轴 |
| 1/2/3 | 切换通道可见性 |
| 空格 | 暂停/恢复 |
| R | 复位视图 |
| Q | 退出 |

---

## 手机端 App

心电监测 App (Flutter)，通过 BLE NUS 连接 ESP32-ECG 设备。

**功能**：
- BLE 扫描连接 "ESP32-ECG"
- 三通道实时波形绘制 (CustomPainter)
- 环形缓冲区 500 点 (2秒)，每 20 点刷新 UI
- R 波检测 + 心率计算 (滑动平均)
- 信息面板显示瞬时值和统计数据

**BLE 服务 UUID**：
| 服务/特征值 | UUID |
|-------------|------|
| NUS Service | `6E400001-B5A3-F393-E0A9-E50E24DCCA9E` |
| TX (Notify) | `6E400002-B5A3-F393-E0A9-E50E24DCCA9E` |
| RX (Write) | `6E400003-B5A3-F393-E0A9-E50E24DCCA9E` |

---

## 温度与功耗诊断

### 内置温度监测

系统内置芯片温度监测模块（`src/thermal/`），利用 ESP32-S3 内置温度传感器，每秒采样一次：

- 8 点滑动平均，抑制 ±1°C 噪声
- 记录 min/max/avg 温度
- 过热保护：>65°C 自动降频至 60MHz，<55°C 自动恢复 80MHz

### 已知发热分析

| 热源 | 贡献 |
|------|------|
| **BLE 协议栈** (广播/连接维持) | 主要 |
| CPU 80MHz 运行 | 次要 |
| USB LDO (5V→3.3V 转换) | 次要 |

> 💡 **改善建议**：用 3.7V 锂电池直供 3.3V 引脚（跳过板载 LDO）可显著降温。

### 诊断测试固件

`test/power_diagnostic_test.cpp` 提供 10 阶段自动扫描工具，通过温度变化率估测各子系统功耗：

| 阶段 | 测试内容 |
|------|---------|
| 0 | 空闲 40MHz (无 BLE) → 基线 |
| 1-3 | CPU 80/160/240MHz |
| 4-6 | BLE 广播/低/高功率 |
| 7 | BLE 连续 Notify |
| 8 | LED 全亮 |
| 9 | 恢复空闲 |

---

## 核心技术要点

### 数字滤波器 (`src/filter/filter.cpp`)

三级级联 IIR Biquad (直接 II 型转置结构)：
- **第1级**：二阶 Butterworth 高通 0.5Hz（去基线漂移）
- **第2级**：二阶 Butterworth 低通 40Hz（去肌电干扰）
- **第3级**：二阶 50Hz 陷波器 Q=20（工频陷零）
- 所有系数在 MATLAB 中设计，硬编码避免运行时计算

### 心电信号模拟 (`src/signal_generator/ecg_simulator.cpp`)

- P-QRS-T 波用 5 个高斯函数叠加模拟，周期 200 点 @250Hz = 75bpm
- 直流偏置 1.65V，心电峰峰值 ~1.3V (0.95~2.28V)
- 7 种噪声按 IEC60601-2-51 场景三分布

### 心率检测 (`src/heartrate/heartrate.cpp`)

- 简化 Pan-Tompkins QRS 检测算法
- 固定阈值 0.3V + 最小间隔 200ms (50点)
- 心率滑动平均 α=0.7
- 信号质量指数 (SQI) + 运动检测

---

## 核心配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 250 Hz | 每样本间隔 4ms |
| 心拍周期 | 200 点 | 对应 75bpm |
| CPU 频率 | 80 MHz | 省电模式 (240MHz→80MHz) |
| BLE TX 功率 | +9 dBm | 原始最高功率 |
| 串口波特率 | 115200 | 数据输出频率 25Hz |
| BLE 设备名 | ESP32-ECG | - |
| 滤波器类型 | IIR Biquad 级联 | HP 0.5→LP 40→Notch 50 |
| R波阈值 | 0.3 V | 滤波后信号 |
| 缓冲区大小 | 500 点 | 2秒数据 |
| 温度过热阈值 | 65°C | 自动降频至 60MHz |
| 温度恢复阈值 | 55°C | 自动恢复 80MHz |