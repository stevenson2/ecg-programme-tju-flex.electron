# ESP32-S3-SUPERMINI 开发板说明

## 硬件规格对比

| 参数 | ESP32-S3-DevKitM-1 (原) | ESP32-S3-SUPERMINI (现) |
|------|-------------------------|------------------------|
| **芯片** | ESP32-S3 (Xtensa LX7) | ESP32S3FH4R2 (Xtensa LX7) |
| **Flash** | 8MB | **4MB** |
| **PSRAM** | 8MB Octal | **2MB Octal** |
| **USB 接口** | 外置 USB-UART (CP2102) | **内置 USB-Serial-JTAG** |
| **LED** | GPIO48 单色 (拉低亮) | **GPIO48 RGB 共阳极 (LOW=亮)** |
| **尺寸** | 54x28mm | **22.5x18mm (拇指大小)** |

## Pin 映射

### 当前使用的 GPIO

| 功能 | 引脚 | SUPERMINI 对应丝印 | 说明 |
|------|------|-------------------|------|
| BOOT 按键 (输入模式切换) | GPIO0 | `IO0` / `0` | 内部上拉，按下切换输入源 |
| AFE ADC 采集 | GPIO4 | `IO4` / `4` | ADC1_CH3 |
| RGB LED (蓝色) | GPIO48 | (板载) | 共阳极，LOW=亮 |

### SUPERMINI 可用 GPIO 列表

| 引脚 | 功能 |
|------|------|
| GPIO0 | BOOT 按键 (内部上拉) |
| GPIO1 | UART0 TX (串口输出) |
| GPIO2 | UART0 RX |
| GPIO3 | IO |
| GPIO4 | ADC1_CH3 (当前用于 AFE) |
| GPIO5 | IO |
| GPIO6 | IO |
| GPIO7 | IO |
| GPIO8 | RGB LED (可能共享) |
| GPIO9 | IO |
| GPIO40 | IO |
| GPIO41 | IO |
| GPIO42 | IO |
| GPIO45 | IO |
| GPIO46 | IO |
| GPIO48 | RGB LED (板载) |

## 烧录说明

### 自动烧录 (推荐)

1. 将 SUPERMINI 通过 USB-C 连接电脑
2. PlatformIO 会自动识别 USB-Serial-JTAG 设备
3. 执行 `pio run -t upload` 烧录

### 手动烧录 (遇到问题时的备用方案)

如果自动烧录失败：

1. 按住板上的 **BOOT (GPIO0)** 按钮
2. 短按 **RESET (EN)** 按钮
3. 松开 **BOOT** 按钮
4. 执行 `pio run -t upload`
5. 烧录完成后按 RESET 重启

### 串口监视器

```bash
pio device monitor
```

或指定端口:
```bash
pio device monitor -p <COM_PORT>
```

## 常见问题

### Q: 串口无法连接
A: SUPERMINI 使用内置 USB-Serial-JTAG，电脑可能需要安装驱动。Windows 通常自动识别为 "ESP32-S3" 或 "USB JTAG/serial"。如果无法识别，尝试按住 BOOT 重新插拔 USB。

### Q: 烧录失败
A: 尝试降低烧录速度或手动进入下载模式（按住 BOOT → 按 RESET → 松开 BOOT）。

### Q: LED 不亮
A: SUPERMINI 的 RGB LED 是共阳极设计，LOW 电平点亮。代码已通过 `LED_ACTIVE_LEVEL` 宏适配。如果 LED 行为相反，将宏值从 `LOW` 改为 `HIGH`。

### Q: PSRAM 未启用
A: 如果编译后运行异常，检查 `platformio.ini` 中的 `board_build.psram_type` 和 `board_build.memory_type` 配置。如使用 `adafruit_qtpy_esp32s3_n4r2` 开发板定义，PSRAM 会自动配置。