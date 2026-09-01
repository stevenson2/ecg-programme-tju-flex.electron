# ADS1292R 固件接入备忘（替代 AD8232）

> 更新时间：2026-08-26
> 对应源码：`include/ads1292r/`, `src/ads1292r/`, `include/respiration/`, `src/respiration/`

## 1. ESP32-S3 引脚接口

| ADS1292R 信号 | ESP32-S3 GPIO | 方向 |
|---|---|---|
| START | IO8 | 输出 |
| PWDN/RESET# | IO9 | 输出（低有效） |
| DRDY# | IO14 | 输入 |
| MISO | IO13 | 输入（SPI） |
| SCLK | IO12 | 输出（SPI） |
| MOSI | IO11 | 输出（SPI） |
| CS# | IO10 | 输出（SPI） |

引脚定义在 `include/ads1292r/ads1292r.h`，同时保留 `src/main.cpp` 顶层别名。

## 2. 通道分配

- **CH1**：呼吸阻抗解调通道（ADS1292R 开启呼吸后不能用 CH1 采 ECG）
- **CH2**：ECG 通道
- 采样率：500 SPS
- SPI：CPOL=0, CPHA=1（Mode 1），1 MHz

## 3. 主要寄存器配置

| 寄存器 | 值 | 说明 |
|---|---|---|
| CONFIG1 | 0x02 | 500 SPS，连续转换 |
| CONFIG2 | 0xE0 | 内部基准 + DC 导联脱落比较器 |
| LOFF | 0x10 | 95%/5% 导联脱落阈值，DC 模式 |
| CH1SET | 0x40 | CH1 PGA=4，呼吸通道 |
| CH2SET | 0x00 | CH2 PGA=6，ECG 通道 |
| RLD_SENS | 0x2C | 使能 RLD，取 CH2 做反馈 |
| RESP1 | 0xEA | 呼吸调制/解调开启，32 kHz，112.5° 相位 |
| RESP2 | 0x02 | 32 kHz，内部 RLDREF |
| LOFF_SENS | 0x0F | 四个电极 DC 导联脱落检测 |

## 4. 输出格式扩展

原有 9 列后追加 3 列，旧 App / PC 绘图仪兼容（读取前 9 列）：

```
clean,noisy,filtered,bpm,true_bpm,sqi,motion,abnormal,confidence,resp,resp_bpm,lead_off,resp_cancel
```

- `resp`：呼吸阻抗解调电压（V）
- `resp_bpm`：呼吸率（次/分）
- `lead_off`：导联脱落掩码 bit0=IN1P, bit1=IN1N, bit2=IN2P, bit3=IN2N, bit4=RLD
- `resp_cancel`：本次从 ECG 中减去的呼吸干扰估计分量（V），用于验证呼吸抑制效果

串口每秒还有 `[呼吸]` 状态行输出。

## 5. AD8232 / ADS1292R 双输入模式切换

固件支持在 AD8232（ESP32 ADC）和 ADS1292R（SPI）两种真实 AFE 之间运行时切换：

- 串口或 BLE 命令：
  - `AFE AD8232`：切换为 AD8232 模拟输入；
  - `AFE ADS1292R`：切换为 ADS1292R SPI + 呼吸阻抗；
  - `AFE?`：查询当前 AFE 类型。
- 默认类型由 `src/main.cpp` 中 `DEFAULT_AFE_TYPE` 宏决定。
- 切换只改变真实模式的采样来源，BLE、WiFi、AI 推理、录制和输出格式不受影响。
