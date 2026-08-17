#ifndef AFE_HAL_H
#define AFE_HAL_H

/**
 * @file afe_hal.h
 * @brief 模拟前端硬件抽象层 (AFE HAL)
 *
 * ========== 设计思想 ==========
 * 本层完全与具体电路解耦，只抽象出 ADC 采样所需的三个物理参数：
 *
 *   ① ADC 引脚号 (GPIO)
 *   ② 直流偏置电压 (DC Bias, V)
 *   ③ ADC 参考电压 (VRef, 通常是 3.3V)
 *
 * 无论你的 PCB 用的是 AD8232、ADS1292R、分立运放还是自制仪表放大器，
 * 只要输出是模拟电压 → ESP32 ADC，这个接口就能适配。
 *
 * ========== 你的 PCB 接线指引 ==========
 * ┌─────────────────────────────────────┐
 * │  你的 PCB (自研模拟前端)             │
 * │                                     │
 * │  VCC  →  3.3V                      │
 * │  GND  →  GND                       │
 * │  OUT  →  GPIO___ (ADC输入)          │
 * │          ↑ 填写实际使用的引脚       │
 * │                                     │
 * │  如果 PCB 有导联脱落检测输出:        │
 * │  LOD  →  GPIO___ (数字输入)         │
 * └─────────────────────────────────────┘
 *
 * ========== 与 ecg_simulator 的关系 ==========
 * 两者实现完全相同的函数签名:
 *   init() , readSample() , readECG() , reset()
 *
 * main.cpp 通过编译宏切换数据源:
 *   #define AFE_SOURCE_REAL  1  → 本模块 (真实 ADC)
 *   #define AFE_SOURCE_REAL  0  → ecg_simulator (软件模拟)
 *
 * 切换时 main.cpp 一行代码不改, 只改 platformio.ini 的 build_flags。
 *
 * ========== ADC 硬件配置 ==========
 * ESP32-S3  ADC1 特性:
 *   - 分辨率: 12-bit (0 ~ 4095)
 *   - 衰减:   11dB (输入范围 0 ~ 3.3V)
 *   - 建议引脚: GPIO1~10 (ADC1), 避免 GPIO11~20 (ADC2, 与 WiFi/BLE 冲突)
 *   - 噪声:   典型 ±2~3 LSB, 建议过采样降噪
 */

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ======================== 配置结构体 ======================== */

/**
 * @brief AFE 硬件参数配置
 *
 * 你需要根据自己 PCB 的实际电路填写这三个参数。
 * 提供 AFE_HAL_DEFAULT_3V3 宏作为快速起点。
 */
typedef struct {
    uint8_t adcPin;         /**< ADC 输入引脚号 (GPIO_NUM_x, 如 GPIO_NUM_4) */
    float   dcBias;         /**< 电路输出的直流偏置 (V), 测一下中点电压填这里 */
    float   vRef;           /**< ADC 参考电压 (V), ESP32-S3 通常是 3.3V */
    uint8_t oversample;     /**< 过采样次数: 1/4/8/16, 推荐 8 或 16 */
    bool    enableCal;      /**< 启用 eFuse ADC 校准补偿 (推荐 true) */
} AFE_HAL_Config;

/**
 * @brief 默认配置 (3.3V 系统, GPIO4)
 *
 * 如果你的 PCB 供电是 3.3V, 偏置是 VCC/2 = 1.65V,
 * 可以直接用这个宏, 只改 adcPin。
 *
 * 用法:
 *   AFE_HAL_Config cfg = AFE_HAL_DEFAULT_3V3;
 *   cfg.adcPin = GPIO_NUM_5;  // 改为你实际用的引脚
 *   afeHalInit(&cfg);
 */
#define AFE_HAL_DEFAULT_3V3 { \
    .adcPin     = GPIO_NUM_4,  \
    .dcBias     = 1.65f,       \
    .vRef       = 3.3f,        \
    .oversample = 8,           \
    .enableCal  = true         \
}

/**
 * @brief 5V 系统默认配置 (偏置 VCC/2 = 2.5V)
 *
 * 如果 PCB 用 5V 供电、运放输出摆幅到 0~3.3V (被 ESP32 ADC 限制),
 * 偏置为 2.5V, 但 vRef 仍为 3.3V (ESP32 引脚耐压)。
 */
#define AFE_HAL_DEFAULT_5V { \
    .adcPin     = GPIO_NUM_4,  \
    .dcBias     = 2.50f,       \
    .vRef       = 3.3f,        \
    .oversample = 8,           \
    .enableCal  = true         \
}

/* ======================== 状态枚举 ======================== */

typedef enum {
    AFE_HAL_OK        = 0,  /**< 正常 */
    AFE_HAL_CLIPPING  = 1,  /**< 信号削顶 (接近 0V 或 VRef) */
    AFE_HAL_SATURATED = 2   /**< ADC 饱和 (读数=0 或 4095) */
} AFE_HAL_Status;

/* ======================== API ======================== */

/**
 * @brief 初始化 AFE HAL 模块
 *
 * 执行:
 *   1. 配置 ADC 引脚 (11dB 衰减, 0~3.3V 量程)
 *   2. 读取 eFuse 校准值 (如果启用)
 *   3. 初始化过采样状态
 *
 * @param config  指向 AFE_HAL_Config 的指针
 *                传入 NULL 则使用 AFE_HAL_DEFAULT_3V3
 */
void afeHalInit(const AFE_HAL_Config *config);

/**
 * @brief 采集一个原始样本 (含 DC 偏置)
 *
 * 完整链路:
 *   GPIO 模拟电压 → ADC 12-bit → 过采样平均
 *   → eFuse 校准补偿 → 浮点电压值 (V)
 *
 * 返回值包含 dcBias, 例如:
 *   - 信号在偏置处: 返回 1.65V
 *   - 信号向上摆:  返回 1.85V (偏置 + 0.2V)
 *   - 信号向下摆:  返回 1.45V (偏置 - 0.2V)
 *
 * 行为与 ecg_simulator 的 generateECGSample() 一致。
 *
 * @return float 当前采样电压 (V)
 */
float afeHalReadSample(void);

/** @brief 运行时修改过采样次数 (1-16, 越界自动钳位) — DIAG OVS 用 */
void afeHalSetOversample(uint8_t oversample);

/** @brief 当前过采样次数 — DIAG 状态输出用 */
uint8_t afeHalGetOversample(void);

/**
 * @brief 采集一个心电信号 (已去除 DC 偏置)
 *
 * 等效于 afeHalReadSample() - dcBias
 * 返回以 0V 为中心的纯心电信号, 与 filtered 通道同基准。
 *
 * 行为与 ecg_simulator 的 getCleanECGValue() 一致。
 *
 * @return float 心电信号 (V, -1.2V ~ +1.2V 典型)
 */
float afeHalReadECG(void);

/**
 * @brief 获取最近一次原始 ADC 码值 (调试用)
 * @return uint16_t  0~4095
 */
uint16_t afeHalGetRawCode(void);

/**
 * @brief 获取当前 AFE 状态
 * @return AFE_HAL_Status
 */
AFE_HAL_Status afeHalGetStatus(void);

/**
 * @brief 检查信号是否削顶
 * @return true  信号触及电源轨, 需要调整增益
 */
bool afeHalIsClipping(void);

/**
 * @brief 复位 AFE HAL 模块
 *
 * 清除过采样累加器和状态,
 * 不重新配置 GPIO。
 */
void afeHalReset(void);

#ifdef __cplusplus
}
#endif

#endif /* AFE_HAL_H */
