#include <Arduino.h>
#include <esp_adc_cal.h>
#include "adc_afe/afe_hal.h"

/**
 * @file afe_hal.cpp
 * @brief 模拟前端硬件抽象层 - 实现
 *
 * ==== ADC 采集链路 ====
 * GPIO 模拟电压 (0~3.3V)
 *   → ADC 12-bit 转换 (0~4095)
 *   → 过采样 N 次平均 (降噪)
 *   → eFuse 校准补偿 (线性校正)
 *   → 浮点电压值 (V)
 *   → 可选: 减去 dcBias 输出 ECG 信号
 *
 * ==== 过采样原理 ====
 * ESP32-S3 ADC 单次噪声约 ±2~3 LSB,
 * 8 次过采样可降低噪声至 ±1 LSB (理论 sqrt(N) 改善)。
 * 代价: 每次 readSample 耗时增加 N×ADC 转换时间 (~15μs 每次)。
 * 8 次 ≈ 120μs, 远小于 4ms 采样间隔, 完全可接受。
 *
 * ==== 校准原理 ====
 * ESP32-S3 在出厂时会在 eFuse 中存储 ADC 校准值。
 * esp_adc_cal 库利用这些值做两点/多点校正,
 * 改善 ADC 线性度和全温漂。
 */

/* ======================== 静态配置与状态 ======================== */
static AFE_HAL_Config            s_config;           /* 当前配置副本 */
static int                       s_adcChannel = -1;  /* ADC1 通道号 (-1=未初始化) */
static uint16_t                  s_rawCode = 0;      /* 最近原始码值 */
static uint64_t                  s_acc = 0;          /* 过采样累加器 */
static AFE_HAL_Status            s_status = AFE_HAL_OK;

/* ADC 校准特性 (初始化时缓存, 避免每次采样重新计算) */
static esp_adc_cal_characteristics_t s_calChars;
static bool                          s_calValid = false;

/* ======================== 工具: 将 GPIO 转为 ADC1 通道号 ======================== */

/**
 * @brief 将 GPIO 引脚号转为 ADC1 通道编号
 *
 * ESP32-S3 ADC1 通道映射:
 *   GPIO1  → ADC1_CH0
 *   GPIO2  → ADC1_CH1
 *   GPIO3  → ADC1_CH2
 *   GPIO4  → ADC1_CH3
 *   GPIO5  → ADC1_CH4
 *   GPIO6  → ADC1_CH5
 *   GPIO7  → ADC1_CH6
 *   GPIO8  → ADC1_CH7
 *   GPIO9  → ADC1_CH8
 *   GPIO10 → ADC1_CH9
 *
 * @param gpio  GPIO 引脚号
 * @return int  ADC1 通道号, -1 表示无效引脚
 */
static int gpioToAdc1Channel(uint8_t gpio)
{
    if (gpio >= GPIO_NUM_1 && gpio <= GPIO_NUM_10) {
        return (int)(gpio - GPIO_NUM_1);
    }
    return -1;
}

/* ======================== 公共 API 实现 ======================== */

void afeHalInit(const AFE_HAL_Config *config)
{
    if (config != NULL) {
        s_config = *config;
    } else {
        /* 使用默认配置 */
        AFE_HAL_Config defaultCfg = AFE_HAL_DEFAULT_3V3;
        s_config = defaultCfg;
    }

    /* 参数合法性检查 */
    if (s_config.oversample == 0) s_config.oversample = 1;
    if (s_config.oversample > 16) s_config.oversample = 16;
    if (s_config.vRef <= 0.0f)    s_config.vRef = 3.3f;

    /* 转换 GPIO → ADC1 通道 */
    s_adcChannel = gpioToAdc1Channel(s_config.adcPin);
    if (s_adcChannel < 0) {
        Serial.print("[AFE] 错误: 不支持的 ADC 引脚 GPIO");
        Serial.println(s_config.adcPin);
        Serial.println("[AFE] 请使用 GPIO1~GPIO10 (ADC1)");
        return;
    }

    /* 配置 ADC 引脚: 11dB 衰减 = 0~3.3V 量程 */
    analogReadResolution(12);                    /* 12-bit 分辨率 */
    pinMode(s_config.adcPin, INPUT);             /* 纯输入模式 */
    analogSetPinAttenuation(s_config.adcPin, ADC_0db);   /* 先设 0dB */
    analogSetPinAttenuation(s_config.adcPin, ADC_11db);  /* 再设 11dB → 实际=ADC_ATTEN_DB_12 (0~3.3V) */

    /* 初始化并缓存 ADC 校准特性 (仅一次) */
    s_calValid = false;
    if (s_config.enableCal) {
        esp_adc_cal_value_t valType = esp_adc_cal_characterize(
            ADC_UNIT_1, ADC_ATTEN_DB_12, ADC_WIDTH_BIT_12, 1100, &s_calChars);
        s_calValid = true;
        switch (valType) {
            case ESP_ADC_CAL_VAL_EFUSE_VREF:
                Serial.println("[AFE] ADC 校准: eFuse Vref 已启用");
                break;
            case ESP_ADC_CAL_VAL_EFUSE_TP:
                Serial.println("[AFE] ADC 校准: eFuse 两点校准已启用");
                break;
            case ESP_ADC_CAL_VAL_DEFAULT_VREF:
                Serial.println("[AFE] ADC 校准: 使用默认 Vref (1100mV)");
                break;
            default:
                Serial.println("[AFE] ADC 校准: 未启用");
                s_calValid = false;
                break;
        }
    }

    /* 复位状态 */
    afeHalReset();

    /* 输出配置信息 */
    Serial.print("[AFE] 初始化完成 | GPIO:");
    Serial.print(s_config.adcPin);
    Serial.print(" | 通道: ADC1_CH");
    Serial.print(s_adcChannel);
    Serial.print(" | DC偏置: ");
    Serial.print(s_config.dcBias, 3);
    Serial.print("V | VRef: ");
    Serial.print(s_config.vRef, 3);
    Serial.print("V | 过采样: ");
    Serial.print(s_config.oversample);
    Serial.println("x");
}

float afeHalReadSample(void)
{
    if (s_adcChannel < 0) {
        return 0.0f;  /* 未初始化, 返回 0 */
    }

    /* ---- 过采样累加 ---- */
    s_acc = 0;
    for (uint8_t i = 0; i < s_config.oversample; i++) {
        s_acc += analogRead(s_config.adcPin);
    }

    /* 计算平均原始码值 (4 舍 5 入) */
    uint32_t avgRaw = 0;
    if (s_config.oversample > 1) {
        avgRaw = (uint32_t)((s_acc + (s_config.oversample >> 1)) / s_config.oversample);
    } else {
        avgRaw = (uint32_t)s_acc;
    }

    s_rawCode = (uint16_t)(avgRaw & 0x0FFF);  /* 12-bit 截断 */

    /* ---- 电压换算 ---- */
    float voltage;

    if (s_calValid) {
        /* 使用 eFuse 校准 (缓存特性, 高效) */
        uint32_t calVoltage_mV = esp_adc_cal_raw_to_voltage(avgRaw, &s_calChars);
        voltage = (float)calVoltage_mV / 1000.0f;
    } else {
        /* 简单线性换算: V = raw * VRef / 4095 */
        voltage = (float)avgRaw * s_config.vRef / 4095.0f;
    }

    /* ---- 削顶检测 ---- */
    if (s_rawCode <= 2 || s_rawCode >= 4093) {
        s_status = AFE_HAL_SATURATED;
    } else if (s_rawCode <= 10 || s_rawCode >= 4085) {
        s_status = AFE_HAL_CLIPPING;
    } else {
        s_status = AFE_HAL_OK;
    }

    return voltage;
}

float afeHalReadECG(void)
{
    float sample = afeHalReadSample();
    return sample - s_config.dcBias;
}

uint16_t afeHalGetRawCode(void)
{
    return s_rawCode;
}

AFE_HAL_Status afeHalGetStatus(void)
{
    return s_status;
}

bool afeHalIsClipping(void)
{
    return (s_status == AFE_HAL_CLIPPING || s_status == AFE_HAL_SATURATED);
}

void afeHalReset(void)
{
    s_acc = 0;
    s_rawCode = 0;
    s_status = AFE_HAL_OK;
}
