#include <Arduino.h>
#include <esp_adc_cal.h>
#include "adc_afe/afe_hal.h"

/**
 * @file afe_hal.cpp
 * @brief 模拟前端硬件抽象层 - 实现
 *
 * ADC采集链路:
 * GPIO 模拟电压 (0~3.3V)
 *   → ADC 12-bit 转换 (0~4095)
 *   → 过采样 N 次平均 (降噪)
 *   → eFuse 校准补偿 (线性校正)
 *   → 浮点电压值 (V)
 *   → 可选: 减去 dcBias 输出 ECG 信号
 *
 * 注意: 50Hz 工频抑制不在本层处理，而是在 filter.cpp 和 main.cpp
 * 中通过级联陷波器 + 梳状滤波器实现。
 * 梳状滤波器不能放在本层，因为 afeHalReadSample() 每帧被调用2次
 * （noisySample + cleanSample），会导致采样间隔不一致。
 */

/* ======================== 静态配置与状态 ======================== */
static AFE_HAL_Config            s_config;
static int                       s_adcChannel = -1;
static uint16_t                  s_rawCode = 0;
static uint64_t                  s_acc = 0;
static AFE_HAL_Status            s_status = AFE_HAL_OK;
static esp_adc_cal_characteristics_t s_calChars;
static bool                          s_calValid = false;

/* ======================== GPIO → ADC1 通道 ======================== */
static int gpioToAdc1Channel(uint8_t gpio)
{
    if (gpio >= GPIO_NUM_1 && gpio <= GPIO_NUM_10) {
        return (int)(gpio - GPIO_NUM_1);
    }
    return -1;
}

/* ======================== API ======================== */

void afeHalInit(const AFE_HAL_Config *config)
{
    if (config != NULL) {
        s_config = *config;
    } else {
        AFE_HAL_Config defaultCfg = AFE_HAL_DEFAULT_3V3;
        s_config = defaultCfg;
    }

    if (s_config.oversample == 0) s_config.oversample = 1;
    if (s_config.oversample > 16) s_config.oversample = 16;
    if (s_config.vRef <= 0.0f)    s_config.vRef = 3.3f;

    s_adcChannel = gpioToAdc1Channel(s_config.adcPin);
    if (s_adcChannel < 0) {
        Serial.print("[AFE] 错误: 不支持的 ADC 引脚 GPIO");
        Serial.println(s_config.adcPin);
        return;
    }

    analogReadResolution(12);
    pinMode(s_config.adcPin, INPUT);
    analogSetPinAttenuation(s_config.adcPin, ADC_11db);

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
                s_calValid = false;
                break;
        }
    }

    afeHalReset();

    Serial.print("[AFE] 初始化完成 | GPIO:");
    Serial.print(s_config.adcPin);
    Serial.print(" | 通道: ADC1_CH");
    Serial.print(s_adcChannel);
    Serial.print(" | DC偏置: ");
    Serial.print(s_config.dcBias, 3);
    Serial.print("V | 过采样: ");
    Serial.print(s_config.oversample);
    Serial.println("x");
}

float afeHalReadSample(void)
{
    if (s_adcChannel < 0) return 0.0f;

    s_acc = 0;
    for (uint8_t i = 0; i < s_config.oversample; i++) {
        s_acc += analogRead(s_config.adcPin);
    }

    uint32_t avgRaw = (s_config.oversample > 1)
        ? (uint32_t)((s_acc + (s_config.oversample >> 1)) / s_config.oversample)
        : (uint32_t)s_acc;

    s_rawCode = (uint16_t)(avgRaw & 0x0FFF);

    float voltage;
    if (s_calValid) {
        uint32_t calVoltage_mV = esp_adc_cal_raw_to_voltage(avgRaw, &s_calChars);
        voltage = (float)calVoltage_mV / 1000.0f;
    } else {
        voltage = (float)avgRaw * s_config.vRef / 4095.0f;
    }

    if (s_rawCode <= 2 || s_rawCode >= 4093)
        s_status = AFE_HAL_SATURATED;
    else if (s_rawCode <= 10 || s_rawCode >= 4085)
        s_status = AFE_HAL_CLIPPING;
    else
        s_status = AFE_HAL_OK;

    return voltage;
}

float afeHalReadECG(void)
{
    float sample = afeHalReadSample();
    return sample - s_config.dcBias;
}

uint16_t afeHalGetRawCode(void) { return s_rawCode; }
AFE_HAL_Status afeHalGetStatus(void) { return s_status; }
bool afeHalIsClipping(void) { return (s_status >= AFE_HAL_CLIPPING); }

void afeHalReset(void)
{
    s_acc = 0;
    s_rawCode = 0;
    s_status = AFE_HAL_OK;
}