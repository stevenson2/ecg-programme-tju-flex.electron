/**
 * @file afe_hal.cpp
 * @brief AFE ADC HAL（ESP-IDF esp_adc 移植版）
 */
#include "adc_afe/afe_hal.h"

#include <stdio.h>
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "driver/gpio.h"

static AFE_HAL_Config s_config;
static adc_oneshot_unit_handle_t s_adc = NULL;
static adc_cali_handle_t s_cali = NULL;
static adc_channel_t s_channel = ADC_CHANNEL_0;
static bool s_initialized = false;
static uint16_t s_rawCode = 0;
static AFE_HAL_Status s_status = AFE_HAL_OK;

static bool init_cali(adc_unit_t unit, adc_channel_t channel, adc_atten_t atten) {
    adc_cali_curve_fitting_config_t cfg = {
        .unit_id = unit,
        .chan = channel,
        .atten = atten,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    return adc_cali_create_scheme_curve_fitting(&cfg, &s_cali) == ESP_OK;
}

static adc_channel_t gpio_to_channel(uint8_t gpio) {
    if (gpio >= 1 && gpio <= 10) {
        return (adc_channel_t)(gpio - 1);
    }
    return ADC_CHANNEL_0;
}

void afeHalInit(const AFE_HAL_Config *config) {
    if (config) s_config = *config;
    else {
        AFE_HAL_Config def = AFE_HAL_DEFAULT_3V3;
        s_config = def;
    }
    if (s_config.oversample == 0) s_config.oversample = 1;
    if (s_config.oversample > 16) s_config.oversample = 16;
    if (s_config.vRef <= 0) s_config.vRef = 3.3f;

    adc_oneshot_unit_init_cfg_t unit_cfg = {.unit_id = ADC_UNIT_1};
    if (adc_oneshot_new_unit(&unit_cfg, &s_adc) != ESP_OK) {
        printf("[AFE] adc_oneshot_new_unit failed\n");
        return;
    }
    s_channel = gpio_to_channel(s_config.adcPin);
    adc_oneshot_chan_cfg_t chan_cfg = {
        .atten = ADC_ATTEN_DB_12,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    if (adc_oneshot_config_channel(s_adc, s_channel, &chan_cfg) != ESP_OK) {
        printf("[AFE] adc_oneshot_config_channel failed\n");
    }
    if (s_config.enableCal) {
        init_cali(ADC_UNIT_1, s_channel, ADC_ATTEN_DB_12);
    }
    s_initialized = true;
    printf("[AFE] init OK GPIO=%u channel=%d oversample=%u\n",
           (unsigned)s_config.adcPin, (int)s_channel, (unsigned)s_config.oversample);
}

float afeHalReadSample(void) {
    if (!s_initialized) return 0.0f;
    uint64_t acc = 0;
    for (uint8_t i = 0; i < s_config.oversample; i++) {
        int raw = 0;
        if (adc_oneshot_read(s_adc, s_channel, &raw) == ESP_OK && raw >= 0) {
            acc += (uint64_t)raw;
        }
    }
    uint32_t avg = (s_config.oversample > 1)
        ? (uint32_t)((acc + (s_config.oversample >> 1)) / s_config.oversample)
        : (uint32_t)acc;
    s_rawCode = (uint16_t)(avg & 0x0FFF);

    int voltage_mv = 0;
    if (s_cali && adc_cali_raw_to_voltage(s_cali, s_rawCode, &voltage_mv) == ESP_OK) {
        return (float)voltage_mv / 1000.0f;
    }
    return (float)s_rawCode * s_config.vRef / 4095.0f;
}

float afeHalReadECG(void) {
    return afeHalReadSample() - s_config.dcBias;
}

void afeHalSetOversample(uint8_t oversample) {
    if (oversample == 0) oversample = 1;
    if (oversample > 16) oversample = 16;
    s_config.oversample = oversample;
}

uint8_t afeHalGetOversample(void) { return s_config.oversample; }
uint16_t afeHalGetRawCode(void) { return s_rawCode; }

AFE_HAL_Status afeHalGetStatus(void) {
    if (s_rawCode <= 2 || s_rawCode >= 4093) return AFE_HAL_SATURATED;
    if (s_rawCode <= 10 || s_rawCode >= 4085) return AFE_HAL_CLIPPING;
    return AFE_HAL_OK;
}

bool afeHalIsClipping(void) { return afeHalGetStatus() >= AFE_HAL_CLIPPING; }

void afeHalReset(void) {
    s_rawCode = 0;
    s_status = AFE_HAL_OK;
}
