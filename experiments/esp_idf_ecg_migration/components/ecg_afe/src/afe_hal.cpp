/** AFE ADC HAL. All calls belong to the sample-loop owner task. */
#include "adc_afe/afe_hal.h"
#include <math.h>
#include <stdio.h>
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "driver/gpio.h"

static AFE_HAL_Config s_config;
static adc_oneshot_unit_handle_t s_adc = NULL;
static adc_cali_handle_t s_cali = NULL;
static adc_channel_t s_channel = ADC_CHANNEL_0;
static uint16_t s_rawCode = 0;
static bool s_readAttempted = false;
static AFE_HAL_Diagnostics s_diag = {
    false, false, false, ESP_ERR_INVALID_STATE, ESP_ERR_INVALID_STATE, 0, 0
};

static void log_error(const char *operation, esp_err_t err) {
    printf("[AFE] %s failed: %s (0x%x)\n", operation,
           esp_err_to_name(err), (unsigned)err);
}
static esp_err_t init_failed(const char *operation, esp_err_t err) {
    s_diag.init_error = err;
    log_error(operation, err);
    return err;
}

esp_err_t afeHalInit(const AFE_HAL_Config *config) {
    AFE_HAL_Config next = AFE_HAL_DEFAULT_3V3;
    if (config) next = *config;
    if (next.oversample == 0) next.oversample = 1;
    if (next.oversample > 16) next.oversample = 16;
    if (!isfinite(next.vRef) || next.vRef <= 0) next.vRef = 3.3f;
    if (next.adcPin < 1 || next.adcPin > 10 || !isfinite(next.dcBias)) {
        return init_failed("config", ESP_ERR_INVALID_ARG);
    }
    if (s_diag.ready) {
        if (next.adcPin != s_config.adcPin || next.enableCal != s_config.enableCal) {
            return init_failed("live GPIO/calibration change", ESP_ERR_INVALID_STATE);
        }
        // Repeated MODE afe reuses ADC1 and preserves cumulative counters.
        s_config = next;
        s_diag.init_error = ESP_OK;
        printf("[AFE] init reused GPIO=%u oversample=%u\n",
               (unsigned)next.adcPin, (unsigned)next.oversample);
        return ESP_OK;
    }
    adc_oneshot_unit_handle_t unit = NULL;
    adc_oneshot_unit_init_cfg_t unit_cfg = {};
    unit_cfg.unit_id = ADC_UNIT_1;
    esp_err_t err = adc_oneshot_new_unit(&unit_cfg, &unit);
    if (err != ESP_OK) return init_failed("adc_oneshot_new_unit", err);
    adc_channel_t channel = (adc_channel_t)(next.adcPin - 1);
    adc_oneshot_chan_cfg_t chan_cfg = {
        .atten = ADC_ATTEN_DB_12,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    err = adc_oneshot_config_channel(unit, channel, &chan_cfg);
    if (err != ESP_OK) {
        esp_err_t cleanup = adc_oneshot_del_unit(unit);
        if (cleanup != ESP_OK) log_error("adc_oneshot_del_unit", cleanup);
        return init_failed("adc_oneshot_config_channel", err);
    }
    adc_cali_handle_t cali = NULL;
    if (next.enableCal) {
        adc_cali_curve_fitting_config_t cal_cfg = {
            .unit_id = ADC_UNIT_1,
            .chan = channel,
            .atten = ADC_ATTEN_DB_12,
            .bitwidth = ADC_BITWIDTH_DEFAULT,
        };
        err = adc_cali_create_scheme_curve_fitting(&cal_cfg, &cali);
        if (err != ESP_OK) {
            // Calibration remains optional; expose the existing vRef fallback.
            cali = NULL;
            log_error("adc calibration (using vRef fallback)", err);
        }
    }
    s_adc = unit;
    s_cali = cali;
    s_channel = channel;
    s_config = next;
    s_diag.ready = true;
    s_diag.sample_valid = false;
    s_readAttempted = false;
    s_diag.calibrated = cali != NULL;
    s_diag.init_error = ESP_OK;
    s_diag.read_error = ESP_ERR_INVALID_STATE; // no sample yet
    printf("[AFE] init OK GPIO=%u channel=%d oversample=%u calibrated=%u\n",
           (unsigned)next.adcPin, (int)channel, (unsigned)next.oversample,
           (unsigned)s_diag.calibrated);
    return ESP_OK;
}

static esp_err_t read_failed(esp_err_t err) {
    // Retain every failure count without flooding the 500Hz loop with logs.
    if (err != s_diag.read_error || s_diag.sample_valid) log_error("sample", err);
    s_diag.sample_valid = false;
    s_readAttempted = true;
    s_diag.read_error = err;
    s_diag.samples_failed++;
    return err;
}
esp_err_t afeHalReadSampleChecked(float *voltage) {
    if (!voltage) return read_failed(ESP_ERR_INVALID_ARG);
    if (!s_diag.ready) return read_failed(ESP_ERR_INVALID_STATE);
    uint32_t acc = 0;
    for (uint8_t i = 0; i < s_config.oversample; i++) {
        int raw = 0;
        esp_err_t err = adc_oneshot_read(s_adc, s_channel, &raw);
        if (err != ESP_OK) return read_failed(err);
        if (raw < 0 || raw > 4095) return read_failed(ESP_ERR_INVALID_RESPONSE);
        acc += (uint32_t)raw;
    }
    uint16_t raw = (uint16_t)((acc + (s_config.oversample >> 1)) / s_config.oversample);
    float result;
    if (s_cali) {
        int mv = 0;
        esp_err_t err = adc_cali_raw_to_voltage(s_cali, raw, &mv);
        if (err != ESP_OK) return read_failed(err);
        result = (float)mv / 1000.0f;
    } else {
        result = (float)raw * s_config.vRef / 4095.0f;
    }
    s_rawCode = raw;
    s_diag.sample_valid = true;
    s_diag.read_error = ESP_OK;
    s_diag.samples_ok++;
    *voltage = result;
    return ESP_OK;
}
float afeHalReadSample(void) {
    float result = NAN;
    afeHalReadSampleChecked(&result);
    return result;
}
float afeHalReadECG(void) { return afeHalReadSample() - s_config.dcBias; }
void afeHalSetOversample(uint8_t n) {
    s_config.oversample = n == 0 ? 1 : (n > 16 ? 16 : n);
}
uint8_t afeHalGetOversample(void) { return s_config.oversample; }
uint16_t afeHalGetRawCode(void) { return s_rawCode; }
void afeHalGetDiagnostics(AFE_HAL_Diagnostics *out) { if (out) *out = s_diag; }
AFE_HAL_Status afeHalGetStatus(void) {
    if (!s_diag.ready) return AFE_HAL_NOT_READY;
    if (!s_diag.sample_valid) {
        return s_readAttempted ? AFE_HAL_READ_ERROR : AFE_HAL_NOT_READY;
    }
    if (s_rawCode <= 2 || s_rawCode >= 4093) return AFE_HAL_SATURATED;
    if (s_rawCode <= 10 || s_rawCode >= 4085) return AFE_HAL_CLIPPING;
    return AFE_HAL_OK;
}
bool afeHalIsClipping(void) {
    AFE_HAL_Status status = afeHalGetStatus();
    return status == AFE_HAL_CLIPPING || status == AFE_HAL_SATURATED;
}
void afeHalReset(void) {
    // Reset sample validity, not the ADC unit or cumulative error evidence.
    s_rawCode = 0;
    s_diag.sample_valid = false;
    s_readAttempted = false;
    s_diag.read_error = ESP_ERR_INVALID_STATE;
}
