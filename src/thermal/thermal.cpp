/**
 * @file thermal.cpp
 * @brief ESP32-S3 内置温度传感器监测模块 - 实现
 *
 * 使用 ESP32-S3 内部温度传感器 (temperatureRead)，
 * 每 1 秒采样一次，8 点环形缓冲滑动平均。
 *
 * 温度传感器精度: ±1°C
 * 有效范围: -40°C ~ +125°C
 * 典型室温误差: ±1°C
 */

#include <Arduino.h>
#include "thermal/thermal.h"

/* ======================== 常量 ======================== */
#define SMOOTH_WINDOW   8       /* 滑动平均窗口 */
#define WARN_THRESHOLD  55.0f   /* 高温警告阈值 (°C) */
#define CRIT_THRESHOLD  65.0f   /* 过热阈值 (°C) */

/* ======================== 全局状态 ======================== */
static float s_buffer[SMOOTH_WINDOW];   /* 环形缓冲区 */
static int   s_index     = 0;           /* 当前写入位置 */
static int   s_count     = 0;           /* 已采样次数 (<SMOOTH_WINDOW 时未填满) */

static float s_min       = 999.0f;      /* 最低温度 */
static float s_max       = -999.0f;     /* 最高温度 */
static float s_avg       = 0.0f;        /* 滑动平均 */

static ThermalAlertLevel s_alert = THERMAL_OK;

static unsigned long s_lastSampleTime = 0;  /* 上次采样时间戳 */
static unsigned long s_startTime      = 0;  /* 初始化时间戳 */

/**
 * @brief 读取 ESP32-S3 内置温度传感器
 *
 * temperatureRead() 是 ESP32-S3 Arduino 核心的内置函数，
 * 返回芯片内部温度传感器读数，单位 °C。
 *
 * @return float 温度值 (°C), 读取失败返回 -999.0f
 */
static float readInternalTemp(void)
{
#if defined(ESP_IDF_VERSION_MAJOR) && ESP_IDF_VERSION_MAJOR >= 5
    /* ESP-IDF v5+ / Arduino 3.x 使用新 API */
    float temp = temperatureRead();
    return temp;
#else
    /* 旧版兼容 */
    float temp = temperatureRead();
    return temp;
#endif
}

/* ======================== 对外接口 ======================== */

void thermalInit(void)
{
    s_index = 0;
    s_count = 0;
    s_min   = 999.0f;
    s_max   = -999.0f;
    s_avg   = 0.0f;
    s_alert = THERMAL_OK;

    s_startTime = millis();
    s_lastSampleTime = 0;

    /* 用首次采样填充缓冲区 */
    float firstVal = readInternalTemp();
    for (int i = 0; i < SMOOTH_WINDOW; i++) {
        s_buffer[i] = firstVal;
    }
    s_count = 1;            /* 表示已有一个有效样本 */
    s_min   = firstVal;
    s_max   = firstVal;
    s_avg   = firstVal;

    Serial.print("[温度] 传感器已初始化, 初始温度: ");
    Serial.print(firstVal, 1);
    Serial.println(" °C");
}

ThermalState thermalUpdate(void)
{
    unsigned long now = millis();

    /* 强制首次读取 (即使 < 1s) */
    if (s_lastSampleTime == 0) {
        s_lastSampleTime = now;
    }

    /* 限制采样周期 ≥ 900ms, 避免调用过于频繁 */
    if (now - s_lastSampleTime < 900) {
        /* 返回上次状态 */
        ThermalState state;
        state.current     = s_buffer[s_index > 0 ? s_index - 1 : SMOOTH_WINDOW - 1];
        state.avg         = s_avg;
        state.min         = s_min;
        state.max         = s_max;
        state.alertLevel  = s_alert;
        state.uptimeMs    = now - s_startTime;
        return state;
    }

    s_lastSampleTime = now;

    /* ==== 读取温度 ==== */
    float raw = readInternalTemp();

    /* ==== 更新环形缓冲区 ==== */
    s_buffer[s_index] = raw;
    s_index = (s_index + 1) % SMOOTH_WINDOW;
    if (s_count < SMOOTH_WINDOW) {
        s_count++;
    }

    /* ==== 计算滑动平均 ==== */
    float sum = 0.0f;
    for (int i = 0; i < s_count; i++) {
        sum += s_buffer[i];
    }
    s_avg = sum / (float)s_count;

    /* ==== 更新最值 ==== */
    if (raw < s_min) s_min = raw;
    if (raw > s_max) s_max = raw;

    /* ==== 判断告警级别 ==== */
    if (s_avg > CRIT_THRESHOLD) {
        s_alert = THERMAL_CRITICAL;
    } else if (s_avg > WARN_THRESHOLD) {
        s_alert = THERMAL_WARN;
    } else {
        s_alert = THERMAL_OK;
    }

    /* ==== 构造状态 ==== */
    ThermalState state;
    state.current     = raw;
    state.avg         = s_avg;
    state.min         = s_min;
    state.max         = s_max;
    state.alertLevel  = s_alert;
    state.uptimeMs    = now - s_startTime;

    return state;
}

const char* thermalGetAlertString(void)
{
    switch (s_alert) {
        case THERMAL_WARN:
            return "⚠ 高温";
        case THERMAL_CRITICAL:
            return "🔥 过热!";
        case THERMAL_OK:
        default:
            return "正常";
    }
}

void thermalPrintStatus(void)
{
    ThermalState st;
    st.current     = s_buffer[s_index > 0 ? s_index - 1 : SMOOTH_WINDOW - 1];
    st.avg         = s_avg;
    st.min         = s_min;
    st.max         = s_max;
    st.alertLevel  = s_alert;
    st.uptimeMs    = millis() - s_startTime;

    Serial.print("[温度] 当前: ");
    Serial.print(st.current, 1);
    Serial.print("°C | 平均: ");
    Serial.print(st.avg, 1);
    Serial.print("°C | 范围: ");
    Serial.print(st.min, 1);
    Serial.print("~");
    Serial.print(st.max, 1);
    Serial.print("°C | 状态: ");
    Serial.println(thermalGetAlertString());

    if (st.alertLevel >= THERMAL_WARN) {
        Serial.println("[温度] ⚠ 请检查散热或降低功耗!");
    }
}