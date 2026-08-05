/**
 * @file heartrate_ludb_test.cpp
 * @brief 心率检测模块自测 (v4.2 参数验证)
 *
 * 在 ESP32-S3 板上运行, 用内置 ECG 模拟器 + 固件完整链路
 * (50Hz梳状 → HP/LP → hrProcess) 验证 v4.2 参数下心率检测精度。
 *
 * LUDB 金标准 (PC 端, pc_tools/ecg_dl/verify_heartrate_ludb.py):
 *   Se=72.9% PPV=82.6% F1=0.774 BPM_MAE=3.19 (±3BPM 73.5%)
 *
 * 编译运行 (替换 main.cpp 或用 test env):
 *   pio run -t upload
 *   串口: 115200, 输出 30 秒内的检测统计
 *
 * 说明: 板上仅能验证"合成信号"场景 (模拟器 QRS 单峰),
 *       真实心电多峰/噪声场景请用 PC 端 LUDB 验证脚本。
 */

#include <Arduino.h>
#include "heartrate/heartrate.h"
#include "filter/filter.h"
#include "signal_generator/ecg_simulator.h"

/* ======================== 测试配置 ======================== */
#define TEST_DURATION_S    30      /* 测试时长 (秒) */
#define COMB_TAPS          10      /* 50Hz 梳状滤波器抽头 (与 main.cpp 一致) */

/* ======================== 50Hz 梳状 (复现 main.cpp) ======================== */
static float s_combBuf1[COMB_TAPS] = {0};
static int   s_combIdx1 = 0;
static float s_combSum1 = 0.0f;
static float s_combBuf2[COMB_TAPS] = {0};
static int   s_combIdx2 = 0;
static float s_combSum2 = 0.0f;

static inline float applyCombFilter(float x)
{
    s_combSum1 -= s_combBuf1[s_combIdx1];
    s_combBuf1[s_combIdx1] = x;
    s_combSum1 += x;
    s_combIdx1 = (s_combIdx1 + 1) % COMB_TAPS;
    float y1 = s_combSum1 / (float)COMB_TAPS;

    s_combSum2 -= s_combBuf2[s_combIdx2];
    s_combBuf2[s_combIdx2] = y1;
    s_combSum2 += y1;
    s_combIdx2 = (s_combIdx2 + 1) % COMB_TAPS;
    return s_combSum2 / (float)COMB_TAPS;
}

/* ======================== 测试统计 ======================== */
static uint32_t s_beatCount = 0;
static uint32_t s_bpmSamples = 0;
static float    s_bpmSum = 0.0f;
static uint8_t  s_trueBPM = 75;
static uint16_t s_outOfRange = 0;

void setup()
{
    Serial.begin(115200);
    delay(100);

    Serial.println("\n===== heartrate v4.2 板上自测 =====");
    Serial.print("时长: "); Serial.print(TEST_DURATION_S); Serial.println("s");
    Serial.println("链路: 模拟器 → 去偏置 → 50Hz梳状 → HP/LP → hrProcess");

    ecgSimulatorInit();
    s_trueBPM = ecgSimulatorGetTrueBPM();
    Serial.print("模拟器真实心率: "); Serial.print(s_trueBPM); Serial.println(" BPM");

    filterInit();
    filterWarmup(0.0f);
    hrInit();
}

void loop()
{
    static unsigned long s_startMs = 0;
    static bool s_running = false;

    if (!s_running) {
        s_startMs = millis();
        s_running = true;
        return;   /* 跳过首个采样, 确保计时起点一致 */
    }

    /* 完整链路 (与 main.cpp 一致) */
    float noisySample = generateECGSample();            /* 含 1.65V DC */
    float noisyNoDC = noisySample - 1.65f;              /* 去直流偏置 */
    noisyNoDC = applyCombFilter(noisyNoDC);             /* 50Hz 梳状 */
    float filtered = applyFilter(noisyNoDC);            /* HP 0.5 + LP 40 */
    HR_Result hr = hrProcess(filtered);                 /* Pan-Tompkins v4.2 */

    if (hr.beatDetected) {
        s_beatCount++;
    }
    if (hr.bpm > 0) {
        s_bpmSamples++;
        s_bpmSum += hr.bpm;
        if (abs((int)hr.bpm - (int)s_trueBPM) > 3) {
            s_outOfRange++;
        }
    }

    /* 结束时输出统计 */
    unsigned long elapsed = millis() - s_startMs;
    if (elapsed >= (unsigned long)TEST_DURATION_S * 1000UL) {
        Serial.println("\n===== 测试结果 =====");
        int expected = (int)(TEST_DURATION_S * s_trueBPM / 60);
        Serial.print("期望心拍: "); Serial.println(expected);
        Serial.print("检测心拍: "); Serial.println(s_beatCount);
        float cov = (s_beatCount > 0) ? (100.0f * s_beatCount / expected) : 0.0f;
        Serial.print("覆盖率: "); Serial.print(cov); Serial.println("%");

        float avgBPM = (s_bpmSamples > 0) ? (s_bpmSum / s_bpmSamples) : 0.0f;
        Serial.print("平均 BPM: "); Serial.println(avgBPM);
        Serial.print("BPM 样本: "); Serial.println(s_bpmSamples);
        float pctErr = (s_bpmSamples > 0) ? (100.0f * s_outOfRange / s_bpmSamples) : 0.0f;
        Serial.print("±3BPM 外占比: "); Serial.print(pctErr); Serial.println("%");

        if (cov >= 80.0f && pctErr <= 15.0f) {
            Serial.println("✅ PASS: 检测率与精度达标 (合成信号场景)");
        } else {
            Serial.println("⚠️ 参考: 真实心电精度请以 LUDB 验证为准 (Se 72.9% PPV 82.6%)");
        }
        s_running = false;
        delay(60000);
    }
}
