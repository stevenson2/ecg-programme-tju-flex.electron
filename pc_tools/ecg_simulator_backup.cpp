#include <math.h>
#include <stdlib.h>
#include "signal_generator/ecg_simulator.h"

/**
 * @file ecg_simulator.cpp
 * @brief 带噪声的心电信号生成模块（适配ESP32 ADC量程）
 *
 * 信号特性：
 *   - 模拟经过模拟前端放大后的信号，直流偏置约1.65V
 *   - 心电波形幅度 1.0~2.3V（峰峰值约1.3V），适合ESP32 ADC输入
 *   - 每个心拍200点 @250Hz = 75bpm
 *
 * 噪声成分：
 *   - 基线漂移：0.2Hz 低频，幅度约 0.15V
 *   - 工频干扰：50Hz，幅度约 0.08V
 *   - 肌电噪声：高斯白噪声，幅度约 0.05V
 */

#define CYCLE_LENGTH    200
#define PHASE_STEP      (2.0f * 3.14159265f / CYCLE_LENGTH)

/* 心电波形幅度缩放因子：将 mV 级心电放大到 V 级 */
#define ECG_AMP_SCALE   1.0f    /* 原始高斯幅度已经是V级别 */

/* 噪声幅度（V级） */
#define BASELINE_AMP    0.15f
#define POWERNOISE_AMP  0.08f
#define MUSCLE_NOISE_SCALE  0.05f

/* 直流偏置（模拟前端参考电压，使信号在ADC最佳范围） */
#define DC_OFFSET       1.65f

/* 高斯波形成分参数（单位：V） */
#define P_AMP    0.25f
#define P_CENTER 0.18f
#define P_SIGMA  0.03f
#define Q_AMP   -0.10f
#define Q_CENTER 0.30f
#define Q_SIGMA  0.02f
#define R_AMP    1.20f
#define R_CENTER 0.33f
#define R_SIGMA  0.015f
#define S_AMP   -0.15f
#define S_CENTER 0.37f
#define S_SIGMA  0.025f
#define T_AMP    0.30f
#define T_CENTER 0.55f
#define T_SIGMA  0.06f

/* 状态变量 */
static int   sampleIndex = 0;
static float phase = 0.0f;
static int   randSeed = 42;
static float cleanValue = 0.0f;

static float gaussian(float x, float amp, float center, float sigma)
{
    float t = (x - center) / sigma;
    return amp * expf(-0.5f * t * t);
}

static float fastRand(void)
{
    randSeed = (randSeed * 1103515245 + 12345) & 0x7FFFFFFF;
    return (float)randSeed / 2147483648.0f;
}

void ecgSimulatorInit(void)
{
    ecgSimulatorReset();
    randSeed = 42;
}

float generateECGSample(void)
{
    float t;
    float ecgValue;
    float baseline;
    float powerNoise;
    float muscleNoise;

    t = (float)sampleIndex / CYCLE_LENGTH;

    /* 纯净心电信号（幅度已经是V级，峰峰值约1.3V） */
    cleanValue = gaussian(t, P_AMP, P_CENTER, P_SIGMA)
               + gaussian(t, Q_AMP, Q_CENTER, Q_SIGMA)
               + gaussian(t, R_AMP, R_CENTER, R_SIGMA)
               + gaussian(t, S_AMP, S_CENTER, S_SIGMA)
               + gaussian(t, T_AMP, T_CENTER, T_SIGMA);

    /* 基线漂移（0.2Hz） */
    baseline = BASELINE_AMP * sinf(2.0f * 3.14159265f * sampleIndex / 1250.0f);

    /* 50Hz 工频干扰 */
    powerNoise = POWERNOISE_AMP * sinf(2.0f * 3.14159265f * 50.0f * sampleIndex / 250.0f);

    /* 肌电噪声 */
    muscleNoise = 0.0f;
    for (int i = 0; i < 12; i++) {
        muscleNoise += fastRand();
    }
    muscleNoise = (muscleNoise - 6.0f) * MUSCLE_NOISE_SCALE;

    /* 合成信号 + 直流偏置（模拟ADC输入范围0~3.3V） */
    float noisyOutput = cleanValue + baseline + powerNoise + muscleNoise + DC_OFFSET;

    sampleIndex++;
    if (sampleIndex >= CYCLE_LENGTH) {
        sampleIndex = 0;
    }
    phase += PHASE_STEP;
    if (phase > 2.0f * 3.14159265f) {
        phase -= 2.0f * 3.14159265f;
    }

    return noisyOutput;
}

float getCleanECGValue(void)
{
    return cleanValue;
}

void ecgSimulatorReset(void)
{
    sampleIndex = 0;
    phase = 0.0f;
    cleanValue = 0.0f;
}
