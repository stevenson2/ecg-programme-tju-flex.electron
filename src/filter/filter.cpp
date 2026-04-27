#include <math.h>
#include "filter/filter.h"

/**
 * @file filter.cpp
 * @brief 心电信号数字滤波器（三级级联结构）
 *
 * 设计参数（采样率 Fs = 250Hz，场景三：家用便携/ESP32设备）：
 *
 * 第1级：二阶 Butterworth 高通 0.5Hz
 *   - 阶跃响应无过冲，0.5Hz为临床监护标准
 *   - 有效抑制基线漂移（占干扰总能量的27%）
 *
 * 第2级：二阶 Butterworth 低通 40Hz
 *   - 保留QRS主频（10Hz）能量，抑制肌电干扰
 *
 * 第3级：二阶 50Hz 陷波器 Q=30
 *   - 精准陷零50Hz（占干扰总能量的43%）
 *   - Q=30窄带不影响邻频ECG成分
 *
 * 级联总通带增益 ≈ 1.0（在5~30Hz范围内），无需额外补偿
 * 对比原设计：修复了旧陷波器系数错误（DC增益3.03的问题）
 */

/* ======================== 第1级：高通 0.5Hz ======================== */
#define HP_A1  -1.982229f
#define HP_A2   0.982385f
#define HP_B0   0.991154f
#define HP_B1  -1.982307f
#define HP_B2   0.991154f

/* ======================== 第2级：低通 40Hz ======================== */
#define LP_A1  -0.671029f
#define LP_A2   0.252325f
#define LP_B0   0.145324f
#define LP_B1   0.290648f
#define LP_B2   0.145324f

/* ======================== 第3级：50Hz 陷波 Q=30 ======================== */
#define NOTCH_A1  -0.608390f
#define NOTCH_A2   0.968793f
#define NOTCH_B0   0.984396f
#define NOTCH_B1  -0.608390f
#define NOTCH_B2   0.984396f

/* ======================== 状态变量 ======================== */
/* 第1级：高通 */
static float hp_w1 = 0.0f;
static float hp_w2 = 0.0f;
/* 第2级：低通 */
static float lp_w1 = 0.0f;
static float lp_w2 = 0.0f;
/* 第3级：陷波 */
static float notch_w1 = 0.0f;
static float notch_w2 = 0.0f;

/**
 * @brief 单级直接II型转置结构双二阶滤波器
 */
static float applyBiquad(float x,
                         float b0, float b1, float b2,
                         float a1, float a2,
                         float *w1, float *w2)
{
    float w = x - a1 * (*w1) - a2 * (*w2);
    float y = b0 * w + b1 * (*w1) + b2 * (*w2);
    *w2 = *w1;
    *w1 = w;
    return y;
}

static float highpassFilter(float x)
{
    return applyBiquad(x, HP_B0, HP_B1, HP_B2, HP_A1, HP_A2, &hp_w1, &hp_w2);
}

static float lowpassFilter(float x)
{
    return applyBiquad(x, LP_B0, LP_B1, LP_B2, LP_A1, LP_A2, &lp_w1, &lp_w2);
}

static float notchFilter(float x)
{
    return applyBiquad(x, NOTCH_B0, NOTCH_B1, NOTCH_B2,
                       NOTCH_A1, NOTCH_A2, &notch_w1, &notch_w2);
}

void filterInit(void)
{
    filterReset();
}

float applyFilter(float inputSample)
{
    float temp;
    /* 三级级联：HP → LP → Notch */
    temp = highpassFilter(inputSample);
    temp = lowpassFilter(temp);
    temp = notchFilter(temp);
    return temp;
}

void filterReset(void)
{
    hp_w1 = 0.0f;
    hp_w2 = 0.0f;
    lp_w1 = 0.0f;
    lp_w2 = 0.0f;
    notch_w1 = 0.0f;
    notch_w2 = 0.0f;
}
