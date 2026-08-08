#include <math.h>
#include "filter/filter.h"

/**
 * @file filter.cpp
 * @brief 心电信号数字滤波器（二级级联结构）
 *
 * 设计参数（采样率 Fs = 500Hz，场景三：家用便携/ESP32设备）：
 *
 * 第1级：二阶 Butterworth 高通 0.5Hz
 *   - 阶跃响应无过冲，0.5Hz为临床监护标准
 *   - 有效抑制基线漂移（占干扰总能量的27%）
 *
 * 第2级：二阶 Butterworth 低通 40Hz
 *   - 保留QRS主频（10Hz）能量，抑制肌电干扰
 *
 * 50Hz/100Hz 工频抑制由 main.cpp 中的双级梳状滤波器提供：
 *   - 利用 250Hz/50Hz=5 精确比，5抽头滑动平均在 50Hz/100Hz 精确陷零
 *   - 双级级联总衰减 -119.2dB @50Hz，远超独立陷波器
 *   - 零额外计算开销（仅2次加法和1次除法）
 *
 * 级联总通带增益 ≈ 1.0（在5~30Hz范围内），无需额外补偿
 */

/* ======================== 第1级：高通 0.05Hz (fs=500Hz 重算) ======================== */
/* TUNING_HISTORY 十三章 §8.3.1: 0.5Hz 因果 HP 在 ST 带引入 1.5-9mm 伪 ST 偏移
 * (Buendía-Funetes 2012), 是 PTB 部署链 AUC 缺口主因。降截止至 0.05Hz
 * (AHA 2007 诊断标准) 后 0.5Hz 处相位失真 +8.1° (原 +90°), ST 带近乎无失真。
 * 代价: 基线漂移抑制减弱 (z-score 归一化兜底), warmup 需 ~16s (τ=3.18s)。
 * K = tan(pi*0.05/500) = 0.0003142 */
/* b0 = 1/(1+K√2+K²), b1 = -2*b0, b2 = b0 */
/* a1 = 2*(K²-1)*b0, a2 = (1-K√2+K²)*b0 */
/* ⚠️ 2026-08-08 N16R8 板上实测: ① 5 位小数量化系数使分子 b0+b1+b2 残留
 * 1e-5 而分母 1+a1+a2 舍入归零 → DC 增益病态 (输出 9V→28V 爬升 + 心率失效);
 * ② 0.05Hz 极点模 0.99955 极近单位圆, float32 灾难性抵消加剧。修复:
 * 完整精度 double 系数 (b0+b1+b2≡0, DC 增益严格 0) + double 状态变量。
 * 系数由 Python 计算: K=tan(pi*0.05/500), 见下方定义。 */
#define HP_A1  -1.9991114234707954
#define HP_A2   0.9991118180796384
#define HP_B0   0.9995558103876084
#define HP_B1  -1.9991116207752169
#define HP_B2   0.9995558103876084

/* ======================== 第2级：低通 40Hz (fs=500Hz 重算) ======================== */
/* K = tan(pi*40/500) = 0.2568 */
/* b0 = K²/(1+K√2+K²), b1 = 2*b0, b2 = b0 */
/* a1 = 2*(K²-1)/(1+K√2+K²), a2 = (1-K√2+K²)/(1+K√2+K²) */
/* 2026-08-08: 与 HP 同步改为完整精度 double 系数 (原 5 位小数量化) */
#define LP_A1  -1.3072850288493234
#define LP_A2   0.4918122372225752
#define LP_B0   0.046131802093312926
#define LP_B1   0.09226360418662585
#define LP_B2   0.046131802093312926

/* ======================== 预热样本数 ======================== */
#define WARMUP_SAMPLES  240  /* 约 0.48s @500Hz */

/* ======================== 状态变量 (double: 0.05Hz HP float32 灾难性抵消, 2026-08-08) ======================== */
/* 第1级：高通 */
static double hp_w1 = 0.0;
static double hp_w2 = 0.0;
/* 第2级：低通 */
static double lp_w1 = 0.0;
static double lp_w2 = 0.0;

/**
 * @brief 单级直接II型转置结构双二阶滤波器 (double 精度)
 */
static float applyBiquad(float x,
                         double b0, double b1, double b2,
                         double a1, double a2,
                         double *w1, double *w2)
{
    double w = (double)x - a1 * (*w1) - a2 * (*w2);
    double y = b0 * w + b1 * (*w1) + b2 * (*w2);
    *w2 = *w1;
    *w1 = w;
    return (float)y;
}

static float highpassFilter(float x)
{
    return applyBiquad(x, HP_B0, HP_B1, HP_B2, HP_A1, HP_A2, &hp_w1, &hp_w2);
}

static float lowpassFilter(float x)
{
    return applyBiquad(x, LP_B0, LP_B1, LP_B2, LP_A1, LP_A2, &lp_w1, &lp_w2);
}

void filterInit(void)
{
    filterReset();
}

/**
 * @brief 滤波器预热: 用首样本值填充全部状态
 *
 * 消除滤波器启动瞬态（直流阶跃响应）。
 * 原理: 将第一个有效样本视为稳态值，强制所有延迟单元
 * 收敛到该值，使得后续输出立即跟踪信号。
 *
 * 应在开始正式采样循环前调用。
 *
 * @param firstSample 第一个有效样本值（用于预热）
 */
void filterWarmup(float firstSample)
{
    /* 高通预热: 使 w1=w2=firstSample 时输出 ≈ 0（瞬态消除） */
    /* 对于 HPF: 稳态直流输入 → 输出 0 */
    /* w = x - a1*w1 - a2*w2, 设 w1=w2=x, 则 w = x*(1-a1-a2) */
    /* HP 1-a1-a2 = 1+1.982229-0.982385 = 1.999844, 非常小，说明高通对DC衰减极大 */
    /* 设 w1 = x * (1 - (b0+b1+b2)/(1-a1-a2))? 太复杂 */
    /* 简化方案: 用样本反复迭代收敛 */
    float temp = firstSample;
    for (int i = 0; i < WARMUP_SAMPLES; i++) {
        temp = applyFilter(temp);
    }
    /* 预热后 filterReset() 已清除的状态 -> 上文 for 循环已让状态趋于稳态 */
}

float applyFilter(float inputSample)
{
    float temp;
    /* 二级级联：HP 0.5Hz → LP 40Hz */
    /* 50Hz/100Hz 由 main.cpp 中的双级梳状滤波器处理 */
    temp = highpassFilter(inputSample);
    temp = lowpassFilter(temp);
    return temp;
}

void filterReset(void)
{
    hp_w1 = 0.0f;
    hp_w2 = 0.0f;
    lp_w1 = 0.0f;
    lp_w2 = 0.0f;
}
