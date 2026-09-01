#include <Arduino.h>
#include <math.h>
#include <string.h>
#include "respiration/respiration.h"

/**
 * @file respiration.cpp
 * @brief 呼吸率检测实现 (500Hz)
 *
 * 采用轻量级过零检测:
 *   1. 低通 (约 2Hz) 滤除高频噪声;
 *   2. 移动平均基线估计 (约 1s) 去除直流/基线漂移;
 *   3. 正向过零 + 幅度阈值判定一次吸气/呼气周期;
 *   4. 最近若干周期取中位数得到稳定 brpm。
 *
 * 呼吸频率范围设定为 4~60 次/分, 对应周期 1s~15s。
 */

#define RESP_FS            500.0f
#define RESP_LP_ALPHA      0.10f    /* 2Hz 附近简单一阶低通 */
#define RESP_BASE_ALPHA    0.005f   /* 约 1s 时间常数的基线估计 */
#define RESP_MIN_INTERVAL  1.0f     /* 60 bpm */
#define RESP_MAX_INTERVAL  15.0f    /* 4 bpm */
#define RESP_MIN_AMP       0.0002f  /* 最小呼吸幅度 (200uV), 防噪声过零 */
#define RESP_HISTORY_MAX   8

typedef struct {
    float interval;
    bool  valid;
} RespInterval;

static float      s_lp = 0.0f;
static float      s_baseline = 0.0f;
static float      s_prevAc = 0.0f;
static bool       s_hasPrev = false;
static uint32_t   s_sampleCount = 0;
static uint32_t   s_lastCrossSample = 0;
static uint32_t   s_breathCount = 0;
static float      s_recentPeak = 0.0f;
static RespInterval s_history[RESP_HISTORY_MAX];
static int        s_historyCount = 0;
static float      s_bpm = 0.0f;
static bool       s_valid = false;

static void respResetInternal(void)
{
    s_lp = 0.0f;
    s_baseline = 0.0f;
    s_prevAc = 0.0f;
    s_hasPrev = false;
    s_sampleCount = 0;
    s_lastCrossSample = 0;
    s_breathCount = 0;
    s_recentPeak = 0.0f;
    memset(s_history, 0, sizeof(s_history));
    s_historyCount = 0;
    s_bpm = 0.0f;
    s_valid = false;
}

void respInit(void)
{
    respResetInternal();
}

void respReset(void)
{
    respResetInternal();
}

static void pushInterval(float interval)
{
    if (interval < RESP_MIN_INTERVAL || interval > RESP_MAX_INTERVAL) return;
    if (s_historyCount < RESP_HISTORY_MAX) {
        s_history[s_historyCount].interval = interval;
        s_history[s_historyCount].valid = true;
        s_historyCount++;
    } else {
        /* 环形覆盖最旧 */
        for (int i = 1; i < RESP_HISTORY_MAX; i++) {
            s_history[i - 1] = s_history[i];
        }
        s_history[RESP_HISTORY_MAX - 1].interval = interval;
        s_history[RESP_HISTORY_MAX - 1].valid = true;
    }
}

static float medianInterval(void)
{
    if (s_historyCount == 0) return 0.0f;
    float tmp[RESP_HISTORY_MAX];
    int n = 0;
    for (int i = 0; i < s_historyCount; i++) {
        if (s_history[i].valid) tmp[n++] = s_history[i].interval;
    }
    if (n == 0) return 0.0f;
    /* 简单插入排序 */
    for (int i = 1; i < n; i++) {
        float key = tmp[i];
        int j = i - 1;
        while (j >= 0 && tmp[j] > key) {
            tmp[j + 1] = tmp[j];
            j--;
        }
        tmp[j + 1] = key;
    }
    if (n % 2 == 1) return tmp[n / 2];
    return 0.5f * (tmp[n / 2 - 1] + tmp[n / 2]);
}

Resp_Result respProcess(float rawResp)
{
    /* 低通平滑 */
    s_lp = s_lp + RESP_LP_ALPHA * (rawResp - s_lp);

    /* 基线估计 */
    s_baseline = s_baseline + RESP_BASE_ALPHA * (s_lp - s_baseline);

    float ac = s_lp - s_baseline;

    /* 峰值幅度跟踪 */
    float a = fabsf(ac);
    if (a > s_recentPeak) s_recentPeak = a;
    s_recentPeak *= 0.9995f;   /* 缓慢衰减, 适应幅度变化 */

    /* 正向过零 */
    if (s_hasPrev && s_prevAc <= 0.0f && ac > 0.0f && s_recentPeak >= RESP_MIN_AMP) {
        if (s_lastCrossSample != 0) {
            float interval = (float)(s_sampleCount - s_lastCrossSample) / RESP_FS;
            pushInterval(interval);
            s_breathCount++;
        }
        s_lastCrossSample = s_sampleCount;
        s_recentPeak = 0.0f;  /* 重置峰值跟踪, 适应下一个周期 */
    }

    s_prevAc = ac;
    s_hasPrev = true;
    s_sampleCount++;

    s_valid = false;
    if (s_historyCount >= 2) {
        float med = medianInterval();
        if (med > 0.0f) {
            s_bpm = 60.0f / med;
            s_valid = (s_bpm >= 4.0f && s_bpm <= 60.0f);
        }
    }

    Resp_Result r;
    r.bpm = s_valid ? s_bpm : 0.0f;
    r.amplitude = s_recentPeak;
    r.breathCount = s_breathCount;
    r.valid = s_valid;
    return r;
}
