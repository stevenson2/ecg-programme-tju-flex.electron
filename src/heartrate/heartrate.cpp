#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "heartrate/heartrate.h"

/**
 * @file heartrate.cpp
 * @brief 板上心率计算模块 - 实现 (v2)
 *
 * ========== 简化 Pan-Tompkins 算法 ==========
 *
 * 经典 Pan-Tompkins 五步 (1985):
 *   ① 带通滤波 5~15Hz  → 信号已由 filter.cpp 预处理, 跳过
 *   ② 一阶差分          → 突出 QRS 的快速上升/下降沿
 *   ③ 逐点平方          → 放大 R 波能量, 使 T/P 波相对更小
 *   ④ 滑动窗口积分      → 150ms 矩形窗, 将 QRS 能量汇聚为单个峰
 *   ⑤ 自适应阈值检测    → 单阈值 + 200ms 不应期
 *
 * ========== v2 改进 ==========
 *   - 中位数 BPM 平滑 (vs 均值): 抗单拍异常跳变
 *   - 阈值系数 0.40 (vs 0.30): 误检更少
 *   - 去除搜索回退: 宁可漏检也不误检
 *   - 峰值质量门限: 必须 > 2×噪声基底
 *   - 学习阶段: 首 5 拍仅建立基线, 不输出 BPM
 */

/* ======================== 算法常数 ======================== */
#define FS              250.0f      /**< 采样率 (Hz) */
#define TS              0.004f      /**< 采样间隔 (s) */

#define MWI_WINDOW      38          /**< 滑动积分窗口 150ms @250Hz */
#define REFRACTORY_SAMP 50          /**< 不应期 200ms = 50 样本 */

#define RR_BUFFER_SIZE  8           /**< BPM 平滑缓冲区容量 */

#define THRESHOLD_INIT  0.002f      /**< 初始阈值 */
#define THRESHOLD_RATIO 0.40f       /**< 阈值 = 噪声 + 0.40×(信号−噪声) */
#define SIGNAL_WEIGHT   0.125f      /**< 信号峰值更新因子 (EMA) */
#define NOISE_WEIGHT    0.125f      /**< 噪声峰值更新因子 (EMA) */

#define MIN_RR_SAMP     75          /**< 最小 RR: 300ms @250Hz */
#define MAX_RR_SAMP     500         /**< 最大 RR: 2000ms @250Hz */

#define TIMEOUT_SAMP    750         /**< 3 秒无 QRS → 复位 */
#define HOLD_SAMP       250         /**< 1 秒无新拍 → 停止输出旧 BPM */
#define MIN_CONF_BEATS  5           /**< 至少 5 拍才开始输出 BPM */
#define MIN_PEAK_RATIO  2.0f        /**< 峰/噪比门限 */

/* ======================== 状态机 ======================== */
typedef enum {
    HR_LEARNING,                    /**< 学习阶段 (前 5 拍, 不输出 BPM) */
    HR_IDLE,                        /**< 等待信号超过阈值 */
    HR_REFRACTORY,                  /**< 不应期 (检测到 QRS 后 200ms) */
    HR_TRACKING                     /**< 正常追踪 */
} HR_State;

/* ======================== 静态变量 ======================== */
static float      s_prevSample;
static float      s_mwiBuf[MWI_WINDOW];
static int        s_mwiIdx;
static float      s_mwiSum;
static float      s_mwiPrev;
static float      s_mwiPrevPrev;

static HR_State   s_state;
static int        s_refractCount;
static float      s_signalPeak;
static float      s_noisePeak;
static float      s_threshold;

static float      s_rrBuf[RR_BUFFER_SIZE];
static int        s_rrIdx;
static int        s_rrCount;
static float      s_medianRR;
static float      s_lastRR;

static int        s_sampSinceBeat;
static uint32_t   s_beatCount;

/* ======================== 工具函数 ======================== */

static inline float computeMWI(float squared)
{
    s_mwiSum -= s_mwiBuf[s_mwiIdx];
    s_mwiBuf[s_mwiIdx] = squared;
    s_mwiSum += squared;
    s_mwiIdx = (s_mwiIdx + 1) % MWI_WINDOW;
    return s_mwiSum / (float)MWI_WINDOW;
}

static void updateThreshold(float peakVal, bool isSignal)
{
    if (isSignal) {
        s_signalPeak = SIGNAL_WEIGHT * peakVal
                     + (1.0f - SIGNAL_WEIGHT) * s_signalPeak;
    } else {
        s_noisePeak = NOISE_WEIGHT * peakVal
                    + (1.0f - NOISE_WEIGHT) * s_noisePeak;
    }

    float delta = s_signalPeak - s_noisePeak;
    if (delta < 0.001f) delta = 0.001f;

    s_threshold = s_noisePeak + THRESHOLD_RATIO * delta;
    if (s_threshold < THRESHOLD_INIT) {
        s_threshold = THRESHOLD_INIT;
    }
}

static int cmpFloat(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb) return -1;
    if (fa > fb) return  1;
    return 0;
}

static float computeMedianRR(void)
{
    if (s_rrCount == 0) return 0.0f;
    float temp[RR_BUFFER_SIZE];
    memcpy(temp, s_rrBuf, sizeof(float) * s_rrCount);
    qsort(temp, s_rrCount, sizeof(float), cmpFloat);
    if (s_rrCount % 2 == 1) {
        return temp[s_rrCount / 2];
    } else {
        return (temp[s_rrCount / 2 - 1] + temp[s_rrCount / 2]) * 0.5f;
    }
}

static void addRRInterval(float rrSeconds)
{
    int rrSamp = (int)(rrSeconds / TS + 0.5f);
    if (rrSamp < MIN_RR_SAMP || rrSamp > MAX_RR_SAMP) return;

    s_rrBuf[s_rrIdx] = rrSeconds;
    s_rrIdx = (s_rrIdx + 1) % RR_BUFFER_SIZE;
    if (s_rrCount < RR_BUFFER_SIZE) s_rrCount++;

    s_medianRR = computeMedianRR();
    s_lastRR = rrSeconds;
}

static bool isQRSValid(float peakVal)
{
    if (peakVal <= s_threshold)                 return false;
    if (s_state == HR_REFRACTORY)               return false;
    if (peakVal < s_noisePeak * MIN_PEAK_RATIO) return false;
    return true;
}

/* ======================== 公共 API ======================== */

void hrInit(void)
{
    hrReset();
    s_signalPeak = THRESHOLD_INIT;
    s_noisePeak  = THRESHOLD_INIT * 0.3f;
    s_threshold  = THRESHOLD_INIT;
}

HR_Result hrProcess(float filteredSample)
{
    HR_Result result = { 0 };

    /* ---- 步骤 1-3: 差分 → 平方 → 滑动积分 ---- */
    float diff = filteredSample - s_prevSample;
    s_prevSample = filteredSample;
    float squared = diff * diff;
    float mwi = computeMWI(squared);

    /* ---- 步骤 4: 峰值检测 ---- */
    bool isPeak = (s_mwiPrev > s_mwiPrevPrev) && (s_mwiPrev > mwi);

    if (isPeak) {
        float peakVal = s_mwiPrev;

        if (isQRSValid(peakVal)) {
            /* √ QRS 检测成功 */
            float rrSec = (float)s_sampSinceBeat * TS;
            addRRInterval(rrSec);
            updateThreshold(peakVal, true);

            s_state = HR_REFRACTORY;
            s_refractCount = 0;
            s_sampSinceBeat = 0;
            s_beatCount++;

            result.beatDetected = true;
            result.beatCount    = s_beatCount;
            result.rrInterval   = s_lastRR;

            /* 学习满 MIN_CONF_BEATS 拍后进入 TRACKING */
            if (s_state == HR_LEARNING && s_beatCount >= MIN_CONF_BEATS) {
                s_state = HR_TRACKING;
            }

            /* TRACKING 状态才输出 BPM */
            if (s_state == HR_TRACKING && s_medianRR > 0.001f) {
                uint8_t bpmRaw = (uint8_t)(60.0f / s_medianRR + 0.5f);
                if (bpmRaw >= 30 && bpmRaw <= 200) {
                    result.bpm = bpmRaw;
                }
                result.confidence = fminf(1.0f,
                    (float)s_rrCount / (float)RR_BUFFER_SIZE);
            }

        } else {
            /* 噪声峰 → 更新噪声估计 */
            if (peakVal > s_noisePeak * 0.5f) {
                updateThreshold(peakVal, false);
            }
        }
    }

    /* ---- 步骤 5: 状态机更新 ---- */
    s_sampSinceBeat++;

    if (s_state == HR_REFRACTORY) {
        s_refractCount++;
        if (s_refractCount >= REFRACTORY_SAMP) {
            s_state = (s_beatCount >= MIN_CONF_BEATS)
                      ? HR_TRACKING : HR_IDLE;
        }
    }

    /* ---- 步骤 6: 超时复位 ---- */
    if (s_sampSinceBeat > TIMEOUT_SAMP) {
        hrReset();
        s_state = HR_LEARNING;
    }

    /* ---- 步骤 7: 滚动 MWI 历史 ---- */
    s_mwiPrevPrev = s_mwiPrev;
    s_mwiPrev     = mwi;

    /* ---- 步骤 8: 无新拍时保持 (1 秒内) ---- */
    if (!result.beatDetected && s_beatCount > 0 && s_medianRR > 0.001f) {
        if (s_sampSinceBeat < HOLD_SAMP) {
            result.bpm = (uint8_t)(60.0f / s_medianRR + 0.5f);
            float decay = 1.0f - (float)s_sampSinceBeat / (float)HOLD_SAMP;
            result.confidence = fminf(0.8f,
                (float)s_rrCount / (float)RR_BUFFER_SIZE) * decay;
        }
    }

    return result;
}

void hrReset(void)
{
    s_prevSample = 0.0f;
    for (int i = 0; i < MWI_WINDOW; i++) s_mwiBuf[i] = 0.0f;
    s_mwiIdx      = 0;
    s_mwiSum      = 0.0f;
    s_mwiPrev     = 0.0f;
    s_mwiPrevPrev = 0.0f;

    s_state        = HR_LEARNING;
    s_refractCount = 0;
    s_signalPeak   = THRESHOLD_INIT;
    s_noisePeak    = THRESHOLD_INIT * 0.3f;
    s_threshold    = THRESHOLD_INIT;

    for (int i = 0; i < RR_BUFFER_SIZE; i++) s_rrBuf[i] = 0.0f;
    s_rrIdx         = 0;
    s_rrCount       = 0;
    s_medianRR      = 0.0f;
    s_sampSinceBeat = 0;
    s_beatCount     = 0;
    s_lastRR        = 0.0f;
}
