#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "heartrate/heartrate.h"

/**
 * @file heartrate.cpp
 * @brief 板上心率计算模块 - 实现 (v4.0, fs=500Hz)
 *
 * ========== v4.0 改进: 新增QRS专用5~15Hz带通滤波器 ==========
 *
 * 经典 Pan-Tompkins 算法的第一步是 5~15Hz 带通滤波，专用于
 * 突出 QRS 波群能量、抑制 T/P 波和肌电干扰。此前代码直接使用
 * 0.5~40Hz 滤波后的信号，其中包含大量肌电噪声和残余工频分量，
 * 导致假阳性高企。
 *
 * v4.0 在 hrProcess() 入口处新增两级 Butterworth 2阶节：
 *   第1节: 低通 15Hz (抑制肌电干扰和工频谐波)
 *   第2节: 高通 5Hz  (抑制 T 波和基线漂移)
 *   → 等效 4阶 5~15Hz 带通，专用于 QRS 检测
 *
 * fs=500Hz 时系数由 scipy.signal.butter(2, [5,15], 'band', fs=500) 重算
 * 
 * ========== 简化 Pan-Tompkins + 4 维形态学验证 + 记录纠错 ==========
 */

/* ======================== 算法常数 (500Hz) ======================== */
#define FS              500.0f      /**< 采样率 (Hz) */
#define TS              0.002f      /**< 采样间隔 (s) */

/*
 * ========== QRS 专用带通滤波器 (v4.0, fs=500Hz 重算) ==========
 *
 * 两级级联 Butterworth 2阶 → 等效 4阶 5~15Hz 带通
 * 系数由 scipy.signal.butter(2, [5, 15], 'band', fs=500) 计算
 *
 * 第1节: 低通 15Hz (2阶 Butterworth)
 *   B = (0.00549, 0.01097, 0.00549)
 *   A = (1, -1.75513, 0.77708)
 *
 * 第2节: 高通 5Hz (2阶 Butterworth)
 *   B = (0.89127, -1.78254, 0.89127)
 *   A = (1, -1.77666, 0.80084)
 */
#define QRS_LP15_A1  -1.75513f
#define QRS_LP15_A2   0.77708f
#define QRS_LP15_B0   0.00549f
#define QRS_LP15_B1   0.01097f
#define QRS_LP15_B2   0.00549f

#define QRS_HP5_A1   -1.77666f
#define QRS_HP5_A2    0.80084f
#define QRS_HP5_B0    0.89127f
#define QRS_HP5_B1   -1.78254f
#define QRS_HP5_B2    0.89127f

/* QRS BPF 状态变量 */
static float qrs_bpf_lp_w1 = 0.0f;
static float qrs_bpf_lp_w2 = 0.0f;
static float qrs_bpf_hp_w1 = 0.0f;
static float qrs_bpf_hp_w2 = 0.0f;

#define MWI_WINDOW      75          /**< 滑动积分窗口 150ms @500Hz */
#define REFRACTORY_SAMP 100         /**< 不应期 200ms = 100 样本 */

#define RR_BUFFER_SIZE  8           /**< BPM 中位数缓冲区容量 */

#define THRESHOLD_INIT  0.002f      /**< 初始阈值 */
#define THRESHOLD_RATIO 0.40f       /**< 阈值 = 噪声 + 0.40×(信号−噪声) */
#define SIGNAL_WEIGHT   0.125f      /**< 信号峰值更新因子 (EMA) */
#define NOISE_WEIGHT    0.125f      /**< 噪声峰值更新因子 (EMA) */
#define SIGNAL_WEIGHT_FAST 0.25f    /**< 运动恢复期快速收敛信号峰值 */
#define SIGNAL_WEIGHT_MOT  0.02f    /**< 运动期极慢更新 signalPeak */

#define MIN_RR_SAMP     150         /**< 最小 RR: 300ms @500Hz */
#define MAX_RR_SAMP     1000        /**< 最大 RR: 2000ms @500Hz */

#define TIMEOUT_SAMP    1500        /**< 3 秒无 QRS → 复位 @500Hz */
#define HOLD_SAMP       500         /**< 1 秒无新拍 → 停止输出旧 BPM */
#define MIN_CONF_BEATS  5           /**< 至少 5 拍才开始输出 BPM */
#define MIN_CONF_FEAT   8           /**< 至少 8 拍才开启特征验证 */
#define MIN_PEAK_RATIO  2.0f        /**< 峰/噪比门限 (静止) */
#define MIN_PEAK_RATIO_MOT 1.5f     /**< 运动期峰噪比 (降低, 易检测) */

/* ======== 信号活动检测 ======== */
#define ACT_WIN_SAMP    500         /**< 活动检测窗口: 1秒 @500Hz */
#define ACT_THRESHOLD   0.005f      /**< 最小信号幅度 5mV */
#define ACT_TIMEOUT_CNT 2           /**< 连续 2 秒无信号→标记消失 */

/* ======== 自适应初始阈值 ======== */
#define ADAPT_INIT_SAMP 100         /**< 自适应学习采样数 (200ms @500Hz) */
#define ADAPT_INIT_FACTOR 2.0f      /**< 阈值 = 基线噪声 RMS × 2.0 */

/* ======== SQI 与运动检测 ======== */
#define SQI_EMA_WEIGHT  0.05f       /**< SQI 指数平滑因子 (慢) */
#define SQI_MOTION_ENTER 0.35f      /**< SQI 低于此值 → 进入运动状态 */
#define SQI_MOTION_EXIT  0.55f      /**< SQI 高于此值 → 退出运动状态 */
#define SQI_SNR_FLOOR    0.001f     /**< SNR 最小值, 防止除零 */
#define MOTION_BPM_HOLD  1500       /**< 运动结束后保持峰值冻结的帧数 (3秒) */

/* ======== 运动检测滞回 ======== */
#define MOTION_ENTER_CNT 250        /**< 连续250帧(0.5秒)SQI低→进入运动 */
#define MOTION_EXIT_CNT  100        /**< 连续100帧(0.2秒)SQI高→退出运动 */

/* ======== 形态学验证常数 (静止) ======== */
#define MWI_HIST_LEN     120        /**< MWI 历史长度 (240ms @500Hz) */
#define PEAK_HIST_LEN    8          /**< 近期峰值历史长度 */
#define MIN_QRS_WIDTH    40         /**< 最小 QRS 半高宽 80ms @500Hz */
#define MAX_QRS_WIDTH    80         /**< 最大 QRS 半高宽 160ms @500Hz */
#define AMP_CONSISTENCY  0.35f      /**< 振幅一致性容差 ±35% */
#define RR_CONSISTENCY   0.30f      /**< RR 一致性容差 ±30% (静止) */
#define RISE_FALL_MIN    0.5f       /**< 最小上升/下降比 */
#define RISE_FALL_MAX    2.0f       /**< 最大上升/下降比 */

/* ======== 运动期放宽约束 ======== */
#define MIN_QRS_WIDTH_MOT  30       /**< 运动期最小 QRS 半高宽 60ms */
#define MAX_QRS_WIDTH_MOT  100      /**< 运动期最大 QRS 半高宽 200ms */
#define RISE_FALL_MIN_MOT  0.35f    /**< 运动期最小上升/下降比 */
#define RISE_FALL_MAX_MOT  2.5f     /**< 运动期最大上升/下降比 */

/* ======== BPM EMA 平滑 ======== */
#define BPM_EMA_WEIGHT_FAST 0.30f
#define BPM_EMA_WEIGHT_SLOW 0.10f
#define BPM_EMA_FADE_STEPS  500

/* ======== BPM 跃升防护 ======== */
#define BPM_EMA_WEIGHT_ANOM 0.05f
#define BPM_ANOMALY_THRESH  0.40f
#define BPM_SLEW_MAX        3.0f

/* ======== BPM 记录+纠错 ======== */
#define BPM_CONFIRMED_LEN   5
#define BPM_REJECT_DEV      0.30f
#define BPM_REJECT_DEV_MOT  0.45f
#define BPM_REJECT_MIN_CNT  3

/* ======================== 状态机 ======================== */
typedef enum {
    HR_LEARNING,
    HR_IDLE,
    HR_REFRACTORY,
    HR_TRACKING
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

static bool       s_signalPresent;
static float      s_winMaxAbs;
static int        s_winCount;
static int        s_noSignalSeconds;

static float      s_sqi;
static bool       s_motionActive;
static bool       s_motionConfirmed;
static int        s_motionLowCount;
static int        s_motionHighCount;
static int        s_motionRecoverCnt;
static float      s_motionHoldSP;
static float      s_motionHoldNP;

static float      s_mwiHistory[MWI_HIST_LEN];
static int        s_mwiHistIdx;
static float      s_recentPeaks[PEAK_HIST_LEN];
static int        s_peakHistIdx;
static int        s_peakHistCount;

static float      s_bpmEMA;
static int        s_bpmEmaFadeCnt;

static float      s_lastOutputBPM;

static float      s_confirmedBPM[BPM_CONFIRMED_LEN];
static int        s_confirmedBPMIdx;
static int        s_confirmedBPMCount;

static bool        s_adaptInitDone;
static int         s_adaptInitCount;
static float       s_adaptInitSumSq;

/* ======================== 工具函数 ======================== */

static float applyQRSBandpass(float x)
{
    float w_lp = x - QRS_LP15_A1 * qrs_bpf_lp_w1 - QRS_LP15_A2 * qrs_bpf_lp_w2;
    float y_lp = QRS_LP15_B0 * w_lp + QRS_LP15_B1 * qrs_bpf_lp_w1
               + QRS_LP15_B2 * qrs_bpf_lp_w2;
    qrs_bpf_lp_w2 = qrs_bpf_lp_w1;
    qrs_bpf_lp_w1 = w_lp;

    float w_hp = y_lp - QRS_HP5_A1 * qrs_bpf_hp_w1 - QRS_HP5_A2 * qrs_bpf_hp_w2;
    float y_hp = QRS_HP5_B0 * w_hp + QRS_HP5_B1 * qrs_bpf_hp_w1
               + QRS_HP5_B2 * qrs_bpf_hp_w2;
    qrs_bpf_hp_w2 = qrs_bpf_hp_w1;
    qrs_bpf_hp_w1 = w_hp;

    return y_hp;
}

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
        float weight;
        if (s_motionConfirmed) {
            weight = SIGNAL_WEIGHT_MOT;
        } else if (s_motionRecoverCnt > 0) {
            weight = SIGNAL_WEIGHT_FAST;
        } else {
            weight = SIGNAL_WEIGHT;
        }
        s_signalPeak = weight * peakVal + (1.0f - weight) * s_signalPeak;
    } else {
        float nWeight = NOISE_WEIGHT;
        s_noisePeak = nWeight * peakVal + (1.0f - nWeight) * s_noisePeak;
    }

    float delta = s_signalPeak - s_noisePeak;
    if (delta < 0.001f) delta = 0.001f;

    s_threshold = s_noisePeak + THRESHOLD_RATIO * delta;
    if (s_threshold < THRESHOLD_INIT) {
        s_threshold = THRESHOLD_INIT;
    }
}

static void updateSQI(void)
{
    float snrDenom = s_signalPeak + s_noisePeak + SQI_SNR_FLOOR;
    float rawSQI = s_signalPeak / snrDenom;
    if (rawSQI > 1.0f) rawSQI = 1.0f;
    if (rawSQI < 0.0f) rawSQI = 0.0f;
    s_sqi = SQI_EMA_WEIGHT * rawSQI + (1.0f - SQI_EMA_WEIGHT) * s_sqi;
}

static void updateMotionState(void)
{
    if (!s_motionConfirmed) {
        if (s_sqi < SQI_MOTION_ENTER) {
            s_motionLowCount++;
            s_motionHighCount = 0;
        } else {
            s_motionLowCount = 0;
        }
        if (s_motionLowCount >= MOTION_ENTER_CNT) {
            s_motionConfirmed = true;
            s_motionActive = true;
            s_motionLowCount = 0;
            s_motionHoldSP = s_signalPeak;
            s_motionHoldNP = s_noisePeak;
            s_motionRecoverCnt = 0;
            s_bpmEmaFadeCnt = 0;
        }
    } else {
        if (s_sqi > SQI_MOTION_EXIT) {
            s_motionHighCount++;
            s_motionLowCount = 0;
        } else {
            s_motionHighCount = 0;
        }
        if (s_motionHighCount >= MOTION_EXIT_CNT) {
            s_motionConfirmed = false;
            s_motionActive = false;
            s_motionHighCount = 0;
            s_motionRecoverCnt = MOTION_BPM_HOLD;
            s_bpmEmaFadeCnt = BPM_EMA_FADE_STEPS;
            if (s_motionHoldSP > s_signalPeak) {
                s_signalPeak = s_motionHoldSP;
            }
            if (s_motionHoldNP < s_noisePeak) {
                s_noisePeak = s_motionHoldNP;
            }
        }
    }

    if (s_motionRecoverCnt > 0) {
        s_motionRecoverCnt--;
    }
    if (s_bpmEmaFadeCnt > 0) {
        s_bpmEmaFadeCnt--;
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

static float getConfirmedBPMMedian(void)
{
    if (s_confirmedBPMCount == 0) return 0.0f;
    float temp[BPM_CONFIRMED_LEN];
    memcpy(temp, s_confirmedBPM, sizeof(float) * s_confirmedBPMCount);
    qsort(temp, s_confirmedBPMCount, sizeof(float), cmpFloat);
    if (s_confirmedBPMCount % 2 == 1) {
        return temp[s_confirmedBPMCount / 2];
    } else {
        return (temp[s_confirmedBPMCount / 2 - 1]
              + temp[s_confirmedBPMCount / 2]) * 0.5f;
    }
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

    float oldMedianRR = s_medianRR;
    s_medianRR = computeMedianRR();
    s_lastRR = rrSeconds;

    float instBPM = 60.0f / rrSeconds;

    if (s_bpmEMA < 1.0f) {
        s_bpmEMA = instBPM;
    } else {
        bool isAnomalous = false;
        if (oldMedianRR > 0.001f) {
            float medBPM = 60.0f / oldMedianRR;
            float deviation = fabsf(instBPM - medBPM) / medBPM;
            if (deviation > BPM_ANOMALY_THRESH) {
                isAnomalous = true;
            }
        }

        float weight;
        if (isAnomalous) {
            weight = BPM_EMA_WEIGHT_ANOM;
        } else if (s_motionConfirmed || s_motionRecoverCnt > 0) {
            weight = BPM_EMA_WEIGHT_FAST;
        } else {
            weight = BPM_EMA_WEIGHT_SLOW;
        }

        s_bpmEMA = weight * instBPM + (1.0f - weight) * s_bpmEMA;
    }
}

static void recordValidPeak(float peakVal)
{
    s_recentPeaks[s_peakHistIdx] = peakVal;
    s_peakHistIdx = (s_peakHistIdx + 1) % PEAK_HIST_LEN;
    if (s_peakHistCount < PEAK_HIST_LEN) {
        s_peakHistCount++;
    }
}

static bool isAmplitudeConsistent(float peakVal)
{
    if (s_peakHistCount < 3) return true;
    float sum = 0.0f;
    for (int i = 0; i < s_peakHistCount; i++) {
        sum += s_recentPeaks[i];
    }
    float mean = sum / (float)s_peakHistCount;
    if (mean < 0.0001f) return true;
    float deviation = fabsf(peakVal - mean) / mean;
    return (deviation <= AMP_CONSISTENCY);
}

static int getQRSWidth(void)
{
    int peakIdx = (s_mwiHistIdx - 2 + MWI_HIST_LEN) % MWI_HIST_LEN;
    float peakVal = s_mwiHistory[peakIdx];
    if (peakVal < 0.00001f) return 999;

    float halfPeak = peakVal * 0.5f;

    int riseCount = 0;
    int scanIdx = (peakIdx - 1 + MWI_HIST_LEN) % MWI_HIST_LEN;

    while (riseCount < (MWI_HIST_LEN - 2)) {
        float val = s_mwiHistory[scanIdx];
        int nextIdx = (scanIdx - 1 + MWI_HIST_LEN) % MWI_HIST_LEN;
        float nextVal = s_mwiHistory[nextIdx];

        if (val <= halfPeak) {
            if (nextVal > halfPeak) {
                riseCount += 1;
            }
            break;
        }
        if (val > nextVal) {
            if (nextVal <= halfPeak) {
                riseCount += 1;
            }
        }

        riseCount++;
        scanIdx = nextIdx;
        if (scanIdx == peakIdx) break;
    }

    int fallIdx = (peakIdx + 1) % MWI_HIST_LEN;
    float fallVal = s_mwiHistory[fallIdx];

    int fallCount = 0;
    if (fallVal >= halfPeak) {
        int fIdx = fallIdx;
        while (fallCount < (MWI_HIST_LEN - 2)) {
            float v = s_mwiHistory[fIdx];
            if (v <= halfPeak) break;
            fallCount++;
            fIdx = (fIdx + 1) % MWI_HIST_LEN;
            if (fIdx == s_mwiHistIdx) break;
        }
    } else {
        float grad = peakVal - fallVal;
        if (grad > 0.00001f) {
            float fallTime = (peakVal - halfPeak) / grad;
            fallCount = (int)(fallTime + 0.5f);
            if (fallCount < 1) fallCount = 1;
        } else {
            fallCount = 1;
        }
    }

    int totalWidth = riseCount + fallCount + 1;
    return totalWidth;
}

static float getRiseFallRatio(void)
{
    int peakIdx = (s_mwiHistIdx - 2 + MWI_HIST_LEN) % MWI_HIST_LEN;
    float peakVal = s_mwiHistory[peakIdx];
    if (peakVal < 0.00001f) return 1.0f;

    float halfPeak = peakVal * 0.5f;

    int riseSamp = 0;
    int scanIdx = (peakIdx - 1 + MWI_HIST_LEN) % MWI_HIST_LEN;
    while (riseSamp < (MWI_HIST_LEN - 2)) {
        float val = s_mwiHistory[scanIdx];
        if (val <= halfPeak) break;
        riseSamp++;
        scanIdx = (scanIdx - 1 + MWI_HIST_LEN) % MWI_HIST_LEN;
        if (scanIdx == peakIdx) break;
    }

    int fallSamp = 0;
    int fIdx = (peakIdx + 1) % MWI_HIST_LEN;
    while (fallSamp < (MWI_HIST_LEN - 2)) {
        float val = s_mwiHistory[fIdx];
        if (val <= halfPeak) break;
        fallSamp++;
        fIdx = (fIdx + 1) % MWI_HIST_LEN;
        if (fIdx == s_mwiHistIdx) break;
    }
    if (fallSamp == 0) {
        int fallIdx = (peakIdx + 1) % MWI_HIST_LEN;
        float fallVal = s_mwiHistory[fallIdx];
        float grad = peakVal - fallVal;
        if (grad > 0.00001f) {
            float fallTime = (peakVal - halfPeak) / grad;
            fallSamp = (int)(fallTime + 0.5f);
            if (fallSamp < 1) fallSamp = 1;
        } else {
            fallSamp = 1;
        }
    }

    if (fallSamp <= 0) fallSamp = 1;
    if (riseSamp <= 0) riseSamp = 1;

    return (float)riseSamp / (float)fallSamp;
}

static bool isRRConsistent(float rrSec)
{
    if (s_motionConfirmed) return true;
    if (s_rrCount < 3) return true;
    if (s_medianRR < 0.001f) return true;
    float deviation = fabsf(rrSec - s_medianRR) / s_medianRR;
    return (deviation <= RR_CONSISTENCY);
}

static bool isQRSValid(float peakVal, float rrSec)
{
    if (!s_signalPresent)                 return false;
    if (s_state == HR_REFRACTORY)         return false;
    if (peakVal <= s_threshold)           return false;

    float peakRatio = s_motionConfirmed ? MIN_PEAK_RATIO_MOT : MIN_PEAK_RATIO;
    if (peakVal < s_noisePeak * peakRatio) return false;

    if (s_beatCount >= MIN_CONF_FEAT) {
        if (s_motionConfirmed) {
            int width = getQRSWidth();
            if (width < MIN_QRS_WIDTH_MOT || width > MAX_QRS_WIDTH_MOT) {
                return false;
            }
            float ratio = getRiseFallRatio();
            if (ratio < RISE_FALL_MIN_MOT || ratio > RISE_FALL_MAX_MOT) {
                return false;
            }
        } else {
            if (!isAmplitudeConsistent(peakVal)) return false;
            int width = getQRSWidth();
            if (width < MIN_QRS_WIDTH || width > MAX_QRS_WIDTH) return false;
            float ratio = getRiseFallRatio();
            if (ratio < RISE_FALL_MIN || ratio > RISE_FALL_MAX) return false;
            if (!isRRConsistent(rrSec)) return false;
        }
    }

    if (s_confirmedBPMCount >= BPM_REJECT_MIN_CNT) {
        float confirmedMed = getConfirmedBPMMedian();
        if (confirmedMed > 1.0f && rrSec > 0.001f) {
            float instBPM = 60.0f / rrSec;
            float dev = fabsf(instBPM - confirmedMed) / confirmedMed;
            float rejectThresh = s_motionConfirmed ? BPM_REJECT_DEV_MOT : BPM_REJECT_DEV;
            if (dev > rejectThresh) {
                return false;
            }
        }
    }

    return true;
}

static void checkSignalActivity(float filteredSample)
{
    float absVal = fabsf(filteredSample);
    if (absVal > s_winMaxAbs) {
        s_winMaxAbs = absVal;
    }
    s_winCount++;

    if (s_winCount >= ACT_WIN_SAMP) {
        if (s_winMaxAbs < ACT_THRESHOLD) {
            s_noSignalSeconds++;
        } else {
            s_noSignalSeconds = 0;
            if (!s_signalPresent) {
                s_signalPresent = true;
                hrReset();
                s_state = HR_LEARNING;
                s_signalPresent = true;
            }
        }

        if (s_noSignalSeconds >= ACT_TIMEOUT_CNT) {
            if (s_signalPresent) {
                s_signalPresent = false;
                hrReset();
                s_state = HR_LEARNING;
            }
        }

        s_winMaxAbs = 0.0f;
        s_winCount  = 0;
    }
}

static uint8_t computeOutputBPM(void)
{
    if (s_medianRR < 0.001f) return 0;

    float medBPM = 60.0f / s_medianRR;
    float bpm;

    if (s_motionConfirmed) {
        bpm = s_bpmEMA;
    } else if (s_bpmEmaFadeCnt > 0 && BPM_EMA_FADE_STEPS > 0) {
        float fadeFrac = (float)s_bpmEmaFadeCnt / (float)BPM_EMA_FADE_STEPS;
        bpm = fadeFrac * s_bpmEMA + (1.0f - fadeFrac) * medBPM;
    } else {
        bpm = medBPM;
    }

    if (s_lastOutputBPM > 1.0f) {
        float delta = bpm - s_lastOutputBPM;
        if (delta > BPM_SLEW_MAX) {
            bpm = s_lastOutputBPM + BPM_SLEW_MAX;
        } else if (delta < -BPM_SLEW_MAX) {
            bpm = s_lastOutputBPM - BPM_SLEW_MAX;
        }
    }

    s_lastOutputBPM = bpm;

    uint8_t bpmRaw = (uint8_t)(bpm + 0.5f);
    if (bpmRaw < 30 || bpmRaw > 200) return 0;
    return bpmRaw;
}

/* ======================== 公共 API ======================== */

void hrInit(void)
{
    hrFullReset();
    s_signalPeak     = THRESHOLD_INIT;
    s_noisePeak      = THRESHOLD_INIT * 0.3f;
    s_threshold      = THRESHOLD_INIT;
    s_sqi            = 0.5f;
    s_lastOutputBPM  = 0.0f;
}

HR_Result hrProcess(float filteredSample)
{
    HR_Result result = { 0 };

    checkSignalActivity(filteredSample);

    updateSQI();
    updateMotionState();

    float qrsSignal = applyQRSBandpass(filteredSample);

    if (!s_adaptInitDone && s_beatCount == 0) {
        s_adaptInitCount++;
        s_adaptInitSumSq += qrsSignal * qrsSignal;
        if (s_adaptInitCount >= ADAPT_INIT_SAMP) {
            float baselineRMS = sqrtf(s_adaptInitSumSq / (float)ADAPT_INIT_SAMP);
            float adaptiveThreshold = baselineRMS * ADAPT_INIT_FACTOR;
            if (adaptiveThreshold > THRESHOLD_INIT) {
                s_threshold = adaptiveThreshold;
                s_signalPeak = adaptiveThreshold;
                s_noisePeak = adaptiveThreshold * 0.3f;
            }
            s_adaptInitDone = true;
        }
    }

    float diff = qrsSignal - s_prevSample;
    s_prevSample = qrsSignal;
    float squared = diff * diff;
    float mwi = computeMWI(squared);

    s_mwiHistory[s_mwiHistIdx] = mwi;
    s_mwiHistIdx = (s_mwiHistIdx + 1) % MWI_HIST_LEN;

    bool isPeak = (s_mwiPrev > s_mwiPrevPrev) && (s_mwiPrev > mwi);

    if (isPeak) {
        float peakVal = s_mwiPrev;
        float rrSec   = (float)s_sampSinceBeat * TS;

        if (isQRSValid(peakVal, rrSec)) {
            addRRInterval(rrSec);
            updateThreshold(peakVal, true);
            recordValidPeak(peakVal);

            s_state = HR_REFRACTORY;
            s_refractCount = 0;
            s_sampSinceBeat = 0;
            s_beatCount++;

            result.beatDetected = true;
            result.beatCount    = s_beatCount;
            result.rrInterval   = s_lastRR;

            if (s_state == HR_LEARNING && s_beatCount >= MIN_CONF_BEATS) {
                s_state = HR_TRACKING;
            }

            if (s_state == HR_TRACKING && s_medianRR > 0.001f) {
                uint8_t bpmRaw = computeOutputBPM();
                if (bpmRaw >= 30 && bpmRaw <= 200) {
                    result.bpm = bpmRaw;
                    s_confirmedBPM[s_confirmedBPMIdx] = (float)bpmRaw;
                    s_confirmedBPMIdx = (s_confirmedBPMIdx + 1) % BPM_CONFIRMED_LEN;
                    if (s_confirmedBPMCount < BPM_CONFIRMED_LEN) {
                        s_confirmedBPMCount++;
                    }
                }
                float bufConf = (float)s_rrCount / (float)RR_BUFFER_SIZE;
                float sqiWeight = (s_sqi < 0.4f) ? (s_sqi / 0.4f) : 1.0f;
                float motionFactor = s_motionConfirmed ? 0.5f : 1.0f;
                result.confidence = fminf(1.0f, bufConf * sqiWeight * motionFactor);
            }

        } else {
            if (s_signalPresent && peakVal > s_noisePeak * 0.5f) {
                updateThreshold(peakVal, false);
            }
        }
    }

    s_sampSinceBeat++;

    if (s_state == HR_REFRACTORY) {
        s_refractCount++;
        if (s_refractCount >= REFRACTORY_SAMP) {
            s_state = (s_beatCount >= MIN_CONF_BEATS) ? HR_TRACKING : HR_IDLE;
        }
    }

    if (s_signalPresent && s_sampSinceBeat > TIMEOUT_SAMP) {
        hrReset();
        s_state = HR_LEARNING;
    }

    s_mwiPrevPrev = s_mwiPrev;
    s_mwiPrev     = mwi;

    if (!result.beatDetected && s_beatCount > 0 && s_medianRR > 0.001f
        && s_signalPresent && s_sampSinceBeat < HOLD_SAMP) {
        uint8_t bpmRaw = computeOutputBPM();
        if (bpmRaw >= 30 && bpmRaw <= 200) {
            result.bpm = bpmRaw;
        }
        float decay = 1.0f - (float)s_sampSinceBeat / (float)HOLD_SAMP;
        float bufConf = (float)s_rrCount / (float)RR_BUFFER_SIZE;
        float sqiWeight = (s_sqi < 0.4f) ? (s_sqi / 0.4f) : 1.0f;
        float motionFactor = s_motionConfirmed ? 0.5f : 1.0f;
        result.confidence = fminf(0.8f, bufConf * sqiWeight * motionFactor) * decay;
    }

    result.sqi          = s_sqi;
    result.motionActive = s_motionActive;

    return result;
}

void hrReset(void)
{
    s_prevSample = 0.0f;
    for (int i = 0; i < MWI_WINDOW; i++)   s_mwiBuf[i] = 0.0f;
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

    for (int i = 0; i < PEAK_HIST_LEN; i++) s_recentPeaks[i] = 0.0f;
    s_peakHistIdx   = 0;
    s_peakHistCount = 0;

    s_bpmEMA       = 0.0f;
    s_bpmEmaFadeCnt = 0;

    for (int i = 0; i < BPM_CONFIRMED_LEN; i++) s_confirmedBPM[i] = 0.0f;
    s_confirmedBPMIdx   = 0;
    s_confirmedBPMCount = 0;

    qrs_bpf_lp_w1 = 0.0f;
    qrs_bpf_lp_w2 = 0.0f;
    qrs_bpf_hp_w1 = 0.0f;
    qrs_bpf_hp_w2 = 0.0f;
    
    s_adaptInitDone = false;
    s_adaptInitCount = 0;
    s_adaptInitSumSq = 0.0f;
}

void hrFullReset(void)
{
    hrReset();

    s_signalPresent   = true;
    s_winMaxAbs       = 0.0f;
    s_winCount        = 0;
    s_noSignalSeconds = 0;

    s_sqi              = 0.5f;
    s_motionActive     = false;
    s_motionConfirmed  = false;
    s_motionLowCount   = 0;
    s_motionHighCount  = 0;
    s_motionRecoverCnt = 0;
    s_motionHoldSP     = 0.0f;
    s_motionHoldNP     = 0.0f;

    for (int i = 0; i < MWI_HIST_LEN; i++) s_mwiHistory[i] = 0.0f;
    s_mwiHistIdx = 0;

    for (int i = 0; i < BPM_CONFIRMED_LEN; i++) s_confirmedBPM[i] = 0.0f;
    s_confirmedBPMIdx   = 0;
    s_confirmedBPMCount = 0;

    s_lastOutputBPM = 0.0f;

    qrs_bpf_lp_w1 = 0.0f;
    qrs_bpf_lp_w2 = 0.0f;
    qrs_bpf_hp_w1 = 0.0f;
    qrs_bpf_hp_w2 = 0.0f;
}

float hrGetSQI(void)
{
    return s_sqi;
}

bool hrIsMotionActive(void)
{
    return s_motionActive;
}