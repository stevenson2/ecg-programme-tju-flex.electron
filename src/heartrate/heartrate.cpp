#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "heartrate/heartrate.h"

/**
 * @file heartrate.cpp
 * @brief 板上心率计算模块 - 实现 (v3.1)
 *
 * ========== 简化 Pan-Tompkins + 4 维形态学验证 ==========
 *
 * 经典 Pan-Tompkins 五步 (1985):
 *   ① 带通滤波 5~15Hz  → 信号已由 filter.cpp 预处理, 跳过
 *   ② 一阶差分          → 突出 QRS 的快速上升/下降沿
 *   ③ 逐点平方          → 放大 R 波能量, 使 T/P 波相对更小
 *   ④ 滑动窗口积分      → 150ms 矩形窗, 将 QRS 能量汇聚为单个峰
 *   ⑤ 自适应阈值检测    → 单阈值 + 200ms 不应期
 *
 * ========== v3.1 新增: 4 维 QRS 形态学验证 (利用边缘算力) ==========
 *   ① 振幅一致性  — 候选峰 vs 近期信号峰均值, 偏差 < 35%
 *   ② 脉宽约束    — MWI 峰半高宽 80~160ms (20~40 样本)
 *   ③ 上升/下降比 — 上升时间 / 下降时间 ∈ [0.5, 2.0]
 *   ④ RR 一致性   — 新 RR 与中位数偏差 < 30%
 *
 * 这 4 个特征可有效排除 T 波误检 (T 波更宽、不对称、
 * 振幅不一致) 和噪声尖峰 (极窄、振幅跳变)。
 *
 * ========== v3.0 保留特性 ==========
 *   - SQI (信号质量指数)
 *   - 运动检测: 滞回判定 + 阈值冻结
 *   - BPM 平滑: 运动恢复期 3 秒快速收敛
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
#define SIGNAL_WEIGHT_FAST 0.25f    /**< 运动恢复期快速收敛信号峰值 */

#define MIN_RR_SAMP     75          /**< 最小 RR: 300ms @250Hz */
#define MAX_RR_SAMP     500         /**< 最大 RR: 2000ms @250Hz */

#define TIMEOUT_SAMP    750         /**< 3 秒无 QRS → 复位 */
#define HOLD_SAMP       250         /**< 1 秒无新拍 → 停止输出旧 BPM */
#define MIN_CONF_BEATS  5           /**< 至少 5 拍才开始输出 BPM */
#define MIN_CONF_FEAT   8           /**< 至少 8 拍才开启特征验证 */
#define MIN_PEAK_RATIO  2.0f        /**< 峰/噪比门限 */

/* ======== 信号活动检测 ======== */
#define ACT_WIN_SAMP    250         /**< 活动检测窗口: 1秒 @250Hz */
#define ACT_THRESHOLD   0.005f      /**< 最小信号幅度 5mV */
#define ACT_TIMEOUT_CNT 2           /**< 连续 2 秒无信号→标记消失 */

/* ======== SQI 与运动检测 ======== */
#define SQI_EMA_WEIGHT  0.05f       /**< SQI 指数平滑因子 (慢) */
#define SQI_MOTION_ENTER 0.35f      /**< SQI 低于此值 → 进入运动状态 */
#define SQI_MOTION_EXIT  0.55f      /**< SQI 高于此值 → 退出运动状态 */
#define SQI_SNR_FLOOR    0.001f     /**< SNR 最小值, 防止除零 */
#define MOTION_BPM_HOLD  750        /**< 运动结束后保持旧 BPM 的帧数 (3秒) */

/* ======== v3.1 形态学验证常数 ======== */
#define MWI_HIST_LEN     60         /**< MWI 历史长度 (240ms @250Hz) */
#define PEAK_HIST_LEN    8          /**< 近期峰值历史长度 */
#define MIN_QRS_WIDTH    20         /**< 最小 QRS 半高宽 80ms */
#define MAX_QRS_WIDTH    40         /**< 最大 QRS 半高宽 160ms */
#define AMP_CONSISTENCY  0.35f      /**< 振幅一致性容差 ±35% */
#define RR_CONSISTENCY   0.30f      /**< RR 一致性容差 ±30% */
#define RISE_FALL_MIN    0.5f       /**< 最小上升/下降比 */
#define RISE_FALL_MAX    2.0f       /**< 最大上升/下降比 */

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

/* ======== 信号活动检测 (独立管理, hrReset 不触碰) ======== */
static bool       s_signalPresent;
static float      s_winMaxAbs;
static int        s_winCount;
static int        s_noSignalSeconds;

/* ======== SQI 与运动检测 (v3.0 新增) ======== */
static float      s_sqi;
static bool       s_motionActive;
static bool       s_motionConfirmed;
static int        s_motionLowCount;
static int        s_motionHighCount;
static int        s_motionRecoverCnt;
static float      s_motionHoldSP;
static float      s_motionHoldNP;

/* ======== v3.1 形态学验证状态 ======== */
static float      s_mwiHistory[MWI_HIST_LEN]; /**< 环形缓冲, 存储最近 60 个 MWI 值 */
static int        s_mwiHistIdx;               /**< 当前写入位置 (即将写入) */
static float      s_recentPeaks[PEAK_HIST_LEN]; /**< 近期验证通过的 QRS 峰值 */
static int        s_peakHistIdx;              /**< 当前写入位置 */
static int        s_peakHistCount;            /**< 已记录峰值数 */

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
    if (s_motionConfirmed && isSignal) {
        return;
    }

    if (isSignal) {
        float weight = (s_motionRecoverCnt > 0)
                       ? SIGNAL_WEIGHT_FAST : SIGNAL_WEIGHT;
        s_signalPeak = weight * peakVal
                     + (1.0f - weight) * s_signalPeak;
    } else {
        float nWeight = s_motionConfirmed ? (NOISE_WEIGHT * 1.5f) : NOISE_WEIGHT;
        if (nWeight > 0.25f) nWeight = 0.25f;
        s_noisePeak = nWeight * peakVal
                    + (1.0f - nWeight) * s_noisePeak;
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
        if (s_motionLowCount >= 250) {
            s_motionConfirmed = true;
            s_motionActive = true;
            s_motionLowCount = 0;
            s_motionHoldSP = s_signalPeak;
            s_motionHoldNP = s_noisePeak;
            s_motionRecoverCnt = 0;
        }
    } else {
        if (s_sqi > SQI_MOTION_EXIT) {
            s_motionHighCount++;
            s_motionLowCount = 0;
        } else {
            s_motionHighCount = 0;
        }
        if (s_motionHighCount >= 125) {
            s_motionConfirmed = false;
            s_motionActive = false;
            s_motionHighCount = 0;
            s_motionRecoverCnt = MOTION_BPM_HOLD;
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
    if (s_motionConfirmed) return;

    s_rrBuf[s_rrIdx] = rrSeconds;
    s_rrIdx = (s_rrIdx + 1) % RR_BUFFER_SIZE;
    if (s_rrCount < RR_BUFFER_SIZE) s_rrCount++;

    s_medianRR = computeMedianRR();
    s_lastRR = rrSeconds;
}

/**
 * @brief 记录已验证的 QRS 峰值 → 用于振幅一致性校验
 */
static void recordValidPeak(float peakVal)
{
    s_recentPeaks[s_peakHistIdx] = peakVal;
    s_peakHistIdx = (s_peakHistIdx + 1) % PEAK_HIST_LEN;
    if (s_peakHistCount < PEAK_HIST_LEN) {
        s_peakHistCount++;
    }
}

/**
 * @brief 校验 ① 振幅一致性: 候选峰 vs 近期 QRS 峰均值
 *
 * 若候选峰偏离均值 > 35%, 很可能是 T 波或伪迹。
 *
 * @param peakVal 候选峰值
 * @return true 通过校验
 */
static bool isAmplitudeConsistent(float peakVal)
{
    if (s_peakHistCount < 3) return true;  /* 历史不足, 放行 */

    float sum = 0.0f;
    for (int i = 0; i < s_peakHistCount; i++) {
        sum += s_recentPeaks[i];
    }
    float mean = sum / (float)s_peakHistCount;
    if (mean < 0.0001f) return true;

    float deviation = fabsf(peakVal - mean) / mean;
    return (deviation <= AMP_CONSISTENCY);
}

/**
 * @brief 校验 ② 脉宽 (半高宽): 80~160ms (20~40 样本 @250Hz)
 *
 * 在 MWI 历史缓冲中逆序扫描找到上升沿 50% 穿越点,
 * 利用当前 mwi 斜率外推下降沿 50% 点,
 * 求和即得半高宽估计。
 *
 * T 波典型半高宽 > 50 样本 (>200ms),
 * 噪声尖峰 < 10 样本 (<40ms),
 * QRS 落在 20~40 样本窄窗内。
 *
 * @return 脉宽样本数, 超出范围返回 999
 */
static int getQRSWidth(void)
{
    /* 峰位于 s_mwiPrev (即历史缓冲中 s_mwiHistIdx-2 位置) */
    int peakIdx = (s_mwiHistIdx - 2 + MWI_HIST_LEN) % MWI_HIST_LEN;
    float peakVal = s_mwiHistory[peakIdx];
    if (peakVal < 0.00001f) return 999;

    float halfPeak = peakVal * 0.5f;

    /* ---- 上升沿: 从峰前 1 位置向更早方向扫描 ---- */
    int riseCount = 0;
    int scanIdx = (peakIdx - 1 + MWI_HIST_LEN) % MWI_HIST_LEN;

    while (riseCount < (MWI_HIST_LEN - 2)) {
        float val = s_mwiHistory[scanIdx];
        int nextIdx = (scanIdx - 1 + MWI_HIST_LEN) % MWI_HIST_LEN;
        float nextVal = s_mwiHistory[nextIdx];

        if (val <= halfPeak) {
            /* 线性插值: 精确 50% 穿越点 */
            if (nextVal > halfPeak) {
                float frac = (halfPeak - val) / (nextVal - val + 0.00001f);
                riseCount += 1;  /* 保守取整 */
            }
            break;
        }
        if (val > nextVal) {
            /* 穿越发生在 val 与 nextVal 之间 */
            if (nextVal <= halfPeak) {
                float frac = (val - halfPeak) / (val - nextVal + 0.00001f);
                riseCount += 1;  /* 不足 1 样本 */
            }
        }

        riseCount++;
        scanIdx = nextIdx;
        if (scanIdx == peakIdx) break;  /* 兜底 */
    }

    /* ---- 下降沿: 用当前 mwi (峰后 1 样本) 线性外推 ---- */
    int fallIdx = (peakIdx + 1) % MWI_HIST_LEN;
    float fallVal = s_mwiHistory[fallIdx];

    int fallCount = 0;
    if (fallVal >= halfPeak) {
        /* 1 个样本后仍未降至半高 → 宽脉冲, 保守扫后续历史 */
        int fIdx = fallIdx;
        while (fallCount < (MWI_HIST_LEN - 2)) {
            float v = s_mwiHistory[fIdx];
            if (v <= halfPeak) break;
            fallCount++;
            fIdx = (fIdx + 1) % MWI_HIST_LEN;
            if (fIdx == s_mwiHistIdx) break;
        }
        /* 插值修正 */
        if (fallCount > 0 && fallCount < MWI_HIST_LEN) {
            int f2Idx = (fIdx + 1) % MWI_HIST_LEN;
            int f1Idx = (fIdx - 1 + MWI_HIST_LEN) % MWI_HIST_LEN;
            float v1 = s_mwiHistory[f1Idx];
            float v2 = s_mwiHistory[f2Idx];
            /* 简化: 不插值, 直接用计数 */
        }
    } else {
        /* 峰后 1 样本已降至半高以下 → 窄脉冲 */
        float grad = peakVal - fallVal;  /* 正数, 峰>fall */
        if (grad > 0.00001f) {
            float fallTime = (peakVal - halfPeak) / grad;
            fallCount = (int)(fallTime + 0.5f);
            if (fallCount < 1) fallCount = 1;
        } else {
            fallCount = 1;
        }
    }

    int totalWidth = riseCount + fallCount + 1;  /* +1 为峰本身 */
    return totalWidth;
}

/**
 * @brief 校验 ③ 上升/下降对称性: 上升时间 / 下降时间 ∈ [0.5, 2.0]
 *
 * QRS 波上升与下降时间大致对称,
 * T 波上升缓慢下降快, 噪声尖峰上升快下降也快。
 *
 * @return 上升/下降比, 无法计算返回 1.0
 */
static float getRiseFallRatio(void)
{
    int peakIdx = (s_mwiHistIdx - 2 + MWI_HIST_LEN) % MWI_HIST_LEN;
    float peakVal = s_mwiHistory[peakIdx];
    if (peakVal < 0.00001f) return 1.0f;

    float halfPeak = peakVal * 0.5f;

    /* 上升时间: 从峰前 1 位置向更早方向扫描, 找到 50% 穿越点 */
    int riseSamp = 0;
    int scanIdx = (peakIdx - 1 + MWI_HIST_LEN) % MWI_HIST_LEN;
    while (riseSamp < (MWI_HIST_LEN - 2)) {
        float val = s_mwiHistory[scanIdx];
        if (val <= halfPeak) break;
        riseSamp++;
        scanIdx = (scanIdx - 1 + MWI_HIST_LEN) % MWI_HIST_LEN;
        if (scanIdx == peakIdx) break;
    }

    /* 下降时间: 从峰开始扫描后续直到低于半高 */
    int fallSamp = 0;
    int fIdx = (peakIdx + 1) % MWI_HIST_LEN;
    while (fallSamp < (MWI_HIST_LEN - 2)) {
        float val = s_mwiHistory[fIdx];
        if (val <= halfPeak) break;
        fallSamp++;
        fIdx = (fIdx + 1) % MWI_HIST_LEN;
        if (fIdx == s_mwiHistIdx) break;
    }
    /* 如果当前 MWI 低于半高, 用梯度补正 */
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

/**
 * @brief 校验 ④ RR 一致性: 新 RR 与中位数偏差 < 30%
 *
 * 异常 RR (如 T 波导致的半周期) 会被此过滤器拒绝。
 * 至少需要 3 个历史 RR 才会生效。
 *
 * @param rrSec 候选 RR 间期 (秒)
 * @return true 通过校验
 */
static bool isRRConsistent(float rrSec)
{
    if (s_rrCount < 3) return true;
    if (s_medianRR < 0.001f) return true;

    float deviation = fabsf(rrSec - s_medianRR) / s_medianRR;
    return (deviation <= RR_CONSISTENCY);
}

/**
 * @brief 综合 QRS 有效性判定 (v3.1)
 *
 * 基础条件: 信号存在 + 非不应期 + 超阈值 + 峰噪比足够
 * 特征验证: 振幅一致性 + 脉宽 + 对称性 + RR 一致性 (历史充足时)
 *
 * @param peakVal  候选峰值
 * @param rrSec    候选 RR 间期 (秒)
 * @return true 认定为 QRS 波
 */
static bool isQRSValid(float peakVal, float rrSec)
{
    if (!s_signalPresent)                 return false;
    if (s_state == HR_REFRACTORY)         return false;
    if (peakVal <= s_threshold)           return false;
    if (peakVal < s_noisePeak * MIN_PEAK_RATIO) return false;

    /* 运动期间提高峰噪比要求 */
    if (s_motionConfirmed && peakVal < s_noisePeak * (MIN_PEAK_RATIO * 1.5f))
        return false;

    /* 特征验证: 需要至少 MIN_CONF_FEAT 拍历史 */
    if (s_beatCount >= MIN_CONF_FEAT) {
        /* ① 振幅一致性 */
        if (!isAmplitudeConsistent(peakVal)) {
            return false;
        }

        /* ② 脉宽约束 */
        int width = getQRSWidth();
        if (width < MIN_QRS_WIDTH || width > MAX_QRS_WIDTH) {
            return false;
        }

        /* ③ 上升/下降对称性 */
        float ratio = getRiseFallRatio();
        if (ratio < RISE_FALL_MIN || ratio > RISE_FALL_MAX) {
            return false;
        }

        /* ④ RR 一致性 (历史充足时) */
        if (!isRRConsistent(rrSec)) {
            return false;
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

/* ======================== 公共 API ======================== */

void hrInit(void)
{
    hrFullReset();
    s_signalPeak     = THRESHOLD_INIT;
    s_noisePeak      = THRESHOLD_INIT * 0.3f;
    s_threshold      = THRESHOLD_INIT;
    s_sqi            = 0.5f;
}

HR_Result hrProcess(float filteredSample)
{
    HR_Result result = { 0 };

    /* 步骤 0: 信号活动检测 */
    checkSignalActivity(filteredSample);

    /* 步骤 0.5: SQI 更新 + 运动状态 */
    updateSQI();
    updateMotionState();

    /* 步骤 1-3: 差分 → 平方 → 滑动积分 */
    float diff = filteredSample - s_prevSample;
    s_prevSample = filteredSample;
    float squared = diff * diff;
    float mwi = computeMWI(squared);

    /* 存入 MWI 历史缓冲 (供形态学验证回扫) */
    s_mwiHistory[s_mwiHistIdx] = mwi;
    s_mwiHistIdx = (s_mwiHistIdx + 1) % MWI_HIST_LEN;

    /* 步骤 4: 峰值检测 (MWI 局部最大值) */
    bool isPeak = (s_mwiPrev > s_mwiPrevPrev) && (s_mwiPrev > mwi);

    if (isPeak) {
        float peakVal = s_mwiPrev;
        float rrSec   = (float)s_sampSinceBeat * TS;

        if (isQRSValid(peakVal, rrSec)) {
            addRRInterval(rrSec);
            updateThreshold(peakVal, true);
            recordValidPeak(peakVal);  /* 记录通过验证的峰值 */

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
                if (!s_motionConfirmed) {
                    uint8_t bpmRaw = (uint8_t)(60.0f / s_medianRR + 0.5f);
                    if (bpmRaw >= 30 && bpmRaw <= 200) {
                        result.bpm = bpmRaw;
                    }
                }
                float bufConf = (float)s_rrCount / (float)RR_BUFFER_SIZE;
                float sqiWeight = (s_sqi < 0.4f) ? (s_sqi / 0.4f) : 1.0f;
                result.confidence = fminf(1.0f, bufConf * sqiWeight);
            }

        } else {
            if (s_signalPresent && peakVal > s_noisePeak * 0.5f) {
                updateThreshold(peakVal, false);
            }
        }
    }

    /* 步骤 5: 状态机 */
    s_sampSinceBeat++;

    if (s_state == HR_REFRACTORY) {
        s_refractCount++;
        if (s_refractCount >= REFRACTORY_SAMP) {
            s_state = (s_beatCount >= MIN_CONF_BEATS)
                      ? HR_TRACKING : HR_IDLE;
        }
    }

    /* 步骤 6: 超时复位 */
    if (s_signalPresent && s_sampSinceBeat > TIMEOUT_SAMP) {
        hrReset();
        s_state = HR_LEARNING;
    }

    /* 步骤 7: 滚动 MWI 历史 */
    s_mwiPrevPrev = s_mwiPrev;
    s_mwiPrev     = mwi;

    /* 步骤 8: 无新拍时保持 */
    if (!result.beatDetected && s_beatCount > 0 && s_medianRR > 0.001f
        && s_signalPresent && s_sampSinceBeat < HOLD_SAMP
        && !s_motionConfirmed) {
        uint8_t bpmRaw = (uint8_t)(60.0f / s_medianRR + 0.5f);
        if (bpmRaw >= 30 && bpmRaw <= 200) {
            result.bpm = bpmRaw;
        }
        float decay = 1.0f - (float)s_sampSinceBeat / (float)HOLD_SAMP;
        float bufConf = (float)s_rrCount / (float)RR_BUFFER_SIZE;
        float sqiWeight = (s_sqi < 0.4f) ? (s_sqi / 0.4f) : 1.0f;
        result.confidence = fminf(0.8f, bufConf * sqiWeight) * decay;
    }

    /* 步骤 9: 填充 SQI 与运动状态 */
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

    /* v3.1: 峰值历史也需重置 (避免新旧混合) */
    for (int i = 0; i < PEAK_HIST_LEN; i++) s_recentPeaks[i] = 0.0f;
    s_peakHistIdx   = 0;
    s_peakHistCount = 0;

    /* 不重置 SQI/运动/MWI 历史, 保留连续性 */
}

void hrFullReset(void)
{
    hrReset();

    s_signalPresent   = true;
    s_winMaxAbs       = 0.0f;
    s_winCount        = 0;
    s_noSignalSeconds = 0;

    /* 运动检测完全重置 */
    s_sqi              = 0.5f;
    s_motionActive     = false;
    s_motionConfirmed  = false;
    s_motionLowCount   = 0;
    s_motionHighCount  = 0;
    s_motionRecoverCnt = 0;
    s_motionHoldSP     = 0.0f;
    s_motionHoldNP     = 0.0f;

    /* MWI 历史清零 */
    for (int i = 0; i < MWI_HIST_LEN; i++) s_mwiHistory[i] = 0.0f;
    s_mwiHistIdx = 0;
}

float hrGetSQI(void)
{
    return s_sqi;
}

bool hrIsMotionActive(void)
{
    return s_motionActive;
}