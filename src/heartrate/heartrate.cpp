#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "heartrate/heartrate.h"

/**
 * @file heartrate.cpp
 * @brief 板上心率计算模块 - 实现 (v3.4)
 *
 * ========== 简化 Pan-Tompkins + 4 维形态学验证 + 记录纠错 ==========
 *
 * 经典 Pan-Tompkins 五步 (1985):
 *   ① 带通滤波 5~15Hz  → 信号已由 filter.cpp 预处理, 跳过
 *   ② 一阶差分          → 突出 QRS 的快速上升/下降沿
 *   ③ 逐点平方          → 放大 R 波能量, 使 T/P 波相对更小
 *   ④ 滑动窗口积分      → 150ms 矩形窗, 将 QRS 能量汇聚为单个峰
 *   ⑤ 自适应阈值检测    → 单阈值 + 200ms 不应期
 *
 * ========== v3.4 新增: BPM 记录+纠错 ==========
 *
 *   v3.3 的 EMA 异常降权和 slew rate 是"软纠错" — 仍让异常 RR 进入
 *   缓冲区污染中位数。v3.4 采用"硬纠错": 在 isQRSValid 层直接拒绝。
 *
 *   记录阶段:
 *     - 维护近期确认输出 BPM 环形缓冲 (5 个值)
 *     - 每拍输出后记录, 取中位数作为"可信 BPM"
 *
 *   纠错阶段:
 *     - instBPM 偏离确认中位数 >30% (运动 45%) → 拒绝该拍
 *     - 拒绝后不污染 RR 缓冲区、不触发阈值更新
 *     - 只当噪声更新阈值 (若峰值够大)
 *
 * ========== v3.3 修复: 运动锁死 + BPM 跃升 ==========
 *
 *   锁死根因分析:
 *     v3.2 虽移除了 addRRInterval 阻断, 但运动期 isQRSValid 仍有:
 *       - 峰噪比 3.0x (静止 2.0x)
 *       - noisePeak 1.5x 加速增长
 *       - signalPeak 完全冻结
 *     → 运动几秒后 noisePeak 超过 signalPeak, 所有 QRS 被峰噪比拒绝
 *     → 且 isRRConsistent 用陈旧中位数拒绝所有新 RR
 *
 *   跃升根因分析:
 *     - BPM EMA 权重 0.3 无条件使用
 *     - T 波误检产生极短 RR → instBPM = 150+ → EMA 快速攀升
 *
 *   修复 (3 项):
 *     ① 运动期阈值策略重写:
 *        - signalPeak: 极慢更新 0.02 (代替完全冻结)
 *        - noisePeak:  正常权重 0.125 (移除 1.5x 加速)
 *        - 峰噪比:     运动期降低到 1.5x (易检测, 不怕误检)
 *     ② 运动期跳过 RR 一致性检查 → 新 RR 畅通进入缓冲区
 *     ③ BPM 跃升防护:
 *        - instBPM 偏离中位数 > 40% → EMA 权重降到 0.05
 *        - 输出 BPM 变化速率限制: 最大 ±2 BPM/拍
 *
 * ========== v3.2 保留特性 ==========
 *   - BPM EMA 快速跟踪 (运动/恢复期)
 *   - 运动期形态学约束放宽 (脉宽/对称性)
 *   - 恢复期 EMA→中位数线性渐变
 *
 * ========== v3.1 保留特性 ==========
 *   - 4 维 QRS 形态学验证 (振幅/脉宽/对称性/RR 一致性)
 *   - 峰值历史 + MWI 历史缓冲
 *
 * ========== v3.0 保留特性 ==========
 *   - SQI (信号质量指数)
 *   - 运动检测: 滞回判定
 *   - BPM 平滑: 运动恢复期 3 秒快速收敛
 */

/* ======================== 算法常数 ======================== */
#define FS              250.0f      /**< 采样率 (Hz) */
#define TS              0.004f      /**< 采样间隔 (s) */

#define MWI_WINDOW      38          /**< 滑动积分窗口 150ms @250Hz */
#define REFRACTORY_SAMP 50          /**< 不应期 200ms = 50 样本 */

#define RR_BUFFER_SIZE  8           /**< BPM 中位数缓冲区容量 */

#define THRESHOLD_INIT  0.002f      /**< 初始阈值 */
#define THRESHOLD_RATIO 0.40f       /**< 阈值 = 噪声 + 0.40×(信号−噪声) */
#define SIGNAL_WEIGHT   0.125f      /**< 信号峰值更新因子 (EMA) */
#define NOISE_WEIGHT    0.125f      /**< 噪声峰值更新因子 (EMA) */
#define SIGNAL_WEIGHT_FAST 0.25f    /**< 运动恢复期快速收敛信号峰值 */
#define SIGNAL_WEIGHT_MOT  0.02f    /**< v3.3: 运动期极慢更新 signalPeak */

#define MIN_RR_SAMP     75          /**< 最小 RR: 300ms @250Hz */
#define MAX_RR_SAMP     500         /**< 最大 RR: 2000ms @250Hz */

#define TIMEOUT_SAMP    750         /**< 3 秒无 QRS → 复位 */
#define HOLD_SAMP       250         /**< 1 秒无新拍 → 停止输出旧 BPM */
#define MIN_CONF_BEATS  5           /**< 至少 5 拍才开始输出 BPM */
#define MIN_CONF_FEAT   8           /**< 至少 8 拍才开启特征验证 */
#define MIN_PEAK_RATIO  2.0f        /**< 峰/噪比门限 (静止) */
#define MIN_PEAK_RATIO_MOT 1.5f     /**< v3.3: 运动期峰噪比 (降低, 易检测) */

/* ======== 信号活动检测 ======== */
#define ACT_WIN_SAMP    250         /**< 活动检测窗口: 1秒 @250Hz */
#define ACT_THRESHOLD   0.005f      /**< 最小信号幅度 5mV */
#define ACT_TIMEOUT_CNT 2           /**< 连续 2 秒无信号→标记消失 */

/* ======== SQI 与运动检测 ======== */
#define SQI_EMA_WEIGHT  0.05f       /**< SQI 指数平滑因子 (慢) */
#define SQI_MOTION_ENTER 0.35f      /**< SQI 低于此值 → 进入运动状态 */
#define SQI_MOTION_EXIT  0.55f      /**< SQI 高于此值 → 退出运动状态 */
#define SQI_SNR_FLOOR    0.001f     /**< SNR 最小值, 防止除零 */
#define MOTION_BPM_HOLD  750        /**< 运动结束后保持峰值冻结的帧数 (3秒) */

/* ======== v3.1 形态学验证常数 (静止) ======== */
#define MWI_HIST_LEN     60         /**< MWI 历史长度 (240ms @250Hz) */
#define PEAK_HIST_LEN    8          /**< 近期峰值历史长度 */
#define MIN_QRS_WIDTH    20         /**< 最小 QRS 半高宽 80ms (静止) */
#define MAX_QRS_WIDTH    40         /**< 最大 QRS 半高宽 160ms (静止) */
#define AMP_CONSISTENCY  0.35f      /**< 振幅一致性容差 ±35% */
#define RR_CONSISTENCY   0.30f      /**< RR 一致性容差 ±30% (静止) */
#define RISE_FALL_MIN    0.5f       /**< 最小上升/下降比 */
#define RISE_FALL_MAX    2.0f       /**< 最大上升/下降比 */

/* ======== v3.2 运动期放宽约束 ======== */
#define MIN_QRS_WIDTH_MOT  15       /**< 运动期最小 QRS 半高宽 60ms */
#define MAX_QRS_WIDTH_MOT  50       /**< 运动期最大 QRS 半高宽 200ms */
#define RISE_FALL_MIN_MOT  0.35f    /**< 运动期最小上升/下降比 */
#define RISE_FALL_MAX_MOT  2.5f     /**< 运动期最大上升/下降比 */

/* ======== v3.2 BPM EMA 平滑 ======== */
#define BPM_EMA_WEIGHT_FAST 0.30f   /**< 运动/恢复期 BPM EMA 权重 (快速跟踪) */
#define BPM_EMA_WEIGHT_SLOW 0.10f   /**< 静止期 BPM EMA 权重 (平滑) */
#define BPM_EMA_FADE_STEPS  500     /**< 恢复期 EMA→中位数渐进步数 (2秒) */

/* ======== v3.3 BPM 跃升防护 ======== */
#define BPM_EMA_WEIGHT_ANOM 0.05f   /**< 异常 instBPM (偏离中位>40%) EMA 权重 */
#define BPM_ANOMALY_THRESH  0.40f   /**< instBPM 偏离中位>40% 视为异常 */
#define BPM_SLEW_MAX        3.0f    /**< 输出 BPM 每次最大变化 ±3 BPM/拍 */

/* ======== v3.4 BPM 记录+纠错 ======== */
#define BPM_CONFIRMED_LEN   5       /**< 近期确认 BPM 历史长度 */
#define BPM_REJECT_DEV      0.30f   /**< instBPM 偏离确认中位数>30% → 拒绝该拍 */
#define BPM_REJECT_DEV_MOT  0.45f   /**< 运动期放宽到 45% */
#define BPM_REJECT_MIN_CNT  3       /**< 至少 3 个确认值才开始纠错 */

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

/* ======== v3.2 BPM EMA 平滑状态 ======== */
static float      s_bpmEMA;        /**< EMA 平滑后的 BPM (快速跟踪) */
static int        s_bpmEmaFadeCnt; /**< 恢复期 EMA→中位数渐进步数 */

/* ======== v3.3 BPM 跃升防护状态 ======== */
static float      s_lastOutputBPM; /**< 上一帧输出 BPM (用于 slew rate) */

/* ======== v3.4 BPM 记录+纠错状态 ======== */
static float      s_confirmedBPM[BPM_CONFIRMED_LEN]; /**< 近期确认输出 BPM 环形缓冲 */
static int        s_confirmedBPMIdx;                 /**< 当前写入位置 */
static int        s_confirmedBPMCount;               /**< 已记录数量 */

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
        /* v3.3: 运动期 signalPeak 极慢更新 (0.02), 防止阈值完全崩溃 */
        /* 恢复期快速收敛 (0.25) */
        float weight;

        if (s_motionConfirmed) {
            weight = SIGNAL_WEIGHT_MOT;    /* 0.02: 极慢, 跟踪趋势 */
        } else if (s_motionRecoverCnt > 0) {
            weight = SIGNAL_WEIGHT_FAST;   /* 0.25: 快速恢复 */
        } else {
            weight = SIGNAL_WEIGHT;        /* 0.125: 正常 */
        }

        s_signalPeak = weight * peakVal
                     + (1.0f - weight) * s_signalPeak;
    } else {
        /* v3.3: 运动期噪声权重不加速, 防止 noisePeak 失控 */
        float nWeight = NOISE_WEIGHT;
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
            s_bpmEmaFadeCnt = 0;
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
            s_bpmEmaFadeCnt = BPM_EMA_FADE_STEPS;
            /* 恢复冻结前的峰值 (防止阈值已漂移) */
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

/**
 * @brief 计算确认 BPM 历史的中位数 (v3.4)
 */
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

/**
 * @brief 记录 RR 间期到缓冲区 + BPM EMA (v3.3: 跃升防护)
 *
 * v3.3 BPM 跃升防护:
 *   - instBPM 偏离中位数 > 40% → EMA 降权到 0.05
 *   - 防止 T 波误检的极短 RR 污染 EMA
 */
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

    /* v3.3: BPM EMA — 跃升防护 */
    float instBPM = 60.0f / rrSeconds;

    if (s_bpmEMA < 1.0f) {
        s_bpmEMA = instBPM;
    } else {
        /* 检测 instBPM 是否异常偏离 (vs 中位数) */
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
            /* 异常骤变: 极低权重, 避免 EMA 被污染 */
            weight = BPM_EMA_WEIGHT_ANOM;
        } else if (s_motionConfirmed || s_motionRecoverCnt > 0) {
            weight = BPM_EMA_WEIGHT_FAST;
        } else {
            weight = BPM_EMA_WEIGHT_SLOW;
        }

        s_bpmEMA = weight * instBPM + (1.0f - weight) * s_bpmEMA;
    }
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
 * @brief 校验 ② 脉宽 (半高宽): 静止 80~160ms, 运动 60~200ms @250Hz
 *
 * 在 MWI 历史缓冲中逆序扫描找到上升沿 50% 穿越点,
 * 利用当前 mwi 斜率外推下降沿 50% 点,
 * 求和即得半高宽估计。
 *
 * @return 脉宽样本数, 超出范围返回 999
 */
static int getQRSWidth(void)
{
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
            if (nextVal > halfPeak) {
                float frac = (halfPeak - val) / (nextVal - val + 0.00001f);
                riseCount += 1;
            }
            break;
        }
        if (val > nextVal) {
            if (nextVal <= halfPeak) {
                float frac = (val - halfPeak) / (val - nextVal + 0.00001f);
                riseCount += 1;
            }
        }

        riseCount++;
        scanIdx = nextIdx;
        if (scanIdx == peakIdx) break;
    }

    /* ---- 下降沿: 用当前 mwi (峰后 1 样本) 线性外推 ---- */
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

/**
 * @brief 校验 ③ 上升/下降对称性
 *
 * @return 上升/下降比, 无法计算返回 1.0
 */
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

/**
 * @brief 校验 ④ RR 一致性 (v3.3: 运动期完全跳过)
 *
 * 静止期: 新 RR 与中位数偏差 < 30%
 * 运动期: 完全跳过 (运动 RR 变化是正常生理反应, 不应拒绝)
 *
 * @param rrSec 候选 RR 间期 (秒)
 * @return true 通过校验
 */
static bool isRRConsistent(float rrSec)
{
    /* v3.3: 运动期完全跳过 RR 一致性检查 */
    if (s_motionConfirmed) return true;

    if (s_rrCount < 3) return true;
    if (s_medianRR < 0.001f) return true;

    float deviation = fabsf(rrSec - s_medianRR) / s_medianRR;
    return (deviation <= RR_CONSISTENCY);
}

/**
 * @brief 综合 QRS 有效性判定 (v3.4)
 *
 * v3.3 修复:
 *   - 运动期峰噪比降到 1.5x (防止阈值崩溃导致全拒)
 *   - 运动期跳过 RR 一致性检查
 *
 * v3.4 新增:
 *   - BPM 记录+纠错: instBPM 偏离确认历史中位数 >30% → 拒绝该拍
 *   - 防止 T 波误检的短 RR 进入缓冲区污染中位数
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

    /* 峰噪比: 运动期降低门限, 易检测 */
    float peakRatio = s_motionConfirmed ? MIN_PEAK_RATIO_MOT : MIN_PEAK_RATIO;
    if (peakVal < s_noisePeak * peakRatio) return false;

    /* 形态学验证: 需要至少 MIN_CONF_FEAT 拍历史 */
    if (s_beatCount >= MIN_CONF_FEAT) {
        if (s_motionConfirmed) {
            /* === v3.3: 运动期放宽约束 === */
            /* 跳过振幅一致性 (运动 QRS 振幅波动大) */

            /* 脉宽: 放宽到 60~200ms */
            int width = getQRSWidth();
            if (width < MIN_QRS_WIDTH_MOT || width > MAX_QRS_WIDTH_MOT) {
                return false;
            }

            /* 上升/下降比: 放宽 */
            float ratio = getRiseFallRatio();
            if (ratio < RISE_FALL_MIN_MOT || ratio > RISE_FALL_MAX_MOT) {
                return false;
            }

            /* v3.3: 运动期跳过 RR 一致性 */
            /* (isRRConsistent 自动返回 true) */
        } else {
            /* === 静止期: 严格 4 维验证 === */
            if (!isAmplitudeConsistent(peakVal)) return false;

            int width = getQRSWidth();
            if (width < MIN_QRS_WIDTH || width > MAX_QRS_WIDTH) return false;

            float ratio = getRiseFallRatio();
            if (ratio < RISE_FALL_MIN || ratio > RISE_FALL_MAX) return false;

            if (!isRRConsistent(rrSec)) return false;
        }
    }

    /* v3.4: BPM 记录+纠错 — instBPM 偏离确认历史中位数过多 → 拒绝 */
    if (s_confirmedBPMCount >= BPM_REJECT_MIN_CNT) {
        float confirmedMed = getConfirmedBPMMedian();
        if (confirmedMed > 1.0f && rrSec > 0.001f) {
            float instBPM = 60.0f / rrSec;
            float dev = fabsf(instBPM - confirmedMed) / confirmedMed;
            float rejectThresh = s_motionConfirmed
                               ? BPM_REJECT_DEV_MOT : BPM_REJECT_DEV;
            if (dev > rejectThresh) {
                return false;  /* T 波误检/伪迹 → 拒绝, 不污染 RR 缓冲 */
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

/**
 * @brief 计算输出 BPM (v3.3: 添加跃升防护 slew rate)
 *
 * 运动/恢复期: EMA 快速跟踪
 * 静止期: 中位数抗噪
 * v3.3 新增: 输出 BPM 变化速率限制 (±3 BPM/拍)
 */
static uint8_t computeOutputBPM(void)
{
    if (s_medianRR < 0.001f) return 0;

    float medBPM = 60.0f / s_medianRR;
    float bpm;

    if (s_motionConfirmed) {
        /* 运动期: 纯 EMA */
        bpm = s_bpmEMA;
    } else if (s_bpmEmaFadeCnt > 0 && BPM_EMA_FADE_STEPS > 0) {
        /* 恢复期: EMA → 中位数 线性渐变 */
        float fadeFrac = (float)s_bpmEmaFadeCnt / (float)BPM_EMA_FADE_STEPS;
        bpm = fadeFrac * s_bpmEMA + (1.0f - fadeFrac) * medBPM;
    } else {
        /* 静止期: 纯中位数 */
        bpm = medBPM;
    }

    /* v3.3: 跃升防护 — slew rate limit (±3 BPM/拍) */
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
                    /* v3.4: 记录确认 BPM 到纠错历史 */
                    s_confirmedBPM[s_confirmedBPMIdx] = (float)bpmRaw;
                    s_confirmedBPMIdx = (s_confirmedBPMIdx + 1)
                                      % BPM_CONFIRMED_LEN;
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

    for (int i = 0; i < PEAK_HIST_LEN; i++) s_recentPeaks[i] = 0.0f;
    s_peakHistIdx   = 0;
    s_peakHistCount = 0;

    /* v3.2: BPM EMA 重置 */
    s_bpmEMA       = 0.0f;
    s_bpmEmaFadeCnt = 0;

    /* v3.4: 确认 BPM 历史重置 */
    for (int i = 0; i < BPM_CONFIRMED_LEN; i++) s_confirmedBPM[i] = 0.0f;
    s_confirmedBPMIdx   = 0;
    s_confirmedBPMCount = 0;

    /* v3.3: slew rate 状态保留, 不重置 (避免重置后跃升) */
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

    /* v3.4: 确认 BPM 历史重置 */
    for (int i = 0; i < BPM_CONFIRMED_LEN; i++) s_confirmedBPM[i] = 0.0f;
    s_confirmedBPMIdx   = 0;
    s_confirmedBPMCount = 0;

    /* v3.3: 重置输出 BPM 跟踪 */
    s_lastOutputBPM = 0.0f;
}

float hrGetSQI(void)
{
    return s_sqi;
}

bool hrIsMotionActive(void)
{
    return s_motionActive;
}