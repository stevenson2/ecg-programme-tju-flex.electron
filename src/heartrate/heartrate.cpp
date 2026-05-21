#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "heartrate/heartrate.h"

/**
 * @file heartrate.cpp
 * @brief 板上心率计算模块 - 实现 (v3.0)
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
 * ========== v3.0 改进 ==========
 *   - SQI (信号质量指数): 基于信号-噪声比, 实时评估质量
 *   - 运动检测: SQI 滞回判定 (0.35 进入 / 0.55 退出)
 *   - 阈值冻结: 运动期间锁定 signal/noise 峰值, 防止漂移
 *   - BPM 平滑: 运动结束后 3 秒过渡, 避免 BPM 突变
 *   - 信号活动检测: 1秒窗口监测最大绝对值, <5mV持续2秒→标记无信号
 *   - 无信号时 isQRSValid() 始终返回 false, 阻止噪声被误检
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
static float      s_sqi;              /**< 当前 SQI 值 (EMA 平滑) */
static bool       s_motionActive;     /**< 运动干扰标志 */
static bool       s_motionConfirmed;  /**< 运动真标记 (静止→运动滞回确认) */
static int        s_motionLowCount;   /**< SQI 连续低于阈值的帧数 */
static int        s_motionHighCount;  /**< SQI 连续高于阈值的帧数 */
static int        s_motionRecoverCnt; /**< 运动退出后 BPM 过渡帧计数 */
static float      s_motionHoldSP;     /**< 运动前的 signalPeak 快照 */
static float      s_motionHoldNP;     /**< 运动前的 noisePeak 快照 */

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
    /* 运动状态冻结阈值更新, 防止噪声抬高 signalPeak */
    if (s_motionConfirmed && isSignal) {
        /* 仅更新噪声 (保守), 不更新信号峰值 */
        /* 信号峰值用恢复期快加速度收敛 */
        return;
    }

    if (isSignal) {
        /* 运动恢复期使用快速收敛因子 */
        float weight = (s_motionRecoverCnt > 0)
                       ? SIGNAL_WEIGHT_FAST : SIGNAL_WEIGHT;
        s_signalPeak = weight * peakVal
                     + (1.0f - weight) * s_signalPeak;
    } else {
        /* 噪声总是更新 (保守), 但运动期间使用更快的因子 */
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

/**
 * @brief 计算瞬时 SQI (基于当前信号-噪声比)
 *
 * SQI = s_signalPeak / (s_signalPeak + s_noisePeak + epsilon)
 * 取值范围 0.0 (纯噪声) ~ 1.0 (纯信号)。
 * 使用 EMA 平滑输出到 s_sqi。
 */
static void updateSQI(void)
{
    float snrDenom = s_signalPeak + s_noisePeak + SQI_SNR_FLOOR;
    float rawSQI = s_signalPeak / snrDenom;
    if (rawSQI > 1.0f) rawSQI = 1.0f;
    if (rawSQI < 0.0f) rawSQI = 0.0f;
    s_sqi = SQI_EMA_WEIGHT * rawSQI + (1.0f - SQI_EMA_WEIGHT) * s_sqi;
}

/**
 * @brief 运动状态滞回判定
 *
 * 进入运动: 连续 250 帧 (1秒) SQI < SQI_MOTION_ENTER
 * 退出运动: 连续 125 帧 (0.5秒) SQI > SQI_MOTION_EXIT
 *
 * 进入时保存阈值快照, 退出时启动 3秒 BPM 过渡。
 */
static void updateMotionState(void)
{
    if (!s_motionConfirmed) {
        /* 当前未确认运动: 检查是否应进入 */
        if (s_sqi < SQI_MOTION_ENTER) {
            s_motionLowCount++;
            s_motionHighCount = 0;
        } else {
            s_motionLowCount = 0;
        }
        if (s_motionLowCount >= 250) {
            /* 确认进入运动状态 */
            s_motionConfirmed = true;
            s_motionActive = true;
            s_motionLowCount = 0;
            /* 保存当前阈值快照, 供恢复期使用 */
            s_motionHoldSP = s_signalPeak;
            s_motionHoldNP = s_noisePeak;
            s_motionRecoverCnt = 0;
        }
    } else {
        /* 当前处于运动状态: 检查是否应退出 */
        if (s_sqi > SQI_MOTION_EXIT) {
            s_motionHighCount++;
            s_motionLowCount = 0;
        } else {
            s_motionHighCount = 0;
        }
        if (s_motionHighCount >= 125) {
            /* 确认退出运动状态 */
            s_motionConfirmed = false;
            s_motionActive = false;
            s_motionHighCount = 0;
            /* 启动 BPM 过渡: 保留旧阈值快照, 用快速收敛恢复 */
            s_motionRecoverCnt = MOTION_BPM_HOLD;
            /* 信号峰值回退到运动前快照, 防止噪声抬高 */
            if (s_motionHoldSP > s_signalPeak) {
                s_signalPeak = s_motionHoldSP;
            }
            if (s_motionHoldNP < s_noisePeak) {
                s_noisePeak = s_motionHoldNP;
            }
        }
    }

    /* 恢复期倒计时 */
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

    /* 运动期间不更新 RR 缓冲区 (避免噪声 RR 污染中位数) */
    if (s_motionConfirmed) return;

    s_rrBuf[s_rrIdx] = rrSeconds;
    s_rrIdx = (s_rrIdx + 1) % RR_BUFFER_SIZE;
    if (s_rrCount < RR_BUFFER_SIZE) s_rrCount++;

    s_medianRR = computeMedianRR();
    s_lastRR = rrSeconds;
}

static bool isQRSValid(float peakVal)
{
    if (!s_signalPresent)                 return false;
    if (s_state == HR_REFRACTORY)         return false;
    if (peakVal <= s_threshold)           return false;
    if (peakVal < s_noisePeak * MIN_PEAK_RATIO) return false;

    /* 运动期间提高峰噪比要求 (减少误检) */
    if (s_motionConfirmed && peakVal < s_noisePeak * (MIN_PEAK_RATIO * 1.5f))
        return false;

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
                /* 信号恢复 → 重置算法, 重新学习 */
                s_signalPresent = true;
                hrReset();
                s_state = HR_LEARNING;
                s_signalPresent = true;
            }
        }

        if (s_noSignalSeconds >= ACT_TIMEOUT_CNT) {
            if (s_signalPresent) {
                /* 信号消失 → 冻结算法 */
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
    s_signalPeak = THRESHOLD_INIT;
    s_noisePeak  = THRESHOLD_INIT * 0.3f;
    s_threshold  = THRESHOLD_INIT;
    s_sqi        = 0.5f;  /* 初始假设中等质量 */
}

HR_Result hrProcess(float filteredSample)
{
    HR_Result result = { 0 };

    /* 步骤 0: 信号活动检测 */
    checkSignalActivity(filteredSample);

    /* 步骤 0.5: SQI 更新 (每帧基于当前 SNR) */
    updateSQI();
    updateMotionState();

    /* 步骤 1-3: 差分 → 平方 → 滑动积分 */
    float diff = filteredSample - s_prevSample;
    s_prevSample = filteredSample;
    float squared = diff * diff;
    float mwi = computeMWI(squared);

    /* 步骤 4: 峰值检测 (MWI 局部最大值) */
    bool isPeak = (s_mwiPrev > s_mwiPrevPrev) && (s_mwiPrev > mwi);

    if (isPeak) {
        float peakVal = s_mwiPrev;

        if (isQRSValid(peakVal)) {
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

            if (s_state == HR_LEARNING && s_beatCount >= MIN_CONF_BEATS) {
                s_state = HR_TRACKING;
            }

            if (s_state == HR_TRACKING && s_medianRR > 0.001f) {
                /* 运动期间不输出新的 BPM, 保持旧值 */
                if (!s_motionConfirmed) {
                    uint8_t bpmRaw = (uint8_t)(60.0f / s_medianRR + 0.5f);
                    if (bpmRaw >= 30 && bpmRaw <= 200) {
                        result.bpm = bpmRaw;
                    }
                }
                /* 置信度 = RR 缓冲区填充率 × (1-SQI衰减) */
                float bufConf = (float)s_rrCount / (float)RR_BUFFER_SIZE;
                float sqiWeight = (s_sqi < 0.4f) ? (s_sqi / 0.4f) : 1.0f;
                result.confidence = fminf(1.0f, bufConf * sqiWeight);
            }

        } else {
            /* 噪声峰 (仅在信号存在时更新噪声估计) */
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

    /* 步骤 6: 超时复位 (仅信号存在时) */
    if (s_signalPresent && s_sampSinceBeat > TIMEOUT_SAMP) {
        hrReset();
        s_state = HR_LEARNING;
    }

    /* 步骤 7: 滚动 MWI 历史 */
    s_mwiPrevPrev = s_mwiPrev;
    s_mwiPrev     = mwi;

    /* 步骤 8: 无新拍时保持 (仅信号存在 + 1秒内 + 非运动) */
    if (!result.beatDetected && s_beatCount > 0 && s_medianRR > 0.001f
        && s_signalPresent && s_sampSinceBeat < HOLD_SAMP
        && !s_motionConfirmed) {
        result.bpm = (uint8_t)(60.0f / s_medianRR + 0.5f);
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

/**
 * @brief 内部复位: 仅重置 QRS 检测状态, 保留信号活动标记
 *
 * 由 checkSignalActivity 和超时机制调用。
 * 不重置 s_signalPresent / s_noSignalSeconds 等信号活动状态,
 * 避免干扰正在进行的信号存在/消失判断。
 */
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

    /* v3.0: 不重置 SQI/运动状态, 保留运动检测连续性 */
}

/**
 * @brief 完全复位: 重置所有状态包括信号活动检测
 *
 * 由 main.cpp 在输入模式切换时调用。
 * 强制重新评估信号存在性, 避免旧环境参数影响新输入。
 */
void hrFullReset(void)
{
    hrReset();

    /* 信号活动检测完全重置 */
    s_signalPresent  = true;   /* 默认假设信号存在, 1秒后重新评估 */
    s_winMaxAbs      = 0.0f;
    s_winCount       = 0;
    s_noSignalSeconds = 0;

    /* v3.0: 运动检测完全重置 */
    s_sqi              = 0.5f;
    s_motionActive     = false;
    s_motionConfirmed  = false;
    s_motionLowCount   = 0;
    s_motionHighCount  = 0;
    s_motionRecoverCnt = 0;
    s_motionHoldSP     = 0.0f;
    s_motionHoldNP     = 0.0f;
}

float hrGetSQI(void)
{
    return s_sqi;
}

bool hrIsMotionActive(void)
{
    return s_motionActive;
}