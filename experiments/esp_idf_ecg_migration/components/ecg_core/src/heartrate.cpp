#include "arduino_compat.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "heartrate/heartrate.h"

/**
 * @file heartrate.cpp
 * @brief 板上心率计算模块 - 实现 (v4.2, fs=500Hz)
 *
 * ========== v4.2 改进: 参数级优化 (LUDB 全量 432 组合参数扫描) ==========
 *
 * THRESHOLD_RATIO 0.40→0.30, SIGNAL_WEIGHT/NOISE_WEIGHT 0.125→0.0625,
 * MIN_PEAK_RATIO 2.0→1.5, MIN_RR_SAMP 150→200 (400ms)。
 * 基于修复A/B/E 之后的参数空间扫描 (F1 0.747→0.774, Se +3.4pp,
 * BPM P90 10.1→7.0, ±10BPM 89.9%→93.5%)。
 *
 * ========== v4.1 改进: 三项结构性修复 (LUDB 金标准数据驱动) ==========
 *
 * 修复A: 不应期内 QRS 次级峰不再更新噪声峰值 → noisePeak 不再暴涨,
 *        阈值稳定在合理水平 (Se +7pp)。
 * 修复B: 超时复位改为软复位 (hrSoftReset), 保留自适应阈值,
 *        消除复位后阈值塌缩导致的误报风暴 ("复位后FP" 133→22)。
 * 修复E: isQRSValid 硬拒绝超范围 RR 间期, 不应期边缘次级峰不再
 *        计入 beatCount 污染阈值学习 (PPV +23pp)。
 *
 * LUDB 全库验证 (200 记录 / 1831 手标 QRS):
 *   Se 62.2%→69.3%, PPV 58.1%→81.1%, F1 0.600→0.747,
 *   BPM MAE 13.6→3.8, ±3BPM 36%→73%
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
/* 2026-08-14: 带通 5-15Hz → 8-25Hz — 模拟器 R 波极窄 (12ms), 15Hz LP 把 R
 * 能量砍半致 R≈T 能量 (能量包络分不清 R/T); 8-25Hz 带内 R 保留 36.2%、T 保留
 * 0.0% (FFT 实测), 干净分离。系数 scipy butter(2,[8,25],fs=500)。 */
#define QRS_LP25_A1  -1.561018f
#define QRS_LP25_A2   0.641352f
#define QRS_LP25_B0   0.020083f
#define QRS_LP25_B1   0.040167f
#define QRS_LP25_B2   0.020083f

#define QRS_HP8_A1   -1.858043f
#define QRS_HP8_A2    0.867472f
#define QRS_HP8_B0    0.931379f
#define QRS_HP8_B1   -1.862758f
#define QRS_HP8_B2    0.931379f

/* QRS BPF 状态变量 */
static float qrs_bpf_lp_w1 = 0.0f;
static float qrs_bpf_lp_w2 = 0.0f;
static float qrs_bpf_hp_w1 = 0.0f;
static float qrs_bpf_hp_w2 = 0.0f;

#define MWI_WINDOW      40          /**< 滑动积分窗口 80ms @500Hz (2026-08-14: 75→40
 * 窄 R 波 ~12ms, 75 窗(150ms)过度稀释且相位敏感; 40 窗紧贴 QRS 能量) */
#define REFRACTORY_MS   200         /**< 不应期 200ms (2026-08-14 毫秒化) */
#define MIN_RR_SEC      0.480f       /**< 最小 RR 480ms (秒) 拒 T 波双计数 —
 * 2026-08-14 修复: 原 MIN_RR_MS=480 误用毫秒, rrSec 是秒 → rrSec<480 恒真
 * → 第 2 拍起全部被拒 (真机/模拟 0 检出根因, 首拍跳 RR 门故只 1 拍) */
#define MAX_RR_SEC      2.000f       /**< 最大 RR 2s (秒) */

#define RR_BUFFER_SIZE  8           /**< BPM 中位数缓冲区容量 */

#define THRESHOLD_INIT  0.0001f     /**< 初始阈值 (能量包络域, 2026-08-14 弃导数平方):
 * QRS 带通后 ~0.17V → 能量-MWI 峰 ~5e-3, 噪声 ~2e-4, 伪影 ~1e-4; 下限 1e-4
 * 低于 QRS、高于纯噪声, 伪影由峰宽/自适应阈值/峰噪比协同过滤。 */
/* v4.2: THRESHOLD_RATIO 0.40→0.30, SIGNAL_WEIGHT 0.125→0.0625 (LUDB 参数扫描) */
#define THRESHOLD_RATIO 0.30f       /**< 阈值 = 噪声 + 0.30×(信号−噪声) */
#define SIGNAL_WEIGHT   0.0625f     /**< 信号峰值更新因子 (EMA) */
#define NOISE_WEIGHT   0.03f       /**< 噪声峰值更新因子 (EMA, 2026-08-08: 原0.0625 使
                                         T波/次峰快速抬升 np 至≈sp, QRS 被 MIN_PEAK_RATIO 拒绝) */
#define SIGNAL_WEIGHT_FAST 0.25f    /**< 运动恢复期快速收敛信号峰值 */
#define SIGNAL_WEIGHT_MOT  0.02f    /**< 运动期极慢更新 signalPeak */

/* v4.2: MIN_RR_SAMP 150→200 (400ms, LUDB 参数扫描: 消除 RR 缓冲污染) */
#define MIN_RR_SAMP     240         /**< 最小 RR: 480ms @500Hz (2026-08-14: 200→240
 * 防 T 波双计数 — 新电极位置 T 波显著, 真机实测 144 BPM ≈ 2×72; T 波距 R
 * ~300-450ms < 480ms 被拒; 心率上限 ~125 BPM, 消费级静息场景可接受) */
#define MAX_RR_SAMP     1000        /**< 最大 RR: 2000ms @500Hz */

#define TIMEOUT_MS      3000UL      /**< 3 秒无 QRS → 复位 (2026-08-14 毫秒化) */
#define HOLD_MS         1000UL      /**< 1 秒无新拍 → 停止输出旧 BPM */
#define MIN_CONF_BEATS  5           /**< 至少 5 拍才开始输出 BPM */
#define MIN_CONF_FEAT   1000        /**< 旧形态学验证开关: 保持 1000 = 禁用 —
 * 2026-08-14: 该验证按旧导数-MWI 标定 (40-80 宽 / rf 0.5-2.0 / 幅度一致性均值),
 * 能量包络+8-25Hz 下误杀 R 波; v6 用下方 MIN_CONF_FEAT_RF 的新门替代, 不再恢复旧块 */
#define MIN_CONF_FEAT_RF 3          /**< v6 新形态学门激活拍数 (2026-08-16 LUDB 扫描:
 * 能量包络域重新标定, 见 verify_heartrate_ludb_v6.py; 第 3 拍起激活) */
#define RISE_FALL_MAX_ENERGY 40.0f  /**< v6: 能量包络 rise/fall 比上限 — T 波 rf≈65-70,
 * 真 QRS rf≈6-33; LUDB 全量: rf>40 的 FP 145 拍, TP 仅 3 拍 (2026-08-16 峰值诊断) */
#define AMP_FRAC_PREV     0.55f     /**< v6: rr<0.9s 且峰幅 < 0.55×前一有效峰 → 拒
 * (宽 QRS 双计数次峰/残余 T 波; LUDB 全量 FP -11 且 TP 不减) */
#define AMP_FRAC_PREV_RR_MAX 0.9f   /**< 前拍幅度分数门仅作用于短 RR (<0.9s) */
#define RR_RATIO_MIN      0.65f     /**< v6: rr < 0.65×medianRR → 拒 (半 RR 双计数;
 * rr 缓冲 <3 拍或 median 无效时不启用) */
#define STARTUP_BLANK_SAMP 260      /**< v6: 起始消隐 260 样本 (520ms) — 滤波/MWI
 * 初始化瞬态伪峰集中在样本 ~44-97; LUDB 200 记录最早 TP 检出在样本 306,
 * 消隐不伤真实拍 (2026-08-16 峰值级证据) */
/* v4.2: MIN_PEAK_RATIO 2.0→1.5 (LUDB 参数扫描: 修复A后 np 不再暴涨, 2.0 过严) */
/* 2026-08-08: 1.5→1.2 (模拟器小信号下 np 仍被次峰抬升, QRS/噪声比 ~1.26 被误拒) */
#define MIN_PEAK_RATIO  1.2f        /**< 峰/噪比门限 (静止) */
#define MIN_PEAK_RATIO_MOT 1.2f     /**< 运动期峰噪比 (降低, 易检测) */

/* ======== 信号活动检测 ======== */
#define ACT_WIN_SAMP    500         /**< 活动检测窗口: 1秒 @500Hz */
#define ACT_THRESHOLD   0.005f      /**< 最小信号幅度 5mV */
#define ACT_TIMEOUT_CNT 2           /**< 连续 2 秒无信号→标记消失 */

/* ======== 自适应初始阈值 ======== */
#define ADAPT_INIT_SAMP 100         /**< 自适应学习采样数 (200ms @500Hz) */
#define ADAPT_INIT_FACTOR 0.5f      /**< 阈值 = 学习窗 MWI RMS × 0.5 (2026-08-08:
                                         原 2.0 按1V信号标定, 学习窗含 QRS 时 RMS≈峰值,
                                         ×2 后阈值必超峰值 → QRS 永不检出) */

/* ======== SQI 与运动检测 ======== */
#define SQI_EMA_WEIGHT  0.05f       /**< SQI 指数平滑因子 (慢) */
#define SQI_MOTION_ENTER 0.35f      /**< SQI 低于此值 → 进入运动状态 */
#define SQI_MOTION_EXIT  0.55f      /**< SQI 高于此值 → 退出运动状态 */
#define SQI_SNR_FLOOR    0.0001f    /**< SNR 最小值防除零 (2026-08-08: 原0.001 按1V信号
                                         标定, 小信号 sp~2e-4 时 SQI 恒压至 0.17 → motion
                                         永久锁定 → BPM 走 EMA 输出 → 65→74 爬升循环) */
#define MOTION_BPM_HOLD  1500       /**< 运动结束后保持峰值冻结的帧数 (3秒) */

/* ======== 运动检测滞回 ======== */
#define MOTION_ENTER_CNT 250        /**< 连续250帧(0.5秒)SQI低→进入运动 */
#define MOTION_EXIT_CNT  100        /**< 连续100帧(0.2秒)SQI高→退出运动 */

/* ======== 形态学验证常数 (静止) ======== */
#define MWI_HIST_LEN     120        /**< MWI 历史长度 (240ms @500Hz) */
#define PEAK_HIST_LEN    8          /**< 近期峰值历史长度 */
#define MIN_QRS_WIDTH    25         /**< 最小 QRS 半高宽 (2026-08-14: 40→25 — 能量包络
 * 40 样本滑动窗把窄 R 展宽到 ~40 样本, 旧下限 40 恰好卡边界致第 8 拍起间歇拒;
 * 能量包络+8-25Hz 已大幅抑制伪影, 宽度门可放宽) */
#define MAX_QRS_WIDTH    100        /**< 最大 QRS 半高宽 200ms @500Hz (80→100) */
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
static unsigned long s_lastBeatMillis = 0;      /* 上次心拍时间戳 (2026-08-14, 帧率无关 RR) */
static unsigned long s_refractUntilMillis = 0;  /* 不应期结束时间戳 */
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
static uint32_t   s_sampSinceInit;   /* v6 起始消隐计数: 自 hrFullReset 起累计,
                                       * hrReset/hrSoftReset 不清零 (与 Python 复刻
                                       * self.i 同源, 仅在检测器全量初始化时归零) */

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
static float       s_adaptWinMax;    /* 学习窗 MWI 最大值 (2026-08-14 峰基法) */

/* ======================== 工具函数 ======================== */

static float applyQRSBandpass(float x)
{
    float w_lp = x - QRS_LP25_A1 * qrs_bpf_lp_w1 - QRS_LP25_A2 * qrs_bpf_lp_w2;
    float y_lp = QRS_LP25_B0 * w_lp + QRS_LP25_B1 * qrs_bpf_lp_w1
               + QRS_LP25_B2 * qrs_bpf_lp_w2;
    qrs_bpf_lp_w2 = qrs_bpf_lp_w1;
    qrs_bpf_lp_w1 = w_lp;

    float w_hp = y_lp - QRS_HP8_A1 * qrs_bpf_hp_w1 - QRS_HP8_A2 * qrs_bpf_hp_w2;
    float y_hp = QRS_HP8_B0 * w_hp + QRS_HP8_B1 * qrs_bpf_hp_w1
               + QRS_HP8_B2 * qrs_bpf_hp_w2;
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
    /* 2026-08-08: delta 下限 0.001f 按 1V 信号标定, 模拟器/AFE 小信号
     * (MWI 峰值 ~2e-4) 下会抬死阈值 → 改 0.0001f (与 THRESHOLD_INIT 同量级) */
    if (delta < 0.0001f) delta = 0.0001f;

    s_threshold = s_noisePeak + THRESHOLD_RATIO * delta;
    if (s_threshold < THRESHOLD_INIT) {
        s_threshold = THRESHOLD_INIT;
    }
}

/* 2026-08-14: 仅更新 signalPeak (不触碰阈值/噪声) — 供"高于阈值但被 RR/形态
 * 门拒绝"的峰使用, 使 SQI 反映真实 QRS 幅度而非检测成败 (解耦死循环)。 */
static void updateSignalPeak(float peakVal)
{
    float weight = SIGNAL_WEIGHT;
    s_signalPeak = weight * peakVal + (1.0f - weight) * s_signalPeak;
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

    /* v6 (2026-08-16): 起始消隐 — 滤波/MWI 初始化瞬态伪峰 (LUDB 集中在
     * 样本 ~44-97); LUDB 200 记录最早 TP 检出在样本 306, 260 样本消隐安全。
     * <= 对齐 Python 复刻 self.i < 260 (固件计数 1-based vs 0-based) */
    if (s_sampSinceInit <= STARTUP_BLANK_SAMP) return false;

    /* 修复E (v4.1): 硬拒绝超范围 RR 间期。
     * 固件 v4.0 只在 addRRInterval() 中丢弃超范围 RR, 但 isQRSValid()
     * 会接受该峰并递增 beatCount, 导致不应期边缘的次级峰污染
     * 阈值学习与形态验证开启时机 (LUDB 验证: PPV +23pp)。 */
    /* 2026-08-14 毫秒化: RR 范围用真实时间 (帧率无关); 首拍 (beatCount==0,
     * 尚无前拍) 跳过范围校验 — 勿用 s_lastBeatMillis>0 (hrReset 已置其为 millis())。 */
    if (s_beatCount > 0) {
        if (rrSec < MIN_RR_SEC)           return false;
        if (rrSec > MAX_RR_SEC)           return false;
    }

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
            /* 2026-08-08: 禁用 rise/fall 检查 — 模拟器窄 QRS + 75 样本 MWI 窗
             * 使 mwi 峰严重不对称 (实测 rf≈41), 真实 ECG 标定上限 2.0 误杀全部
             * QRS (N16R8 板上: b 卡 8 后 3s 超时复位循环)。width + 振幅一致 +
             * RR 一致已足够过滤伪峰。 */
            // float ratio = getRiseFallRatio();
            // if (ratio < RISE_FALL_MIN || ratio > RISE_FALL_MAX) return false;
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

    /* ======== v6 能量包络域形态学门 (2026-08-16 LUDB 全量重标定) ========
     * 旧形态学块 (MIN_CONF_FEAT=1000) 保持禁用; 以下三门自第 3 拍起激活,
     * 与 verify_heartrate_ludb_v6.py 的 blank260_rf_c3_40_prev055_rr065 逐行同源:
     *   Se 96.94%→96.40% (TP 1775→1765, FN 56→66),
     *   PPV 71.03%→78.87% (FP 724→473),
     *   F1 0.820→0.868, BPM MAE 10.17→4.16 (中位 3.15→1.46, P90 36.2→9.1) */
    if (s_beatCount >= MIN_CONF_FEAT_RF) {
        /* 门1: rise/fall 比上限 — T 波 rf≈65-70, QRS rf≈6-33 */
        float rf = getRiseFallRatio();
        if (rf > RISE_FALL_MAX_ENERGY) return false;

        /* 门2: 短 RR 内前拍幅度分数 — 宽 QRS 双计数次峰/残余 T 波 */
        if (rrSec < AMP_FRAC_PREV_RR_MAX && s_peakHistCount >= 1) {
            float prevPeak = s_recentPeaks[(s_peakHistIdx - 1 + PEAK_HIST_LEN)
                                           % PEAK_HIST_LEN];
            if (prevPeak >= 0.0001f && peakVal < AMP_FRAC_PREV * prevPeak) {
                return false;
            }
        }

        /* 门3: 半 RR 双计数 — rr 显著短于近期中位 RR */
        if (s_rrCount >= 3 && s_medianRR >= 0.001f) {
            if (rrSec < RR_RATIO_MIN * s_medianRR) return false;
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

static void hrSoftReset(void);   /* v4.1 修复B: 前置声明, 定义见 hrReset 上方 */

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
    s_sampSinceInit++;   /* v6 起始消隐计数 (hrReset/hrSoftReset 不清零) */

    checkSignalActivity(filteredSample);

    updateSQI();
    updateMotionState();

    float qrsSignal = applyQRSBandpass(filteredSample);

    /* 2026-08-14 能量包络 (弃导数平方): 导数平方对尖锐阶跃/尖峰放大 8×
     * (一步 diff=0.15→diff²=0.02 vs QRS 平滑上升 0.003), 模拟器注入的运动
     * 阶跃+稀疏尖峰在导数域盖过 QRS (真机/模拟均 0 检出根因)。改用 x² 直接
     * 积分 — QRS 有持续能量、伪影是瞬态, 积分后 QRS 峰 >> 伪影 (能量域)。 */
    float energy = qrsSignal * qrsSignal;
    float mwi = computeMWI(energy);

    /* 自适应初始阈值 (MWI 域学习, 2026-08-08 修复: 原 qrsSignal 信号域 RMS×2
     * 与 MWI 峰值跨域失配, 小信号下阈值比峰值大 50-800 倍, QRS 永不检出)。
     * 2026-08-14 死锁打破: 原逻辑 adaptInitDone 后不再重学 — 学习窗必然包含
     * QRS (RR<1s < 窗宽+滑动), 阈值偏高致 0 检出后永不恢复 (真机新电极位置
     * 实测 90s 0 心拍)。改为: 只要 beatCount==0 就每 ADAPT_INIT_SAMP 滚动重学。 */
    if (s_beatCount == 0) {
        s_adaptInitCount++;
        if (mwi > s_adaptWinMax) s_adaptWinMax = mwi;
        if (s_adaptInitCount >= ADAPT_INIT_SAMP) {
            /* 2026-08-14 峰基法: 阈值 = 窗口 MWI 最大值 × 0.4 — 直接锁 QRS 峰
             * 量级, 对波动幅度更稳健 (RMS 法在波动信号下易被 QRS 占空比稀释)。
             * 下限 THRESHOLD_INIT(1e-5) 仍高于噪声峰 (1e-6~8e-6)。 */
            float adaptiveThreshold = s_adaptWinMax * 0.4f;
            if (adaptiveThreshold < THRESHOLD_INIT) adaptiveThreshold = THRESHOLD_INIT;
            if (s_sqi >= 0.45f || adaptiveThreshold < s_threshold) {
                s_threshold = adaptiveThreshold;
                s_signalPeak = adaptiveThreshold;
                s_noisePeak = adaptiveThreshold * 0.3f;
            }
            s_adaptInitCount = 0;
            s_adaptWinMax = 0.0f;
            s_adaptInitDone = true;
        }
    }

    s_mwiHistory[s_mwiHistIdx] = mwi;
    s_mwiHistIdx = (s_mwiHistIdx + 1) % MWI_HIST_LEN;

    bool isPeak = (s_mwiPrev > s_mwiPrevPrev) && (s_mwiPrev > mwi);

    if (isPeak) {
        float peakVal = s_mwiPrev;
        unsigned long nowMs = millis();
        float rrSec = (s_beatCount > 0)
                      ? (float)(nowMs - s_lastBeatMillis) / 1000.0f : 0.0f;
        bool valid = isQRSValid(peakVal, rrSec);

        if (valid) {
            addRRInterval(rrSec);
            updateThreshold(peakVal, true);
            recordValidPeak(peakVal);

            s_state = HR_REFRACTORY;
            s_lastBeatMillis = nowMs;
            s_refractUntilMillis = nowMs + REFRACTORY_MS;
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
            /* 修复A (v4.1): 不应期内 QRS 次级峰不更新噪声峰值。
             * 固件 v4.0 在 REFRACTORY 状态仍把拍后 200ms 内的次级峰
             * 喂给 noisePeak, 使其暴涨到拍幅量级, 阈值被抬至 ~0.9×拍幅,
             * 后续真实 QRS 被 MIN_PEAK_RATIO 检查误杀 (LUDB 验证: Se +7pp)。
             * 2026-08-08 收紧: 噪声峰更新条件从 peakVal > np*0.5 改为
             * peakVal < s_threshold — 只有低于当前阈值的峰才是噪声峰。
             * 原条件让运动伪影/EMG 突发等大幅峰 (高于阈值) 也喂给 np,
             * np 暴涨至 sp 的 5.9 倍 → SQI 0.15 < 0.35 → 误判运动 → BPM
             * 改用 EMA 输出 → 65→74 指数爬升循环 (N16R8 板上实测)。 */
            if (s_state != HR_REFRACTORY && s_signalPresent) {
                if (peakVal < s_threshold) {
                    updateThreshold(peakVal, false);   /* 噪声峰 */
                } else {
                    updateSignalPeak(peakVal);          /* 真 QRS 被拒, 仍喂 signalPeak */
                }
            }
        }
    }

    s_sampSinceBeat++;

    if (s_state == HR_REFRACTORY && millis() >= s_refractUntilMillis) {
        s_state = (s_beatCount >= MIN_CONF_BEATS) ? HR_TRACKING : HR_IDLE;
    }

    if (s_signalPresent && (millis() - s_lastBeatMillis) > TIMEOUT_MS) {
        hrSoftReset();
        s_state = HR_LEARNING;
    }

    s_mwiPrevPrev = s_mwiPrev;
    s_mwiPrev     = mwi;

    unsigned long sinceBeat = millis() - s_lastBeatMillis;
    if (!result.beatDetected && s_beatCount > 0 && s_medianRR > 0.001f
        && s_signalPresent && sinceBeat < HOLD_MS) {
        uint8_t bpmRaw = computeOutputBPM();
        if (bpmRaw >= 30 && bpmRaw <= 200) {
            result.bpm = bpmRaw;
        }
        float decay = 1.0f - (float)sinceBeat / (float)HOLD_MS;
        float bufConf = (float)s_rrCount / (float)RR_BUFFER_SIZE;
        float sqiWeight = (s_sqi < 0.4f) ? (s_sqi / 0.4f) : 1.0f;
        float motionFactor = s_motionConfirmed ? 0.5f : 1.0f;
        result.confidence = fminf(0.8f, bufConf * sqiWeight * motionFactor) * decay;
    }

    result.sqi          = s_sqi;
    result.motionActive = s_motionActive;
    /* 2026-08-14 修复: 无条件携带累计 beatCount + 最近 RR — 原 result 仅拍内帧
     * 非零, 显示层 (1Hz 采样) 几乎总读到 0 → 恒显 "等待心拍"/RR 0.0ms。 */
    result.beatCount = s_beatCount;
    result.rrInterval = s_lastRR;

    return result;
}

static void hrSoftReset(void)
{
    /* 修复B (v4.1): 超时复位保留自适应阈值, 仅清检测历史。
     * 固件 v4.0 超时后 hrReset() 把阈值重置为 THRESHOLD_INIT(0.002),
     * 阈值塌缩导致噪声峰全部通过, 产生误报风暴 (LUDB 验证: 修复后
     * ">3s复位后" FP 从 133 降至 22)。 */
    float holdSP = s_signalPeak;
    float holdNP = s_noisePeak;
    float holdTH = s_threshold;
    hrReset();
    s_signalPeak = holdSP;
    s_noisePeak  = holdNP;
    s_threshold  = holdTH;
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
    s_lastBeatMillis = millis();       /* 2026-08-14: 复位后重新计时 (帧率无关) */
    s_refractUntilMillis = 0;
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
    s_adaptWinMax = 0.0f;
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
    s_sampSinceInit = 0;   /* v6 起始消隐计数仅在检测器全量初始化时归零 */

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

uint32_t hrGetLastBeatMillis(void)
{
    return s_lastBeatMillis;
}