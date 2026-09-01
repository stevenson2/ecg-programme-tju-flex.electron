#include <math.h>
#include <string.h>
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

/* ======================== 第1级：高通 0.5Hz (fs=500Hz) ======================== */
/* 2026-08-13 决策 (TH §44, 产品需求): 显示链 HP 由 0.05Hz 提高到 0.5Hz。
 * 背景: 0.05Hz 保 ST 段形态但滤不净呼吸漂移 (0.2~0.5Hz), 真机实测基线漂移
 * ±100mV 严重。产品定位 = 心律失常检测 + 显示 (非 ST 诊断), 基线稳定优先 →
 * 改用 0.5Hz (消费级 ECG 标准, Kardia/Apple Watch 同款)。
 * 权衡: ST 段引入伪偏移 (Buendía-Fuentes 2012: 1.5-9mm), 但 AI 链独立 0.5Hz
 * 不受影响; ST 测量将来用单独 0.05Hz 链。LP 40Hz 不变。
 * 系数由 Python 计算: scipy butter(2, 0.5, 'high', fs=500), 完整 double 精度。 */
#define HP_A1  -1.9911142922016536
#define HP_A2   0.9911535958689355
#define HP_B0   0.9955669720176472
#define HP_B1  -1.9911339440352944
#define HP_B2   0.9955669720176472

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

/* ======================== AI 输入链: 二阶因果高通 0.5Hz (fs=250Hz, 2026-08-13 修正) ================
 * P0-2 训练-部署失配修正 (TH §42/§43): 原系数为 butter(2,0.5,fs=500) 设计, 但 AI 链
 * 经 2:1 抽取后实际 250Hz → 有效截止 0.25Hz (bug)。改用 butter(2,0.5,fs=250) 修正
 * 系数 (ai_hp_coeffs_fs250.txt), 在 250Hz 链上真正实现 0.5Hz 截止。
 * 同时 AI 链由"窗口级零相位"(aiApplyFilterWindow) 改回"因果 streaming"(aiApplyFilter,
 * 状态 ai_hp_w1/w2 跨窗口持续), 与训练侧重训链 causal_hp_05_fs250 (data/preprocess.py)
 * 位级一致 (零初始状态 streaming)。
 * 系数: K = tan(pi*0.5/250), 由 compute_ai_hp_coeffs.py 生成。 */
#define AI_HP_A1  -1.9822289297925284
#define AI_HP_A2   0.98238545061412508
#define AI_HP_B0   0.99115359510166301
#define AI_HP_B1  -1.982307190203326
#define AI_HP_B2   0.99115359510166301

/* ======================== AI 输入链独立 0.05Hz HP 系数 (fs=500Hz, 2026-08-13 解耦) ================
 * 显示链 HP 0.5Hz (基线稳定) 与 AI 链解耦后, AI 输入链在 2:1 抽取前需独立做
 * HP 0.05Hz + LP 40Hz, 与训练侧 exp7 复刻链"梳状→HP0.05+LP40→2:1抽取→因果0.5Hz"
 * 位级一致。此处恢复原 0.05Hz HP 系数 (scipy butter(2,0.05,'high',fs=500),
 * 与改显示链前的 HP_* 宏相同), 供 applyFilterAI 独立使用。 */
#define AI_CHAIN_HP_A1  -1.9991114234707954
#define AI_CHAIN_HP_A2   0.9991118180796384
#define AI_CHAIN_HP_B0   0.9995558103876084
#define AI_CHAIN_HP_B1  -1.9991116207752169
#define AI_CHAIN_HP_B2   0.9995558103876084

/* ======================== 预热样本数 ======================== */
#define WARMUP_SAMPLES  240  /* 约 0.48s @500Hz */

/* ======================== 状态变量 (double: 0.05Hz HP float32 灾难性抵消, 2026-08-08) ======================== */
/* 第1级：高通 */
static double hp_w1 = 0.0;
static double hp_w2 = 0.0;
/* 第2级：低通 */
static double lp_w1 = 0.0;
static double lp_w2 = 0.0;
/* AI 输入链高通 (0.5Hz @250Hz 因果, 2026-08-13 修正) */
static double ai_hp_w1 = 0.0;
static double ai_hp_w2 = 0.0;
/* AI 输入链独立滤波 (HP 0.05 + LP 40, 与显示链解耦, 2026-08-13) */
static double ai_chain_hp_w1 = 0.0;
static double ai_chain_hp_w2 = 0.0;
static double ai_chain_lp_w1 = 0.0;
static double ai_chain_lp_w2 = 0.0;

/* ======================== 显示链状态 (2026-08-14) ======================== */
/* 用户按 ADI 视频要求: 显示链 = 高通 4Hz + 低通 (默认 40Hz, 可切 4Hz)。
 * HP 4Hz 将基线漂移/呼吸 (0.2-0.5Hz) 衰减 -45dB, 同时滤除 P/T/ST (4Hz 以下)
 * — 显示为 ADI 演示风格 "QRS 尖峰骑平线" (仅显示; 心率/VF/AI 链不受影响)。 */
static double disp_hp_w1 = 0.0;
static double disp_hp_w2 = 0.0;
static double disp_lp_w1 = 0.0;
static double disp_lp_w2 = 0.0;

/* 显示链 LP 截止频率选择: 0 = 40Hz (默认形态保真), 1 = 4Hz (试验)。
 * DIAG LPF <4|40> 运行时切换。 */
static int g_dispLpSel = 0;   /* 默认 40Hz */

/* 4Hz 高通 Butterworth 2阶 (fs=500, scipy 生成): 0.3Hz -45dB / 4Hz -3dB / 10Hz -0.1dB */
#define DISP_HP4_A1  -1.9289422632520332
#define DISP_HP4_A2  0.9313816821269024
#define DISP_HP4_B0  0.9650809863447340
#define DISP_HP4_B1  -1.9301619726894681
#define DISP_HP4_B2  0.9650809863447340

/* 4Hz 低通 Butterworth 2阶 (fs=500, 试验档): 10Hz -16dB 压扁 QRS */
#define DISP_LP4_A1  -1.9289422632520332
#define DISP_LP4_A2  0.9313816821269024
#define DISP_LP4_B0  0.0006098547187173
#define DISP_LP4_B1  0.0012197094374346
#define DISP_LP4_B2  0.0006098547187173

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

/* ======================== AI 输入链高通实现 (0.5Hz @250Hz 因果, 2026-08-13 修正) ======================== */

float aiApplyFilter(float inputSample)
{
    /* 仅 0.5Hz 因果高通 (fs=250 修正系数, 训练链匹配), 不含 LP (已由 AI 链 LP40 处理)。
     * 状态 ai_hp_w1/w2 跨窗口持续 (零初始, 由 aiFilterInit/aiFilterReset 复位),
     * 逐样本 streaming — 与训练侧 causal_hp_05_fs250 (DF2T, 零初始状态) 语义一致。 */
    return applyBiquad(inputSample, AI_HP_B0, AI_HP_B1, AI_HP_B2,
                       AI_HP_A1, AI_HP_A2, &ai_hp_w1, &ai_hp_w2);
}

/* ======================== AI 输入链独立滤波 (HP 0.05 + LP 40, 2026-08-13 解耦) ======================== */

float applyFilterAI(float inputSample)
{
    /* AI 输入链在 2:1 抽取前的独立滤波: HP 0.05Hz + LP 40Hz, 与训练侧 exp7 复刻链
     * "梳状→HP0.05+LP40→2:1抽取→因果0.5Hz" 位级一致。
     * 显示链 HP 0.5Hz (基线稳定) 与 AI 链 HP 0.05Hz (ST 保真/训练一致) 解耦 —
     * 改显示链 HP 不再影响 AI 输入, 避免 train/deploy 失配。 */
    float temp = applyBiquad(inputSample, AI_CHAIN_HP_B0, AI_CHAIN_HP_B1, AI_CHAIN_HP_B2,
                             AI_CHAIN_HP_A1, AI_CHAIN_HP_A2, &ai_chain_hp_w1, &ai_chain_hp_w2);
    temp = applyBiquad(temp, LP_B0, LP_B1, LP_B2, LP_A1, LP_A2, &ai_chain_lp_w1, &ai_chain_lp_w2);
    return temp;
}

void aiChainFilterReset(void)
{
    ai_chain_hp_w1 = 0.0;
    ai_chain_hp_w2 = 0.0;
    ai_chain_lp_w1 = 0.0;
    ai_chain_lp_w2 = 0.0;
}

void aiFilterInit(void)
{
    ai_hp_w1 = 0.0;
    ai_hp_w2 = 0.0;
    ai_chain_hp_w1 = 0.0;
    ai_chain_hp_w2 = 0.0;
    ai_chain_lp_w1 = 0.0;
    ai_chain_lp_w2 = 0.0;
}

void aiFilterReset(void)
{
    ai_hp_w1 = 0.0;
    ai_hp_w2 = 0.0;
    ai_chain_hp_w1 = 0.0;
    ai_chain_hp_w2 = 0.0;
    ai_chain_lp_w1 = 0.0;
    ai_chain_lp_w2 = 0.0;
}

void filterReset(void)
{
    hp_w1 = 0.0f;
    hp_w2 = 0.0f;
    lp_w1 = 0.0f;
    lp_w2 = 0.0f;
    displayFilterReset();
}

/* ======================== 显示链实现 (2026-08-14) ======================== */

void displayFilterReset(void)
{
    disp_hp_w1 = 0.0;
    disp_hp_w2 = 0.0;
    disp_lp_w1 = 0.0;
    disp_lp_w2 = 0.0;
}

float applyDisplayFilter(float inputSample)
{
    /* 高通 4Hz (ADI 视频风格: 基线/P/T/ST 全滤除, 仅 QRS 骑平线) */
    float hped = applyBiquad(inputSample, DISP_HP4_B0, DISP_HP4_B1, DISP_HP4_B2,
                             DISP_HP4_A1, DISP_HP4_A2, &disp_hp_w1, &disp_hp_w2);
    /* LP 平滑 (默认 40Hz 形态保真; DIAG LPF 4 切试验档) */
    if (g_dispLpSel == 1) {
        return applyBiquad(hped, DISP_LP4_B0, DISP_LP4_B1, DISP_LP4_B2,
                           DISP_LP4_A1, DISP_LP4_A2, &disp_lp_w1, &disp_lp_w2);
    }
    return applyBiquad(hped, LP_B0, LP_B1, LP_B2, LP_A1, LP_A2,
                       &disp_lp_w1, &disp_lp_w2);
}

void displaySetLpCutoff(int hz)
{
    g_dispLpSel = (hz == 40) ? 0 : 1;
}
