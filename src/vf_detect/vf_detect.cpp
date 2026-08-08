/**
 * @file vf_detect.cpp
 * @brief 模块2: VF/VT 检测器 (T4-9)
 *
 * 5s 窗 DSP 特征 + 逻辑回归分数 + 连续 2 窗确认。
 * 特征与 PC 原型 eval_vf_detect.py 一致 (PC 验证 Se 0.957/Sp 0.824)。
 * 4-10Hz 带通: 4 阶 Butterworth (直接 II 型, 系数来自 scipy butter(2, [4,10], fs=250)).
 */
#include "vf_detect/vf_detect.h"

#include <string.h>
#include <math.h>

static float s_buf[VF_WIN_SAMPLES];     /* 5s 窗缓冲 */
static uint32_t s_pos = 0;              /* 写指针 */
/* 4-10Hz 带通状态 (直接 II 型 4 阶: 2 个二阶节) */
static float s_f1[2], s_f2[2];
/* 历史窗判定 */
static uint8_t s_lastSuspect = 0;       /* 上一窗是否疑似 */
static VF_Result s_res;

/* 4-10Hz 带通 (biquad 级联, scipy butter(2,[4,10]/125)):
   b = [0.00512927, 0, -0.01025854, 0, 0.00512927]
   a = [1, -3.73959554, 5.2925656, -3.36034609, 0.80794959]
   分解为 2 个二阶节 (scipy sos 近似): */
static const float SOS[2][6] = {
    {0.00512927f, 0.0f, -0.00512927f, 1.0f, -1.90644760f, 0.95561144f},
    {1.0f, 0.0f, -1.0f, 1.0f, -1.83314794f, 0.84572785f},
};

static float bandpass4(float x)
{
    /* 级联 2 个二阶 (transposed direct form II) */
    float y1 = SOS[0][0] * x + s_f1[0];
    s_f1[0] = SOS[0][1] * x - SOS[0][4] * y1 + s_f1[1];
    s_f1[1] = SOS[0][2] * x - SOS[0][5] * y1;
    float y2 = SOS[1][0] * y1 + s_f2[0];
    s_f2[0] = SOS[1][1] * y1 - SOS[1][4] * y2 + s_f2[1];
    s_f2[1] = SOS[1][2] * y1 - SOS[1][5] * y2;
    return y2;
}

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

static void compute_features(float *feat)
{
    /* 均值去除 (窗内) */
    float mean = 0.0f;
    for (uint32_t i = 0; i < VF_WIN_SAMPLES; i++) mean += s_buf[i];
    mean /= VF_WIN_SAMPLES;

    float rms = 0.0f, sum_abs = 0.0f;
    float xf[VF_WIN_SAMPLES];
    for (uint32_t i = 0; i < VF_WIN_SAMPLES; i++) {
        float v = s_buf[i] - mean;
        rms += v * v;
        sum_abs += fabsf(v);
        xf[i] = bandpass4(v);
    }
    rms = sqrtf(rms / VF_WIN_SAMPLES);
    /* 幅度中位 (近似: 排序太贵 → 用均值绝对值的缩放近似, PC 端 med_abs≈0.78×mean_abs) */
    float mean_abs = sum_abs / VF_WIN_SAMPLES;

    /* VF 滤波比 + VF 带 ZCR */
    float vf_e = 0.0f;
    uint32_t zc = 0;
    for (uint32_t i = 0; i < VF_WIN_SAMPLES; i++) {
        vf_e += xf[i] * xf[i];
        if (i > 0 && ((xf[i] > 0) != (xf[i - 1] > 0))) zc++;
    }
    float vf_ratio = vf_e / (rms * rms * VF_WIN_SAMPLES + 1e-12f);
    float vf_zcr = (float)zc / VF_WIN_SAMPLES;

    /* 峰谷率 (原始信号局部峰) */
    uint32_t n_pk = 0;
    for (uint32_t i = 1; i + 1 < VF_WIN_SAMPLES; i++) {
        if (s_buf[i] > s_buf[i - 1] && s_buf[i] >= s_buf[i + 1]) n_pk++;
    }
    float pv_rate = (float)n_pk / 5.0f;  /* 5s 窗 */

    /* 主频近似: 原始信号零交叉率 → 频率 (正弦近似 f = zcr_samples/2) */
    uint32_t zc_raw = 0;
    for (uint32_t i = 1; i < VF_WIN_SAMPLES; i++) {
        if (((s_buf[i] - mean) > 0) != ((s_buf[i - 1] - mean) > 0)) zc_raw++;
    }
    float dom_freq = (float)zc_raw / VF_WIN_SAMPLES * 125.0f;  /* zcr×fs/2 */

    feat[0] = rms;
    feat[1] = mean_abs * 0.78f;  /* med_abs 近似 */
    feat[2] = vf_ratio;
    feat[3] = vf_zcr;
    feat[4] = pv_rate;
    feat[5] = dom_freq;
}

void vfInit(void)
{
    memset(s_buf, 0, sizeof(s_buf));
    memset(s_f1, 0, sizeof(s_f1));
    memset(s_f2, 0, sizeof(s_f2));
    s_pos = 0;
    s_lastSuspect = 0;
    memset(&s_res, 0, sizeof(s_res));
}

void vfReset(void)
{
    vfInit();
}

VF_Result vfProcess(float sample)
{
    s_res.vfAlarm = false;
    s_res.windowSuspect = false;

    s_buf[s_pos++] = sample;
    if (s_pos < VF_WIN_SAMPLES) {
        return s_res;  /* 窗未满 */
    }
    s_pos = 0;  /* 每窗计算一次 (非滑动; 5s 粒度 + 2 窗确认 = 时延 ≤10s) */

    /* 特征 + 标准化 + 逻辑回归 */
    float feat[VF_FEATURES];
    compute_features(feat);
    const float mean_[6] = {VF_MEAN_0, VF_MEAN_1, VF_MEAN_2, VF_MEAN_3, VF_MEAN_4, VF_MEAN_5};
    const float std_[6] = {VF_STD_0, VF_STD_1, VF_STD_2, VF_STD_3, VF_STD_4, VF_STD_5};
    const float coef[6] = {VF_COEF_0, VF_COEF_1, VF_COEF_2, VF_COEF_3, VF_COEF_4, VF_COEF_5};
    float z = VF_INTERCEPT;
    for (int i = 0; i < VF_FEATURES; i++) {
        z += coef[i] * ((feat[i] - mean_[i]) / std_[i]);
    }
    float p = sigmoid(z);
    s_res.score = p;
    s_res.lastRms = feat[0];

    uint8_t suspect = (p >= VF_THETA) ? 1 : 0;
    s_res.windowSuspect = (suspect == 1);
    /* 连续 2 窗确认 */
    if (suspect && s_lastSuspect) {
        s_res.vfAlarm = true;
    }
    s_lastSuspect = suspect;
    return s_res;
}
