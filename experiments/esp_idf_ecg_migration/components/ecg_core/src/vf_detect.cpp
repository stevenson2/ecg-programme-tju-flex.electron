#include <stdlib.h>
/**
 * @file vf_detect.cpp
 * @brief 模块2: VF/VT 检测器 (T4-9)
 *
 * 5s 窗 DSP 特征 + 逻辑回归分数 + 连续 2 窗确认。
 * v2 (2026-08-16): 特征与 PC 训练逐位复刻 (4 节全精度 SOS forward-backward +
 * ZCR 主频), 输入由 main.cpp 换算 mV; PC 验证 eval_vf_detect_v2.py
 * (VFDB 留出 Se 0.985 / MIT 对照 Sp 0.888 / CUDB Se 0.921)。
 * 4-10Hz 带通: 4 阶 Butterworth (直接 II 型, 系数来自 scipy butter(2, [4,10], fs=250)).
 */
#include "vf_detect/vf_detect.h"

#include <string.h>
#include <math.h>

static float s_buf[VF_WIN_SAMPLES];     /* 5s 窗缓冲 */
static uint32_t s_pos = 0;              /* 写指针 */
/* 4-10Hz 带通状态: scipy butter(4,[4,10],fs=250) 的 4 个二阶节 (全精度, v2) */
static float s_f1[2], s_f2[2], s_f3[2], s_f4[2];
/* v2 窗内计算工作区 (静态, 避免 15KB 栈) */
static float s_work[VF_WIN_SAMPLES];    /* demeaned 原始窗 */
static float s_fwd[VF_WIN_SAMPLES];     /* 前向滤波结果 / abs 排序暂存 */
static float s_xf[VF_WIN_SAMPLES];      /* 反向滤波最终结果 */
/* 历史窗判定 */
static uint8_t s_lastSuspect = 0;       /* 上一窗是否疑似 */
static VF_Result s_res;

/* scipy.signal.butter(4,[4,10],fs=250, output='sos') 全精度 4 节 (2026-08-16
 * 修正: 旧固件只有 2 节 5 位小数近似, 与 PC 训练失配致 CUDB Se 0.936→0.86) */
static const float SOS4[4][6] = {
    { 2.67349040e-05f,  5.34698080e-05f,  2.67349040e-05f, 1.0f, -1.81135233f, 0.846093824f},
    { 1.0f,  2.0f,  1.0f, 1.0f, -1.87777543f, 0.893812835f},
    { 1.0f, -2.0f,  1.0f, 1.0f, -1.86519459f, 0.922431828f},
    { 1.0f, -2.0f,  1.0f, 1.0f, -1.95573823f, 0.966220895f},
};

static void bpfResetStates(void)
{
    s_f1[0] = s_f1[1] = 0.0f;
    s_f2[0] = s_f2[1] = 0.0f;
    s_f3[0] = s_f3[1] = 0.0f;
    s_f4[0] = s_f4[1] = 0.0f;
}

static float bandpass4(float x)
{
    /* 级联 4 个二阶节 (transposed direct form II) */
    float y = SOS4[0][0] * x + s_f1[0];
    s_f1[0] = SOS4[0][1] * x - SOS4[0][4] * y + s_f1[1];
    s_f1[1] = SOS4[0][2] * x - SOS4[0][5] * y;

    float y2 = SOS4[1][0] * y + s_f2[0];
    s_f2[0] = SOS4[1][1] * y - SOS4[1][4] * y2 + s_f2[1];
    s_f2[1] = SOS4[1][2] * y - SOS4[1][5] * y2;

    float y3 = SOS4[2][0] * y2 + s_f3[0];
    s_f3[0] = SOS4[2][1] * y2 - SOS4[2][4] * y3 + s_f3[1];
    s_f3[1] = SOS4[2][2] * y2 - SOS4[2][5] * y3;

    float y4 = SOS4[3][0] * y3 + s_f4[0];
    s_f4[0] = SOS4[3][1] * y3 - SOS4[3][4] * y4 + s_f4[1];
    s_f4[1] = SOS4[3][2] * y3 - SOS4[3][5] * y4;
    return y4;
}

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

static int cmpFloatAsc(const void* a, const void* b)
{
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < fb) return -1;
    if (fa > fb) return  1;
    return 0;
}

static void compute_features(float *feat)
{
    /* 均值去除 (窗内) */
    float mean = 0.0f;
    for (uint32_t i = 0; i < VF_WIN_SAMPLES; i++) mean += s_buf[i];
    mean /= VF_WIN_SAMPLES;

    float rms = 0.0f;
    for (uint32_t i = 0; i < VF_WIN_SAMPLES; i++) {
        float v = s_buf[i] - mean;
        s_work[i] = v;
        rms += v * v;
    }
    rms = sqrtf(rms / VF_WIN_SAMPLES);

    /* 幅度中位 (与 PC np.median(|x|) 一致; 旧 mean_abs×0.78 近似废弃) */
    for (uint32_t i = 0; i < VF_WIN_SAMPLES; i++) s_fwd[i] = fabsf(s_work[i]);
    qsort(s_fwd, VF_WIN_SAMPLES, sizeof(float), cmpFloatAsc);
    float med_abs = (s_fwd[VF_WIN_SAMPLES / 2 - 1] + s_fwd[VF_WIN_SAMPLES / 2]) * 0.5f;

    /* 4-10Hz 带通: forward → backward (等价 scipy sosfiltfilt padlen=0, v2) */
    bpfResetStates();
    for (uint32_t i = 0; i < VF_WIN_SAMPLES; i++) {
        s_fwd[i] = bandpass4(s_work[i]);
    }
    bpfResetStates();
    for (uint32_t i = VF_WIN_SAMPLES; i-- > 0; ) {
        s_xf[i] = bandpass4(s_fwd[i]);
    }

    /* VF 滤波比 + VF 带 ZCR */
    float vf_e = 0.0f;
    uint32_t zc = 0;
    for (uint32_t i = 0; i < VF_WIN_SAMPLES; i++) {
        vf_e += s_xf[i] * s_xf[i];
        if (i > 0 && ((s_xf[i] > 0) != (s_xf[i - 1] > 0))) zc++;
    }
    float vf_ratio = vf_e / (rms * rms * VF_WIN_SAMPLES + 1e-12f);
    float vf_zcr = (float)zc / VF_WIN_SAMPLES;

    /* 峰谷率 (demeaned 窗局部峰, 与 PC diff 峰计数一致) */
    uint32_t n_pk = 0;
    for (uint32_t i = 1; i + 1 < VF_WIN_SAMPLES; i++) {
        if (s_work[i] > s_work[i - 1] && s_work[i] >= s_work[i + 1]) n_pk++;
    }
    float pv_rate = (float)n_pk / 5.0f;  /* 5s 窗 */

    /* 主频近似: demeaned 零交叉率 → 频率 (PC v2 同公式, 非 FFT) */
    uint32_t zc_raw = 0;
    for (uint32_t i = 1; i < VF_WIN_SAMPLES; i++) {
        if (((s_work[i] > 0) != (s_work[i - 1] > 0))) zc_raw++;
    }
    float dom_freq = (float)zc_raw / VF_WIN_SAMPLES * 125.0f;  /* zcr×fs/2 */

    feat[0] = rms;
    feat[1] = med_abs;
    feat[2] = vf_ratio;
    feat[3] = vf_zcr;
    feat[4] = pv_rate;
    feat[5] = dom_freq;
}

void vfInit(void)
{
    memset(s_buf, 0, sizeof(s_buf));
    bpfResetStates();
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
