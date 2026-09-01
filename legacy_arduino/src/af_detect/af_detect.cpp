/**
 * @file af_detect.cpp
 * @brief 模块3: AF RR 不规则度检测 (T4-8 + 待办#5 10s 快检)
 *
 * 10s RR 环形缓冲窗 → CV + Shannon 熵 → 三档 (0正常/1AF疑似/2无法判定)。
 * SQI 门控: 窗内平均 SQI < 0.5 → 无法判定 (R4 equivocal 范式)。
 * 阈值 (CV>0.08, 熵>1.2) 由 PTB-XL 10s 窗全量验证校准 (FINAL_RESULTS 表6 补充行);
 * 与 PC 原型 eval_rhythm_af.py / eval_rhythm_af_ptbxl.py 一致 (交叉验证)。
 */
#include "af_detect/af_detect.h"

#include <string.h>
#include <math.h>

static float    s_rrBuf[AF_RR_BUF];
static float    s_sqiBuf[AF_RR_BUF];
static uint32_t s_head = 0;
static uint32_t s_count = 0;
static float    s_winStartTime = 0.0f; /* 窗起始时刻 (秒, 相对) */
static float    s_elapsed = 0.0f;      /* 窗内累计时长 */

void afInit(void)
{
    memset(s_rrBuf, 0, sizeof(s_rrBuf));
    memset(s_sqiBuf, 0, sizeof(s_sqiBuf));
    s_head = 0;
    s_count = 0;
    s_winStartTime = 0.0f;
    s_elapsed = 0.0f;
}

void afReset(void)
{
    afInit();
}

static float afShannonEntropy(const float *rr, uint32_t n)
{
    /* 16 bins, 范围 0.3-1.5s */
    uint32_t hist[16] = {0};
    const float lo = 0.3f, hi = 1.5f;
    for (uint32_t i = 0; i < n; i++) {
        float v = rr[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        int b = (int)((v - lo) / (hi - lo) * 16.0f);
        if (b >= 16) b = 15;
        hist[b]++;
    }
    float ent = 0.0f;
    for (int b = 0; b < 16; b++) {
        if (hist[b] == 0) continue;
        float p = (float)hist[b] / (float)n;
        ent -= p * logf(p);
    }
    return ent;
}

AF_Result afProcess(const HR_Result *hr)
{
    AF_Result res;
    res.label = 2;          /* 默认无法判定 */
    res.score = 0.0f;
    res.cv = 0.0f;
    res.entropy = 0.0f;
    res.nRr = 0;
    res.windowReady = false;

    if (hr == NULL || !hr->beatDetected) {
        return res;
    }
    float rr = hr->rrInterval;
    if (rr <= 0.0f || rr > 3.0f) {
        return res;
    }

    /* 写入缓冲 */
    s_rrBuf[s_head] = rr;
    s_sqiBuf[s_head] = hr->sqi;
    s_head = (s_head + 1) % AF_RR_BUF;
    if (s_count < AF_RR_BUF) {
        s_count++;
    }
    s_elapsed += rr;

    /* 窗未满 30s → 无法判定 */
    if (s_elapsed < AF_WIN_S) {
        return res;
    }

    /* 收集窗内 RR (累计 ≥ 30s) */
    float acc = 0.0f;
    float sum = 0.0f, sum2 = 0.0f, sqiSum = 0.0f;
    uint32_t n = 0;
    uint32_t i = (s_head + AF_RR_BUF - s_count) % AF_RR_BUF;
    for (uint32_t k = 0; k < s_count; k++) {
        uint32_t idx = (i + k) % AF_RR_BUF;
        float v = s_rrBuf[idx];
        acc += v;
        sum += v;
        sum2 += v * v;
        sqiSum += s_sqiBuf[idx];
        n++;
        if (acc >= AF_WIN_S) {
            break;
        }
    }
    res.nRr = (uint8_t)(n > 255 ? 255 : n);
    res.windowReady = (acc >= AF_WIN_S);

    if (n < AF_MIN_RR) {
        return res; /* 无法判定 (RR 不足) */
    }

    /* SQI 门控 → 无法判定 */
    if (sqiSum / (float)n < AF_SQI_GATE) {
        return res;
    }

    float mean = sum / (float)n;
    float var = sum2 / (float)n - mean * mean;
    if (var < 0.0f) var = 0.0f;
    float sd = sqrtf(var);
    float cv = (mean > 1e-6f) ? sd / mean : 0.0f;
    float ent = afShannonEntropy(&s_rrBuf[i], n);

    res.cv = cv;
    res.entropy = ent;
    /* 组合分数 (与 PC 原型一致: 0.5*CV/0.2 + 0.5*熵/4.5) */
    res.score = 0.5f * (cv / 0.2f) + 0.5f * (ent / 4.5f);

    if (cv > AF_CV_THR && ent > AF_ENT_THR) {
        res.label = 1; /* AF 疑似 */
    } else {
        res.label = 0; /* 正常 */
    }

    /* 滑窗: 丢弃窗首部 (50% 重叠近似: 丢一半时长) */
    if (s_count > 0) {
        /* 简单实现: 保留后 50% (按时间) */
        float drop = 0.0f;
        uint32_t nKeep = 0;
        for (uint32_t k = 0; k < s_count; k++) {
            uint32_t idx = (i + k) % AF_RR_BUF;
            if (drop >= AF_WIN_S * 0.5f) {
                nKeep++;
            }
            drop += s_rrBuf[idx];
        }
        /* 移动窗口头部指针: 丢弃前若干拍 */
        uint32_t dropN = s_count - nKeep;
        for (uint32_t k = 0; k < dropN; k++) {
            s_rrBuf[i] = 0.0f;
            s_sqiBuf[i] = 0.0f;
            i = (i + 1) % AF_RR_BUF;
        }
        s_count = nKeep;
        s_elapsed -= AF_WIN_S * 0.5f;
        /* 头指针无法直接表达 → 用数组压缩 (小缓冲, 直接搬移) */
        float rrTmp[AF_RR_BUF], sqiTmp[AF_RR_BUF];
        uint32_t m = 0;
        for (uint32_t k = 0; k < s_count; k++) {
            uint32_t idx = (i + k) % AF_RR_BUF;
            rrTmp[m] = s_rrBuf[idx];
            sqiTmp[m] = s_sqiBuf[idx];
            m++;
        }
        memcpy(s_rrBuf, rrTmp, m * sizeof(float));
        memcpy(s_sqiBuf, sqiTmp, m * sizeof(float));
        s_head = m;
    }

    return res;
}
