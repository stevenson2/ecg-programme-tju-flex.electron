/* ai_qrs.c — AI 通道宽松 QRS 触发, 见 ai_qrs.h */
#include "ai_qrs/ai_qrs.h"
#include <string.h>
#include <math.h>

/* 8–25 Hz 带通 (与 heartrate.cpp 相同系数, fs=500): HP8 → LP25 */
#define HP_B0  0.931379f
#define HP_B1 -1.862758f
#define HP_B2  0.931379f
#define HP_A1 -1.858043f
#define HP_A2  0.867472f
#define LP_B0  0.020083f
#define LP_B1  0.040167f
#define LP_B2  0.020083f
#define LP_A1 -1.561018f
#define LP_A2  0.641352f

#define MWI_N        40      /* 80 ms */
#define LEARN_SAMP   1000    /* 2 s 学习 */
#define QRS_MAX_SAMP 75      /* 150 ms: 单个 QRS 搜索上限 */
#define RBUF_N       128     /* |bp| 历史, 用于回找 R 位置 */
#define NO_BEAT_DECAY_SAMP 1250 /* 2.5 s 无拍 → SPK 衰减 */
#define SPK_W   0.125f
#define NPK_W   0.125f

static AIQRS_Params P;
static float hw1, hw2, lw1, lw2;
static float mwiBuf[MWI_N]; static int mwiIdx; static float mwiSum;
static float absBuf[RBUF_N];
static uint32_t n;            /* 样本计数 */
static float SPK, NPK, learnMax, learnSum;
static bool  inQrs; static float qMax; static uint32_t qStart, qMaxAt;
static uint32_t lastR; static bool haveR;
static float prevMwi, prevprevMwi;
static uint32_t beats;
#define PKH_N 8
static float pkHist[PKH_N]; static int pkCnt, pkIdx;
static float spk_robust(void) {           /* 近 8 拍峰值中位数: 抗 PVC 大峰/伪影 */
    float t[PKH_N]; int k = pkCnt < PKH_N ? pkCnt : PKH_N;
    for (int i = 0; i < k; i++) t[i] = pkHist[i];
    for (int i = 1; i < k; i++) { float v = t[i]; int j = i - 1; while (j >= 0 && t[j] > v) { t[j + 1] = t[j]; j--; } t[j + 1] = v; }
    return (k % 2) ? t[k / 2] : 0.5f * (t[k / 2 - 1] + t[k / 2]);
}

void aiqrsDefaultParams(AIQRS_Params *p) {
    p->refractory_samp = 125;   /* 250 ms (INCART train 调参, 17_tune_aiqrs_train.py) */
    p->thr_ratio = 0.05f;
    p->thr_floor_frac = 0.12f;
    p->tguard_samp = 180;       /* 360 ms T 波保护窗 (按 SPK 中位数, 非前拍幅度) */
    p->tguard_frac = 0.5f;
    p->r_offset_samp = 16;
}

void aiqrsInit(const AIQRS_Params *p) {
    if (p) P = *p; else aiqrsDefaultParams(&P);
    hw1 = hw2 = lw1 = lw2 = 0.0f;
    memset(mwiBuf, 0, sizeof(mwiBuf)); mwiIdx = 0; mwiSum = 0.0f;
    memset(absBuf, 0, sizeof(absBuf));
    n = 0; SPK = NPK = learnMax = learnSum = 0.0f;
    inQrs = false; qMax = 0.0f; qStart = qMaxAt = 0;
    lastR = 0; haveR = false; prevMwi = prevprevMwi = 0.0f; beats = 0;
    memset(pkHist, 0, sizeof(pkHist)); pkCnt = 0; pkIdx = 0;
}

uint32_t aiqrsBeatCount(void) { return beats; }

static float bq(float x, float b0, float b1, float b2, float a1, float a2, float *w1, float *w2) {
    float w = x - a1 * (*w1) - a2 * (*w2);
    float y = b0 * w + b1 * (*w1) + b2 * (*w2);
    *w2 = *w1; *w1 = w; return y;
}

AIQRS_Result aiqrsProcess(float x) {
    AIQRS_Result r = {false, 0, 0.0f};
    float y = bq(x, HP_B0, HP_B1, HP_B2, HP_A1, HP_A2, &hw1, &hw2);
    y = bq(y, LP_B0, LP_B1, LP_B2, LP_A1, LP_A2, &lw1, &lw2);
    float e = y * y;
    mwiSum += e - mwiBuf[mwiIdx]; mwiBuf[mwiIdx] = e; mwiIdx = (mwiIdx + 1) % MWI_N;
    if (mwiSum < 0.0f) mwiSum = 0.0f;
    float m = mwiSum / (float)MWI_N;
    absBuf[n % RBUF_N] = fabsf(y);
    n++;

    if (n <= LEARN_SAMP) {           /* 学习期: 不触发 */
        if (m > learnMax) learnMax = m;
        learnSum += m;
        if (n == LEARN_SAMP) { SPK = 0.5f * learnMax; NPK = 0.5f * learnSum / (float)LEARN_SAMP; }
        prevprevMwi = prevMwi; prevMwi = m;
        return r;
    }

    if (NPK > 0.25f * SPK) NPK = 0.25f * SPK;   /* 防止漏检拍被当噪声峰抬高阈值 (自锁) */
    float thr = NPK + P.thr_ratio * (SPK - NPK);
    if (thr < P.thr_floor_frac * SPK) thr = P.thr_floor_frac * SPK;

    /* 无拍超时: 大伪影把 SPK 抬高后逐步恢复 */
    if (haveR && (n - lastR) > NO_BEAT_DECAY_SAMP && ((n - lastR) % 250u) == 0u) {
        SPK *= 0.7f; if (SPK < NPK * 1.5f) SPK = NPK * 1.5f;
        for (int i = 0; i < PKH_N; i++) pkHist[i] *= 0.7f;
    }

    bool refr = haveR && (n - lastR) < (uint32_t)P.refractory_samp;
    if (!inQrs) {
        if (!refr && m > thr) { inQrs = true; qMax = m; qStart = n; qMaxAt = n; }
        else if (prevMwi > prevprevMwi && prevMwi >= m && prevMwi < thr) {
            NPK = NPK + NPK_W * (prevMwi - NPK);      /* 噪声峰 */
        }
    } else {
        if (m > qMax) { qMax = m; qMaxAt = n; }
        if (m < 0.5f * qMax || (n - qStart) >= QRS_MAX_SAMP) {
            inQrs = false;
            /* R 定位: 在 [qStart−MWI_N, qMaxAt] 内找 |带通| 最大点 */
            uint32_t lo = (qStart > (uint32_t)MWI_N) ? qStart - MWI_N : 1u;
            if (n - lo >= RBUF_N) lo = n - RBUF_N + 1;
            uint32_t best = qMaxAt; float bv = -1.0f;
            for (uint32_t k = lo; k <= qMaxAt; k++) {
                float v = absBuf[(k - 1) % RBUF_N];   /* 样本 k 存于 (k−1) */
                if (v > bv) { bv = v; best = k; }
            }
            int32_t rpos = (int32_t)best - P.r_offset_samp;
            if (rpos < 1) rpos = 1;
            bool tg = (P.tguard_samp > 0 && haveR && (uint32_t)rpos - lastR < (uint32_t)P.tguard_samp
                       && qMax < P.tguard_frac * SPK);
            if (haveR && (uint32_t)rpos <= lastR + (uint32_t)P.refractory_samp / 2u) tg = true;
            if (!tg) {
                pkHist[pkIdx] = qMax; pkIdx = (pkIdx + 1) % PKH_N; if (pkCnt < PKH_N) pkCnt++;
                if (pkCnt >= 3) SPK = spk_robust();
                else { float pk = qMax; if (SPK > 0.0f && pk > 3.0f * SPK) pk = 3.0f * SPK; SPK = SPK + SPK_W * (pk - SPK); }
                lastR = (uint32_t)rpos; haveR = true; beats++;
                r.fired = true; r.r_lag = (uint16_t)(n - (uint32_t)rpos); r.peak = qMax;
            } else {
                NPK = NPK + NPK_W * (qMax - NPK);
            }
        }
    }
    prevprevMwi = prevMwi; prevMwi = m;
    return r;
}
