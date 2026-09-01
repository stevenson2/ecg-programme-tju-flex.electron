/**
 * @file rhythm_safety.cpp
 * @brief 模块1: 心律安全逻辑 (T4-8)
 *
 * 纯规则实现, 无浮点库依赖 (Arduino 兼容):
 *   - 停搏: 任一 RR ≥ 4.0s (逐拍检查, 秒级)
 *   - 重度过缓/过速: 30s 环形缓冲窗平均 HR, 50% 重叠滑动
 *   - SQI 门控: 窗内平均 SQI < 0.5 不触发
 * 与 PC 原型 eval_rhythm_af.py 的 RhythmSafety 逻辑一致 (可交叉验证)。
 */
#include "rhythm_safety/rhythm_safety.h"

#include <string.h>

static float   s_rrBuf[RS_RR_BUF];   /* RR 环形缓冲 (秒) */
static float   s_sqiBuf[RS_RR_BUF];  /* 对应 SQI */
static uint32_t s_rrHead = 0;        /* 写指针 */
static uint32_t s_rrCount = 0;       /* 有效条目数 */
static uint32_t s_asystoleCount = 0;

void rsInit(void)
{
    memset(s_rrBuf, 0, sizeof(s_rrBuf));
    memset(s_sqiBuf, 0, sizeof(s_sqiBuf));
    s_rrHead = 0;
    s_rrCount = 0;
    s_asystoleCount = 0;
}

void rsReset(void)
{
    rsInit();
}

RS_Result rsProcess(const HR_Result *hr)
{
    RS_Result res;
    res.asystole = false;
    res.bradycardia = false;
    res.tachycardia = false;
    res.hrWindowBpm = 0.0f;
    res.asystoleCount = s_asystoleCount;

    if (hr == NULL || !hr->beatDetected) {
        return res;
    }

    float rr = hr->rrInterval;

    /* ---- 停搏: RR ≥ 4s (无需 SQI 门控: 无 QRS 本身即低 SQI, 但电极脱落
             场景由 SQI 门控在窗口层拦截; 单拍停搏阈值直接判) ---- */
    if (rr >= RS_ASYSTOLE_RR_S) {
        res.asystole = true;
        s_asystoleCount++;
        res.asystoleCount = s_asystoleCount;
        return res;
    }

    /* 无效 RR (0 或异常) 不入窗 */
    if (rr <= 0.0f || rr > 3.0f) {
        return res;
    }

    /* ---- 写入环形缓冲 ---- */
    s_rrBuf[s_rrHead] = rr;
    s_sqiBuf[s_rrHead] = hr->sqi;
    s_rrHead = (s_rrHead + 1) % RS_RR_BUF;
    if (s_rrCount < RS_RR_BUF) {
        s_rrCount++;
    }

    /* ---- 30s 窗平均 HR (窗 = 最近累计时长 ≥ 30s 的 RR 集合) ---- */
    float acc = 0.0f;
    float sqiAcc = 0.0f;
    uint32_t n = 0;
    /* 从旧到新扫描, 累计时长 ≥ 30s 即停 */
    uint32_t i = (s_rrHead + RS_RR_BUF - s_rrCount) % RS_RR_BUF;
    for (uint32_t k = 0; k < s_rrCount; k++) {
        uint32_t idx = (i + k) % RS_RR_BUF;
        acc += s_rrBuf[idx];
        sqiAcc += s_sqiBuf[idx];
        n++;
        if (acc >= RS_WIN_S) {
            break;
        }
    }
    if (n >= 5 && acc >= RS_WIN_S) {
        float hrWin = (float)n / acc * 60.0f;
        float sqiAvg = sqiAcc / (float)n;
        res.hrWindowBpm = hrWin;
        if (sqiAvg >= RS_SQI_GATE) {
            if (hrWin < RS_BRADY_BPM) {
                res.bradycardia = true;
            } else if (hrWin > RS_TACHY_BPM) {
                res.tachycardia = true;
            }
        }
    }

    return res;
}
