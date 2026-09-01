/**
 * @file ecg_replay.cpp
 * @brief 数据库 ECG 回放模块实现 (2026-08-08)
 *
 * 数据: include/signal_generator/ecg_replay_data.h (make_replay_data.py 生成)
 *   - 45s @500Hz, 单位 V (mV×1000, clip ±2V)
 *   - 段 0: MIT-BIH 100 (窦性)   段 1: MIT-BIH 106 (VEB 密集)
 */

#include "signal_generator/ecg_replay.h"
#include "signal_generator/ecg_replay_data.h"

/* ======================== 状态 ======================== */

static uint8_t  s_segment = 0;              /* 当前段 */
static uint32_t s_index   = 0;              /* 段内样本索引 */

/* ======================== 实现 ======================== */

void ecgReplayInit(void)
{
    s_segment = 0;
    s_index   = 0;
}

float ecgReplayNextSample(void)
{
    const float* data;
    uint32_t len;

    if (s_segment == 1) {
        data = ecg_replay_abnormal;
        len  = ECG_REPLAY_ABNORMAL_LEN;
    } else {
        data = ecg_replay_normal;
        len  = ECG_REPLAY_NORMAL_LEN;
    }

    float v = data[s_index];
    s_index++;
    if (s_index >= len) {
        s_index = 0;   /* 循环播放 */
    }
    return v;
}

void ecgReplaySetSegment(uint8_t segment)
{
    if (segment > 1) segment = 1;
    s_segment = segment;
    s_index   = 0;
}

uint8_t ecgReplayGetSegment(void)
{
    return s_segment;
}

void ecgReplayReset(void)
{
    s_index = 0;
}
