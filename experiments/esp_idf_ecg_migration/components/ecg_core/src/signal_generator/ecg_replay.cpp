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

/* 段 2 (M1 平线段): MIT-100 前 15s 正常 -> 45s 平线 (0.2 基线), 60s 循环。
 * 平线模拟电极脱落/拔接头: 无 QRS -> 主循环时间停搏路径 (ALARM_SRC_FLAT)。 */
#define REPLAY_FLAT_LEAD_SAMPLES   (15 * ECG_REPLAY_SR)
#define REPLAY_FLAT_TOTAL_SAMPLES  (60 * ECG_REPLAY_SR)
#define REPLAY_FLAT_BASELINE       0.2f

float ecgReplayNextSample(void)
{
    if (s_segment == ECG_REPLAY_SEG_FLAT) {
        float v = (s_index < REPLAY_FLAT_LEAD_SAMPLES)
                    ? ecg_replay_normal[s_index]
                    : REPLAY_FLAT_BASELINE;
        s_index++;
        if (s_index >= REPLAY_FLAT_TOTAL_SAMPLES) {
            s_index = 0;   /* 循环播放 */
        }
        return v;
    }

    const float* data;
    uint32_t len;

    if (s_segment == ECG_REPLAY_SEG_ABNORMAL) {
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
    if (segment > ECG_REPLAY_SEG_FLAT) segment = ECG_REPLAY_SEG_FLAT;
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
