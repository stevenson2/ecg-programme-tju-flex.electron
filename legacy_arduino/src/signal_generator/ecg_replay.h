/**
 * @file ecg_replay.h
 * @brief 数据库 ECG 回放模块 (2026-08-08)
 *
 * 播放 MIT-BIH 真实心电数据 (include/signal_generator/ecg_replay_data.h):
 *   - 段 0: record 100 (窦性心律正常段)
 *   - 段 1: record 106 (VEB 室早密集异常段)
 * 用于验证 AI 异常检测的端到端报警效果 (模拟器高斯信号无法触发报警)。
 * 播放速率与固件采样率一致 (500Hz), 循环播放。
 */

#ifndef ECG_REPLAY_H
#define ECG_REPLAY_H

#include <stdint.h>

/** 初始化回放模块 (索引归零, 段 0) */
void ecgReplayInit(void);

/** 取当前段下一个样本 (循环播放) */
float ecgReplayNextSample(void);

/** 切换播放段: 0=正常 (MIT-BIH 100), 1=异常 (MIT-BIH 106) */
void ecgReplaySetSegment(uint8_t segment);

/** 获取当前播放段 */
uint8_t ecgReplayGetSegment(void);

/** 重置播放索引到段首 */
void ecgReplayReset(void);

#endif /* ECG_REPLAY_H */
