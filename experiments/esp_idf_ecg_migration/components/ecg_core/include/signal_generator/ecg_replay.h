/**
 * @file ecg_replay.h
 * @brief 数据库 ECG 回放模块 (2026-08-08; M1 2026-09-12 增平线段)
 *
 * 播放 MIT-BIH 真实心电数据 (include/signal_generator/ecg_replay_data.h):
 *   - 段 0: record 100 (窦性心律正常段)
 *   - 段 1: record 106 (VEB 室早密集异常段)
 *   - 段 2: MIT-100 前 15s 正常 -> 45s 平线 (电极脱落等效), 60s 循环 (M1)
 * 用于验证 AI 异常检测的端到端报警效果 (模拟器高斯信号无法触发报警)。
 * 播放速率与固件采样率一致 (500Hz), 循环播放。
 */

#ifndef ECG_REPLAY_H
#define ECG_REPLAY_H

#include <stdint.h>

/** 段编号 (M1) */
#define ECG_REPLAY_SEG_NORMAL   0
#define ECG_REPLAY_SEG_ABNORMAL 1
#define ECG_REPLAY_SEG_FLAT     2

/** 初始化回放模块 (索引归零, 段 0) */
void ecgReplayInit(void);

/** 取当前段下一个样本 (循环播放) */
float ecgReplayNextSample(void);

/** 切换播放段: 0=正常, 1=异常, 2=平线循环 (M1) */
void ecgReplaySetSegment(uint8_t segment);

/** 获取当前播放段 */
uint8_t ecgReplayGetSegment(void);

/** 重置播放索引到段首 */
void ecgReplayReset(void);

#endif /* ECG_REPLAY_H */
