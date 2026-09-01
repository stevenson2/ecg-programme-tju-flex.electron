#ifndef RHYTHM_SAFETY_H
#define RHYTHM_SAFETY_H

/**
 * @file rhythm_safety.h
 * @brief 模块1: 心律安全逻辑 (T4-8, 纯规则, 秒级)
 *
 * 设计 (consumer_ecg_architecture_plan.md 模块1):
 *   - 停搏   (asystole):   RR 间期 ≥ 4.0s
 *   - 重度过缓 (bradycardia): 30s 滑窗平均 HR < 40 bpm
 *   - 过速   (tachycardia): 30s 滑窗平均 HR > 180 bpm
 *   - SQI 门控: 窗内平均 SQI < 0.5 不触发 (防噪声/电极脱落误报)
 * 输入: HR_Result (heartrate.h), 每帧调用
 * 输出: RS_Result 报警标志 (秒级)
 */

#include <stdint.h>
#include <stdbool.h>
#include "heartrate/heartrate.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===== 参数 (架构计划定稿, 可在编译期覆盖) ===== */
#ifndef RS_ASYSTOLE_RR_S
#define RS_ASYSTOLE_RR_S   4.0f     /* 停搏 RR 阈值 (秒) */
#endif
#ifndef RS_BRADY_BPM
#define RS_BRADY_BPM       40.0f    /* 重度过缓 */
#endif
#ifndef RS_TACHY_BPM
#define RS_TACHY_BPM       180.0f   /* 过速 */
#endif
#ifndef RS_WIN_S
#define RS_WIN_S           30.0f    /* 过缓/过速评估窗 */
#endif
#ifndef RS_SQI_GATE
#define RS_SQI_GATE        0.5f     /* SQI 门控 */
#endif
#ifndef RS_RR_BUF
#define RS_RR_BUF          120      /* RR 缓冲 (30s @ 平均 0.8s/拍 = 37拍, 裕量) */
#endif

typedef struct {
    bool     asystole;        /**< 停搏报警 (RR ≥ 4s) */
    bool     bradycardia;     /**< 重度过缓报警 (30s 窗 < 40bpm) */
    bool     tachycardia;     /**< 过速报警 (30s 窗 > 180bpm) */
    float    hrWindowBpm;     /**< 最近 30s 窗平均心率 (0=窗未满) */
    uint32_t asystoleCount;   /**< 停搏累计次数 */
} RS_Result;

/** @brief 初始化 (setup 调用一次) */
void rsInit(void);

/**
 * @brief 每帧处理 (与 hrProcess 同频调用)
 * @param hr 心率检测结果 (含 rrInterval 秒 / beatDetected / sqi)
 * @return 报警标志
 */
RS_Result rsProcess(const HR_Result *hr);

/** @brief 复位 (信号源切换时调用) */
void rsReset(void);

#ifdef __cplusplus
}
#endif

#endif /* RHYTHM_SAFETY_H */
