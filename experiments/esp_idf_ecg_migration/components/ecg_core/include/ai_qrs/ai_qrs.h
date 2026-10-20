#ifndef AI_QRS_H
#define AI_QRS_H
/**
 * @file ai_qrs.h
 * @brief AI 通道专用"宽松" QRS 触发 (2026-10-20)
 *
 * 与 heartrate.cpp (心率用严格检测器) 完全独立:
 *  - 不做早搏拒绝: 无 MIN_RR 480ms / 无 0.65×中位RR / 无 0.55×前拍幅度门
 *  - 仅短不应期 (默认 250 ms) + 自适应阈值 (信号/噪声峰 EMA) + 噪声鲁棒处理
 *    (峰值钳位、无拍超时衰减、阈值下限)
 * 输入: 500 Hz、与 hrProcess 相同的 applyFilter() 输出 (0.5–40 Hz)。
 * 输出: fired=true 时, r_lag = 推定 R 峰距"当前样本"的滞后 (500 Hz 样本)。
 * 纯 C, 无平台依赖 → 固件与 PC 回放 (tools/hr_host) 共用同一份源码。
 */
#include <stdint.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int   refractory_samp;  /* 不应期 @500Hz, 默认 125 (250 ms) */
    float thr_ratio;        /* 阈值 = NPK + ratio×(SPK−NPK), 默认 0.05 (训练集调参) */
    float thr_floor_frac;   /* 阈值下限 = floor×SPK, 默认 0.12 */
    int   tguard_samp;      /* T 波保护窗 @500Hz (0=关), 默认 180 (360 ms) */
    float tguard_frac;      /* 保护窗内需 MWI 峰 ≥ frac×SPK */
    int   r_offset_samp;    /* R 定位补偿 @500Hz (正=往前移), 默认 16 (32 ms) */
} AIQRS_Params;

typedef struct {
    bool     fired;
    uint16_t r_lag;         /* 当前样本 − 推定 R 位置 (500 Hz 样本) */
    float    peak;          /* MWI 峰值 */
} AIQRS_Result;

void aiqrsDefaultParams(AIQRS_Params *p);
void aiqrsInit(const AIQRS_Params *p);   /* p=NULL → 默认 */
AIQRS_Result aiqrsProcess(float filteredSample);
uint32_t aiqrsBeatCount(void);

#ifdef __cplusplus
}
#endif
#endif
