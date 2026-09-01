#ifndef RESPIRATION_H
#define RESPIRATION_H

/**
 * @file respiration.h
 * @brief 呼吸率检测模块 (基于 ADS1292R 呼吸阻抗通道)
 *
 * 输入: 500Hz 呼吸阻抗解调信号 (V)
 * 算法: 移动平均去基线 + 低通平滑 + 正向过零检测
 * 输出: 每分钟呼吸次数 (brpm)
 */

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float    bpm;        /**< 呼吸率 (次/分), 0=尚未稳定 */
    float    amplitude;  /**< 最近呼吸波峰峰值 (V) */
    uint32_t breathCount;/**< 累计检测到呼吸次数 */
    bool     valid;      /**< 当前呼吸率是否有效 */
} Resp_Result;

void respInit(void);
Resp_Result respProcess(float rawResp);
void respReset(void);

#ifdef __cplusplus
}
#endif

#endif /* RESPIRATION_H */
