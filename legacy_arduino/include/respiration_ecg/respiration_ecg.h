#ifndef RESPIRATION_ECG_H
#define RESPIRATION_ECG_H

/**
 * @file respiration_ecg.h
 * @brief 利用 ADS1292R 呼吸阻抗通道抑制 ECG 呼吸基线漂移
 *
 * 原理：
 *   呼吸阻抗信号与 ECG 上的呼吸基线漂移高度相关。
 *   以呼吸阻抗解调信号为参考，使用归一化 LMS 自适应滤波器估计
 *   ECG 中的呼吸成分，并从 ECG 中减去，从而减小呼吸对 ECG 的影响。
 *
 * 输出 correctedEcg 会继续进入心率、显示、AI 和 BLE/串口链路。
 */

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float correctedEcg;  /**< 去除呼吸成分后的 ECG (V) */
    float estimate;      /**< 本次估计并减去的呼吸干扰分量 (V) */
    float coeff;         /**< 当前自适应滤波器系数 (调试用) */
    bool  active;        /**< 当前是否正在做有效自适应 */
} RespEcgCancelResult;

void respEcgCancelInit(void);
RespEcgCancelResult respEcgCancelProcess(float ecgRaw, float respRef);
void respEcgCancelReset(void);

#ifdef __cplusplus
}
#endif

#endif /* RESPIRATION_ECG_H */
