#ifndef FILTER_H
#define FILTER_H

/**
 * @file filter.h
 * @brief 心电信号数字滤波器模块
 *
 * 级联结构：高通0.5Hz → 低通40Hz → 50Hz陷波
 * 所有级均为二阶直接II型转置结构（biquad）
 *
 * 设计参考 IEC 60601-2-51 家用单导联设备（场景三）
 * 采样率固定 250Hz
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 初始化滤波器模块，重置内部状态
 */
void filterInit(void);

/**
 * @brief 对单个输入样本进行滤波处理
 *
 * 处理流程：原始信号 → 带通滤波器（0.5~40Hz）→ 50Hz 陷波器 → 输出
 *
 * @param inputSample 原始输入样本值
 * @return float 滤波后的输出样本值
 */
float applyFilter(float inputSample);

/**
 * @brief 重置滤波器内部状态，清除记忆效应
 */
void filterReset(void);

#ifdef __cplusplus
}
#endif

#endif /* FILTER_H */
