#ifndef FILTER_H
#define FILTER_H

/**
 * @file filter.h
 * @brief 心电信号数字滤波器模块
 *
 * 级联结构：高通0.5Hz → 低通40Hz
 * 所有级均为二阶直接II型转置结构（biquad）
 *
 * 50Hz/100Hz 工频抑制由 main.cpp 中的双级梳状滤波器提供。
 *
 * 设计参考 IEC 60601-2-51 家用单导联设备（场景三）
 * 采样率固定 500Hz
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
 * 处理流程：原始信号 → HP 0.5Hz → LP 40Hz → 输出
 * 注：50Hz/100Hz 由 main.cpp 双级梳状滤波处理（500/50=10 精确陷零）
 *
 * @param inputSample 原始输入样本值
 * @return float 滤波后的输出样本值
 */
float applyFilter(float inputSample);

/**
 * @brief 重置滤波器内部状态，清除记忆效应
 */
void filterReset(void);

/**
 * @brief 滤波器预热: 消除启动瞬态
 *
 * 用首样本反复迭代使内部状态收敛，消除高通阻尼振荡。
 * 应在开始正式采样循环前调用一次。
 *
 * @param firstSample 第一个有效样本值
 */
void filterWarmup(float firstSample);

#ifdef __cplusplus
}
#endif

#endif /* FILTER_H */
