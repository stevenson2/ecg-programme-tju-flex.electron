#ifndef ECG_SIMULATOR_H
#define ECG_SIMULATOR_H

/**
 * @file ecg_simulator.h
 * @brief 带噪声的真实心电信号生成模块
 *
 * 在无外部模拟前端电路的情况下，模拟生成真实心电波形，
 * 用于验证滤波算法和蓝牙传输功能。
 *
 * 模拟信号包含：
 * - 典型心电波形（P波、QRS波群、T波）- 高斯脉冲叠加
 * - 基线漂移噪声（0.1~0.5Hz 低频正弦波）
 * - 50Hz 工频干扰
 * - 肌电噪声（高斯白噪声）
 *
 * 采样率：250Hz
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 初始化心电信号生成器
 */
void ecgSimulatorInit(void);

/**
 * @brief 生成一个带噪声的模拟心电样本
 * @return float 模拟的心电信号样本值（单位：mV），范围约 ±2mV
 */
float generateECGSample(void);

/**
 * @brief 获取当前拍纯净心电信号（无噪声，仅P-QRS-T波形）
 * @return float 纯净心电值
 */
float getCleanECGValue(void);

/**
 * @brief 重置信号生成器状态
 */
void ecgSimulatorReset(void);

/**
 * @brief 获取模拟器真实心率 (BPM)
 *
 * 基于 CYCLE_LENGTH 计算: BPM = 60 × 250 / CYCLE_LENGTH
 * 默认 200 样本/拍 = 75 BPM
 *
 * @return uint8_t 真实心率 (30~200)
 */
uint8_t ecgSimulatorGetTrueBPM(void);

/**
 * @brief 获取模拟器一个心拍的样本数
 *
 * 用于外部模块计算预期 RR 间期
 *
 * @return uint16_t 每拍样本数
 */
uint16_t ecgSimulatorGetCycleLength(void);

#ifdef __cplusplus
}
#endif

#endif /* ECG_SIMULATOR_H */
