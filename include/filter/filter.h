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

/* ======================== AI 输入链独立高通 (2026-08-13 修正) ========================
 * P0-2 (TH §42/§43): AI 输入链在 2:1 抽取后的 250Hz 流上做因果 0.5Hz 高通 (fs=250
 * 修正系数), 与训练侧重训链 causal_hp_05_fs250 位级一致 (零初始状态 streaming)。
 * 由 aiApplyFilter 逐样本调用 (ai_inference_push 内), 状态跨窗口持续。
 * 替代原"窗口级零相位" aiApplyFilterWindow (其系数 fs=500 设计致 0.25Hz 有效截止 bug)。
 */

/** @brief 初始化 AI 输入链高通 (重置状态) */
void aiFilterInit(void);

/** @brief AI 输入链高通滤波 (0.5Hz @250Hz 因果, 匹配训练分布) */
float aiApplyFilter(float inputSample);

/** @brief 重置 AI 输入链高通状态 (输入源切换时调用) */
void aiFilterReset(void);

/* ======================== AI 输入链独立滤波 (2026-08-13 解耦) ========================
 * 显示链 HP 0.5Hz (基线稳定) 与 AI 链解耦: AI 输入在 2:1 抽取前独立做
 * HP 0.05Hz + LP 40Hz, 与训练侧 exp7 复刻链位级一致。 */

/** @brief AI 输入链独立滤波 (HP 0.05Hz + LP 40Hz, 与显示链解耦) */
float applyFilterAI(float inputSample);

/** @brief 重置 AI 输入链独立滤波状态 */
void aiChainFilterReset(void);

/* ======================== 显示链 (2026-08-14) ========================
 * 用户验收反馈: 显示基线"斜 + 毛糙" (§45 去 HP/LP 后显示梳状后原始)。
 * 显示链独立处理: 两级中值基线去除 (0.2s/0.6s, de Chazal 2004 经典法,
 * 保留 QRS/ST 形态且无高通相位失真) + LP 40Hz 平滑。仅用于串口/BLE 显示列,
 * AI 链 (applyFilterAI) 与心率/VF 链 (applyFilter) 不受影响。 */

/** @brief 显示链滤波: 中值基线去除 + LP40 (输入=梳状后原始) */
float applyDisplayFilter(float inputSample);

/** @brief 重置显示链状态 */
void displayFilterReset(void);

/** @brief 显示链 LP 截止频率切换 (4=镜面平滑试验, 40=形态保真默认) */
void displaySetLpCutoff(int hz);

#ifdef __cplusplus
}
#endif

#endif /* FILTER_H */
