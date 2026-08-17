#ifndef HEARTRATE_H
#define HEARTRATE_H

/**
 * @file heartrate.h
 * @brief 板上心率计算模块 (简化 Pan-Tompkins 算法)
 *
 * ========== 算法流程 ==========
 * filteredSample
 *   → 一阶差分 (突出 QRS 高频斜率)
 *   → 平方 (放大 R 波，抑制 T 波)
 *   → 滑动窗口积分 150ms (平滑包络)
 *   → 自适应阈值峰值检测
 *   → RR 间期 → BPM 平滑输出
 *
 * ========== 设计要点 ==========
 * - 输入使用已经过 0.5~40Hz 带通 + 50Hz 陷波处理的信号
 * - 自适应阈值跟踪信号/噪声峰值，不需人工调参
 * - 200ms 不应期防止 T 波误检
 * - BPM 基于最近 8 个 RR 间期中位数平滑
 * - 3 秒无心跳自动复位检测器
 *
 * ========== 集成方式 (main.cpp) ==========
 *   #include "heartrate/heartrate.h"
 *
 *   void setup() {
 *       ...
 *       hrInit();
 *   }
 *
 *   void loop() {
 *       float filteredSample = applyFilter(noisySample);
 *       HR_Result hr = hrProcess(filteredSample);  // 每帧调用
 *
 *       if (hr.beatDetected) {
 *           // LED 闪烁 / 蜂鸣器提示
 *       }
 *
 *       // 发送到 BLE: clean,noisy_no_dc,filtered,bpm
 *       snprintf(csvLine, sizeof(csvLine),
 *                "%.3f,%.3f,%.3f,%u\r\n",
 *                clean, noisy, filtered, hr.bpm);
 *   }
 */

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ======================== 心率结果结构体 ======================== */

/**
 * @brief 心率检测结果 (每帧更新)
 */
typedef struct {
    uint8_t  bpm;            /**< 当前心率 (0~255), 0=尚未锁定 */
    float    rrInterval;     /**< 最近一次 RR 间期 (秒) */
    uint32_t beatCount;      /**< 累计检测心拍数 */
    bool     beatDetected;   /**< 本轮采样是否检测到新心拍 */
    float    confidence;     /**< 置信度 0.0~1.0, <0.5 建议显示 "--" */
    float    sqi;            /**< 信号质量指数 0.0~1.0, ≥0.6 可靠 */
    bool     motionActive;   /**< 是否处于运动/噪声干扰状态 */
} HR_Result;

/* ======================== API ======================== */

/**
 * @brief 初始化心率检测模块
 *
 * 清零所有状态变量和缓冲区。
 * 在 setup() 中调用一次。
 */
void hrInit(void);

/**
 * @brief 处理一个滤波后样本，检测 QRS -> 计算 BPM
 *
 * 必须在每帧 (4ms) 精确调用，依赖采样间隔计算 RR 间期。
 *
 * @param filteredSample  已经过 0.5~40Hz + 50Hz 陷波处理的信号
 * @return HR_Result      检测结果
 */
HR_Result hrProcess(float filteredSample);

/**
 * @brief 复位心率检测器 (保留信号活动状态)
 *
 * 清除 QRS 检测历史与 RR 缓冲区, 但保留信号存在/消失标记。
 * 由 checkSignalActivity 内部调用。
 */
void hrReset(void);

/**
 * @brief 完全复位心率检测器 (含信号活动检测)
 *
 * 输入模式切换 (模拟↔真实AFE) 时必须调用此函数,
 * 清除所有状态包括信号存在标记, 强制重新学习。
 */
void hrFullReset(void);

/**
 * @brief 获取当前信号质量指数 (SQI)
 *
 * SQI 基于信号-噪声比计算, 0.0=纯噪声, 1.0=理想信号。
 * ≥0.6 表示信号质量可接受; <0.4 建议冻结 BPM 输出。
 *
 * @return SQI 值 0.0~1.0
 */
float hrGetSQI(void);

/**
 * @brief 查询当前是否处于运动/高噪声干扰状态
 *
 * 当连续多帧 SQI < 0.35 时置位, SQI > 0.55 时清除。
 * 运动状态下自动冻结阈值更新, 防止阈值漂移。
 *
 * @return true=运动干扰中, false=正常
 */
bool hrIsMotionActive(void);

/**
 * @brief 最近一次有效心拍的 millis() 时间戳 (0=尚无有效拍)
 *
 * 供 VF/VT 报警互锁使用: 真 VF/VT 下 QRS 检测无法连续检出有效拍,
 * 距最近有效拍 >2.5s 才放行 VF 报警, 抑制正常窦律下的 VF 误报
 * (2026-08-16 AFE 实测: 正常窦律 VF v2 仍 12 次/60s 误报)。
 */
uint32_t hrGetLastBeatMillis(void);

#ifdef __cplusplus
}
#endif

#endif /* HEARTRATE_H */
