#ifndef THERMAL_H
#define THERMAL_H

/**
 * @file thermal.h
 * @brief ESP32-S3 内置温度传感器监测模块
 *
 * 利用 ESP32-S3 芯片内置温度传感器 (temperatureRead())
 * 实时监测芯片温度，支持滑动平均、最值记录、过温告警。
 *
 * 使用场景：
 *   - 可穿戴设备温度监测 (目标 ≤ 40°C)
 *   - 过温保护 (≥ 65°C 触发降功率)
 *   - 调试阶段量化功耗-温度关系
 *
 * 采样策略：
 *   - 每秒采样一次 (由外部 timer 驱动)
 *   - 8 点环形缓冲滑动平均，抑制 ±1°C 噪声
 */

#ifdef __cplusplus
extern "C" {
#endif

/** 温度告警级别 */
typedef enum {
    THERMAL_OK       = 0,  /**< 温度正常 (≤ 55°C) */
    THERMAL_WARN     = 1,  /**< 高温警告 (> 55°C, 建议检查) */
    THERMAL_CRITICAL = 2   /**< 过热 (> 65°C, 需立即降功率) */
} ThermalAlertLevel;

/** 温度状态快照 */
typedef struct {
    float            current;     /**< 当前温度 (°C) */
    float            avg;         /**< 8 点滑动平均 (°C) */
    float            min;         /**< 启动以来最低 (°C) */
    float            max;         /**< 启动以来最高 (°C) */
    ThermalAlertLevel alertLevel;  /**< 告警级别 */
    unsigned long    uptimeMs;    /**< 传感器已运行时间 (ms) */
} ThermalState;

/**
 * @brief 初始化温度监测模块
 *
 * 填充滑动缓冲区初始值，清除 min/max 记录。
 * 应在 setup() 中调用一次。
 */
void thermalInit(void);

/**
 * @brief 更新温度状态 (每秒调用一次)
 *
 * 执行一次温度读取，更新 8 点滑动平均，刷新 min/max 和告警级别。
 *
 * @return ThermalState 当前温度状态快照
 */
ThermalState thermalUpdate(void);

/**
 * @brief 获取可读告警字符串
 *
 * @return const char* 告警文本 (正常/高温/过热)
 */
const char* thermalGetAlertString(void);

/**
 * @brief 一键打印完整温度状态到串口
 *
 * 格式示例:
 * [温度] 当前: 42.3°C | 平均: 41.8°C | 范围: 38.2~43.1°C | 状态: 正常
 */
void thermalPrintStatus(void);

#ifdef __cplusplus
}
#endif

#endif /* THERMAL_H */