/**
 * @file ai_inference.h
 * @brief ECG 异常检测推理模块 API
 *
 * 基于 TFLite Micro 的 1D-CNN 推理引擎
 * 运行在 Core 0 (FreeRTOS 任务), 与 Core 1 主循环解耦
 *
 * 使用方式:
 *   1. 在 main.cpp 中调用 ai_inference_init()
 *   2. 主循环将 250 样本通过 ai_inference_push() 推入
 *   3. 轮询 ai_inference_result_ready() 获取结果
 */

#ifndef AI_INFERENCE_H
#define AI_INFERENCE_H

#include <Arduino.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/* ======================== 常量定义 ======================== */
#define AI_WINDOW_SIZE    250    /* 输入窗口大小 (样本数) */
#define AI_STRIDE         125    /* 步进 (50% 重叠) */
#define AI_MAX_RESULTS    8      /* 最大缓存结果数 */
#define AI_TASK_STACK     8192   /* FreeRTOS 任务栈 (字节) */
#define AI_TASK_PRIORITY  1      /* 任务优先级 (0=最低) */

/* ======================== 推理结果 ======================== */

typedef struct {
    float confidence;       /* 置信度 [0, 1] */
    uint8_t is_abnormal;    /* 1=异常, 0=正常 */
    uint32_t sample_index;  /* 对应样本索引 */
    uint32_t latency_us;    /* 推理耗时 (微秒) */
} ai_result_t;

/* ======================== API ======================== */

/**
 * @brief 初始化推理模块
 * 
 * @return true 成功, false 失败 (模型加载出错)
 * 
 * 必须在 FreeRTOS 调度器启动后 (setup末尾) 调用
 * 自动创建推理任务绑定到 Core 0
 */
bool ai_inference_init(void);

/**
 * @brief 推送一个样本到推理缓冲区
 * 
 * @param value 滤波后 ECG 样本值 (float)
 * 
 * 从 Core 1 主循环调用
 * 当缓冲区累积 window_size 样本后, 自动触发推理
 */
void ai_inference_push(float value);

/**
 * @brief 检查推理结果是否就绪
 * 
 * @return true 有结果可读取
 */
bool ai_inference_result_ready(void);

/**
 * @brief 获取最新的推理结果 (非阻塞)
 * 
 * @param result 输出: 推理结果
 * @return true 成功读取, false 无结果
 */
bool ai_inference_pop_result(ai_result_t* result);

/**
 * @brief 获取推理统计
 * 
 * @param total_inferences 输出: 总推理次数
 * @param total_abnormal   输出: 异常次数
 * @param avg_latency_us   输出: 平均延迟 (us)
 */
void ai_inference_stats(uint32_t* total_inferences,
                        uint32_t* total_abnormal,
                        uint32_t* avg_latency_us);

/**
 * @brief 运行时开关推理
 * 
 * @param enable true=开启, false=暂停
 */
void ai_inference_set_enabled(bool enable);

/**
 * @brief 获取推理是否启用
 */
bool ai_inference_is_enabled(void);

/**
 * @brief 重置推理缓冲区 (切换模式时使用)
 */
void ai_inference_reset(void);

#ifdef __cplusplus
}
#endif

#endif /* AI_INFERENCE_H */
