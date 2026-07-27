/**
 * @file tflite_settings.h
 * @brief TFLite Micro 运行时配置
 *
 * 适配 ESP32-S3 (240MHz, 2MB PSRAM)
 * 
 * 内存预算:
 *   - Tensor Arena: 32KB (模型输入 1KB + 中间特征图 ~25KB)
 *   - 模型权重: ~5KB (INT8, 嵌入 Flash)
 *   - 总计: ~37KB (从 SRAM 分配)
 */

#ifndef TFLITE_SETTINGS_H
#define TFLITE_SETTINGS_H

/* ======================== 模型参数 ======================== */
#define MODEL_INPUT_SIZE    250    /* 输入样本数 */
#define MODEL_INPUT_CHANNELS 1     /* 单通道 ECG */
#define MODEL_OUTPUT_CLASSES 2     /* 二分类: Normal / Abnormal */
#define MODEL_QUANTIZED      1     /* 1 = INT8, 0 = FP32 */

/* ======================== TFLite Micro Arena ======================== */
/* 
 * Tensor Arena 大小:
 *   - 输入张量:  250 * 1 * 1 byte (INT8) = 250 B
 *   - 中间特征:  ~25KB (Conv1D + MaxPool 中间结果)
 *   - 输出张量:  2 * 1 byte (INT8) = 2 B
 *   - TFLite 元数据: ~4KB
 *   - 预留余量: 2.75KB
 *   - 总计: 32 * 1024 = 32768 B
 */
#define TENSOR_ARENA_SIZE    (32 * 1024)   /* 32KB */

/* ======================== 推理配置 ======================== */
#define INFERENCE_THRESHOLD  0.5f    /* 异常判定阈值 (>0.5 = 异常) */
#define INFERENCE_ENABLED    1       /* 默认启用 */

/* ======================== 性能配置 ======================== */
#define AI_CORE_ID          0       /* 推理任务绑定核心 (Core 0) */
#define AI_STACK_SIZE       8192    /* 任务栈 (8KB) */
#define AI_TASK_PRIO        1       /* 优先级 (低于主循环的2) */
#define AI_QUEUE_LENGTH     8       /* 结果队列深度 */

/* ======================== 调试选项 ======================== */
// #define AI_DEBUG_OUTPUT            /* 取消注释以启用串口调试输出 */
// #define AI_PROFILE_LATENCY          /* 取消注释以记录每次推理耗时 */

#endif /* TFLITE_SETTINGS_H */
