#ifndef ECG_AI_H
#define ECG_AI_H

/**
 * @file ecg_ai.h
 * @brief ESP-IDF AI 推理组件（esp-tflite-micro + ESP-NN, exp7c INT8）
 *
 * 迁移自 Arduino src/ai_inference/ai_inference.cpp，保留：
 *   - 2:1 输入抽取（500Hz -> 250Hz）
 *   - 因果 0.5Hz 高通（aiApplyFilter 同系数，流式状态跨窗口）
 *   - 250 点窗口（默认每 250 个抽取样本触发一次，可配置 stride/offset）
 *   - Z-score（总体标准差，std<1e-6 置 1）
 *   - INT8 输入填充（round + zero_point + clip）
 *   - exp7c INT8 推理（esp-tflite-micro + ESP-NN）
 *   - 输出反量化直接取 abnormal 概率（不二次 softmax）
 *   - 可配置阈值 / K-of-N / 1-of-N / 冷却合并 / 回调/队列
 *
 * 注意：
 *   1. ecg_ai_feed_sample() 的输入语义与 Arduino ai_inference_push() 一致：
 *      调用方需先提供 500Hz 去直流、双级 10 抽头梳状、HP0.05+LP40 后的样本；
 *      组件内部完成 2:1 抽取和因果 0.5Hz @250Hz 高通。
 *   2. ecg_ai_run_preprocessed() 用于直接喂已经 Z-score 过的 250 点窗口，
 *      主要用于 PC/板端一致性测试。
 */
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ECG_AI_CONFIRM_K_OF_N = 0,
    ECG_AI_CONFIRM_ONE_OF_N = 1,
} ecg_ai_confirm_mode_t;

typedef struct {
    float threshold;          /* abnormal 概率阈值，默认 0.60 */
    int confirm_mode;         /* ECG_AI_CONFIRM_K_OF_N / ECG_AI_CONFIRM_ONE_OF_N */
    int confirm_n;            /* 窗口拍数 N，默认 5 */
    int confirm_k;            /* K-of-N 的 K；1-of-N 可忽略 */
    int cooldown_beats;       /* 报警后冷却拍数，默认 5 */
    int stride;               /* 抽取后触发步进，默认 250（1s） */
    int trigger_offset;       /* 群延迟补偿，默认 6 */
    int arena_size;           /* Tensor Arena 字节数，建议 >=64KB */
    bool use_psram;           /* true 时从 PSRAM 分配 arena */
    int queue_length;         /* 结果队列长度，0 表示只用回调 */
} ecg_ai_config_t;

typedef struct {
    float confidence;         /* abnormal 概率 [0,1] */
    uint8_t raw_abnormal;     /* 单拍超阈值 */
    uint8_t confirmed;        /* 经确认/冷却策略后的报警标志 */
    uint32_t sample_index;    /* 窗口末尾样本计数 */
    uint32_t latency_us;      /* Invoke 耗时（不含预处理） */
} ecg_ai_result_t;

typedef void (*ecg_ai_result_cb_t)(const ecg_ai_result_t *result);

void ecg_ai_config_default(ecg_ai_config_t *cfg);

bool ecg_ai_init(const uint8_t *model_data, size_t model_size,
                 const ecg_ai_config_t *cfg);
void ecg_ai_reset(void);
bool ecg_ai_is_initialized(void);

/* 流式喂入 500Hz（已梳状 + HP0.05 + LP40）样本 */
void ecg_ai_feed_sample(float sample_500hz);

/* 直接跑已经 Z-score 过的 250 点窗口（用于一致性测试） */
bool ecg_ai_run_preprocessed(const float *window_250);

/* 直接跑带 Z-score 的 250 点窗口 */
bool ecg_ai_run_window_raw(const float *window_250);

/* 直接跑已经量化好的 int8 窗口（250 字节，用于测试向量） */
bool ecg_ai_run_int8(const int8_t *input_250);

void ecg_ai_set_result_callback(ecg_ai_result_cb_t cb);
bool ecg_ai_pop_result(ecg_ai_result_t *out);
uint32_t ecg_ai_total_inferences(void);
uint32_t ecg_ai_total_confirmed(void);

#ifdef __cplusplus
}
#endif

#endif /* ECG_AI_H */
