/**
 * @file ai_inference.cpp
 * @brief ECG 异常检测推理模块实现
 *
 * 架构:
 *   - FreeRTOS 任务运行在 Core 0
 *   - 环形缓冲接收 Core 1 的样本数据
 *   - 每 window_size 样本触发一次 TFLite Micro 推理
 *   - 结果通过 FreeRTOS 队列发送给主循环
 *
 * 依赖:
 *   - TensorFlowLite_ESP32 库 (tanakamasayuki)
 *   - ecg_model_data.h (由 export.py --pipeline 生成)
 */

#include "ai_inference/ai_inference.h"
#include "ai_inference/tflite_settings.h"

/* TFLite Micro */
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

/* 模型权重 (由 export.py 生成) */
#include "ai_inference/ecg_model_data.h"

/* FreeRTOS */
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include <freertos/semphr.h>

/* ======================== 静态变量 ======================== */

static const tflite::Model*    g_model        = nullptr;
static tflite::MicroInterpreter* g_interpreter = nullptr;
static TfLiteTensor*           g_input        = nullptr;
static TfLiteTensor*           g_output       = nullptr;

/* Tensor Arena (32KB, 16 字节对齐) */
static uint8_t g_tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));

/* 环形样本缓冲 */
static float g_sample_buffer[AI_WINDOW_SIZE];
static volatile uint32_t g_buffer_idx = 0;

/* FreeRTOS 同步原语 */
static QueueHandle_t    g_result_queue    = nullptr;
static SemaphoreHandle_t g_data_ready_sem = nullptr;
static SemaphoreHandle_t g_mutex          = nullptr;

/* 统计 */
static volatile uint32_t g_total_inferences = 0;
static volatile uint32_t g_total_abnormal   = 0;
static volatile uint32_t g_total_latency_us = 0;
static volatile bool     g_inference_enabled = true;

static TaskHandle_t g_inference_task = nullptr;


/* ======================== 内部函数 ======================== */

/** Z-score 归一化 (就地) */
static void preprocess_samples(float* buffer, int length) {
    float sum = 0.0f;
    for (int i = 0; i < length; i++) sum += buffer[i];
    float mean = sum / length;

    float var = 0.0f;
    for (int i = 0; i < length; i++) {
        float diff = buffer[i] - mean;
        var += diff * diff;
    }
    float std_dev = sqrtf(var / length);
    if (std_dev < 1e-6f) std_dev = 1.0f;

    for (int i = 0; i < length; i++) {
        buffer[i] = (buffer[i] - mean) / std_dev;
    }
}

/** 填充 INT8/FP32 输入张量 */
static void fill_input_tensor(const float* buffer) {
    if (g_input->type == kTfLiteInt8) {
        float scale = g_input->params.scale;
        int32_t zero_point = g_input->params.zero_point;
        for (int i = 0; i < AI_WINDOW_SIZE; i++) {
            int32_t q_val = (int32_t)(buffer[i] / scale + 0.5f) + zero_point;
            q_val = constrain(q_val, -128, 127);
            g_input->data.int8[i] = (int8_t)q_val;
        }
    } else {
        for (int i = 0; i < AI_WINDOW_SIZE; i++) {
            g_input->data.f[i] = buffer[i];
        }
    }
}

/** 从输出解析异常置信度 (Softmax) */
static float parse_output_confidence(void) {
    float val_n, val_a;
    if (g_output->type == kTfLiteInt8) {
        float scale = g_output->params.scale;
        int32_t zp = g_output->params.zero_point;
        val_n = (g_output->data.int8[0] - zp) * scale;
        val_a = (g_output->data.int8[1] - zp) * scale;
    } else {
        val_n = g_output->data.f[0];
        val_a = g_output->data.f[1];
    }
    float max_val = max(val_n, val_a);
    float exp_n = expf(val_n - max_val);
    float exp_a = expf(val_a - max_val);
    return exp_a / (exp_n + exp_a);
}

/** 执行单次推理 (含多拍确认滤波) */
static ai_result_t run_single_inference(const float* samples, uint32_t sample_index) {
    ai_result_t result = {0};
    uint32_t t_start = micros();

    float local_buf[AI_WINDOW_SIZE];
    memcpy(local_buf, samples, sizeof(float) * AI_WINDOW_SIZE);
    preprocess_samples(local_buf, AI_WINDOW_SIZE);
    fill_input_tensor(local_buf);

    TfLiteStatus status = g_interpreter->Invoke();
    uint32_t t_elapsed = micros() - t_start;

    if (status != kTfLiteOk) {
        result.confidence = 0.0f;
        result.is_abnormal = 0;
        result.sample_index = sample_index;
        result.latency_us = t_elapsed;
        return result;
    }

    float abnormal_conf = parse_output_confidence();
    bool raw_abnormal = (abnormal_conf > INFERENCE_THRESHOLD);

    /* P0优化: 多拍确认滤波 — 连续MULTI_BEAT_CONFIRM拍异常才报警 */
    static uint32_t s_consecutive_abnormal = 0;
    if (raw_abnormal) {
        s_consecutive_abnormal++;
    } else {
        s_consecutive_abnormal = 0;
    }

    result.confidence = abnormal_conf;
    result.is_abnormal = (s_consecutive_abnormal >= MULTI_BEAT_CONFIRM) ? 1 : 0;
    result.sample_index = sample_index;
    result.latency_us = t_elapsed;
    return result;
}



/** 推理任务 (Core 0) */
static void inference_task(void* pvParameters) {
    (void)pvParameters;
    float samples[AI_WINDOW_SIZE];

    for (;;) {
        if (xSemaphoreTake(g_data_ready_sem, portMAX_DELAY) == pdTRUE) {
            if (!g_inference_enabled) continue;

            if (xSemaphoreTake(g_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
                memcpy(samples, (const void*)g_sample_buffer,
                       sizeof(float) * AI_WINDOW_SIZE);
                uint32_t idx = g_buffer_idx;
                xSemaphoreGive(g_mutex);

                ai_result_t result = run_single_inference(samples, idx);
                g_total_inferences++;
                if (result.is_abnormal) g_total_abnormal++;
                g_total_latency_us += result.latency_us;
                xQueueSend(g_result_queue, &result, 0);
            }
        }
    }
}


/* ======================== 公开 API ======================== */

bool ai_inference_init(void) {
    g_mutex = xSemaphoreCreateMutex();
    g_data_ready_sem = xSemaphoreCreateBinary();
    g_result_queue = xQueueCreate(AI_MAX_RESULTS, sizeof(ai_result_t));
    if (!g_mutex || !g_data_ready_sem || !g_result_queue) {
        return false;
    }

    tflite::InitializeTarget();

    g_model = tflite::GetModel(ecg_model_data);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        return false;
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interp(
        g_model, resolver, g_tensor_arena, TENSOR_ARENA_SIZE
    );
    g_interpreter = &static_interp;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        return false;
    }

    g_input  = g_interpreter->input(0);
    g_output = g_interpreter->output(0);

    if (xTaskCreatePinnedToCore(
            inference_task, "ai_inference", AI_STACK_SIZE,
            nullptr, AI_TASK_PRIO, &g_inference_task, AI_CORE_ID
        ) != pdPASS) {
        return false;
    }

    return true;
}

void ai_inference_push(float value) {
    if (!g_inference_enabled) return;

    if (xSemaphoreTake(g_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        g_sample_buffer[g_buffer_idx % AI_WINDOW_SIZE] = value;
        g_buffer_idx++;

        if ((g_buffer_idx % AI_STRIDE) == 0 && g_buffer_idx >= AI_WINDOW_SIZE) {
            xSemaphoreGive(g_data_ready_sem);
        }
        xSemaphoreGive(g_mutex);
    }
}

bool ai_inference_result_ready(void) {
    if (!g_result_queue) return false;
    return uxQueueMessagesWaiting(g_result_queue) > 0;
}

bool ai_inference_pop_result(ai_result_t* result) {
    if (!g_result_queue || !result) return false;
    return xQueueReceive(g_result_queue, result, 0) == pdTRUE;
}

void ai_inference_stats(uint32_t* ti, uint32_t* ta, uint32_t* al) {
    if (ti) *ti = g_total_inferences;
    if (ta) *ta = g_total_abnormal;
    if (al) *al = (g_total_inferences > 0)
        ? (g_total_latency_us / g_total_inferences) : 0;
}

void ai_inference_set_enabled(bool enable) {
    g_inference_enabled = enable;
    if (!enable) {
        ai_result_t discard;
        while (xQueueReceive(g_result_queue, &discard, 0) == pdTRUE) {}
    }
}

bool ai_inference_is_enabled(void) {
    return g_inference_enabled;
}

void ai_inference_reset(void) {
    if (xSemaphoreTake(g_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        memset(g_sample_buffer, 0, sizeof(g_sample_buffer));
        g_buffer_idx = 0;
        xSemaphoreGive(g_mutex);
    }
    ai_result_t discard;
    while (xQueueReceive(g_result_queue, &discard, 0) == pdTRUE) {}
}

