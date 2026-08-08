/**
 * @file ai_inference.cpp
 * @brief ECG 异常检测推理模块实现
 *
 * 架构:
 *   - FreeRTOS 任务运行在 Core 0
 *   - 环形缓冲接收 Core 1 的样本数据
 *   - 每 window_size 样本触发一次 TFLite Micro 推理
 *   - 2:1 输入抽取 (AI_INPUT_DECIMATION): 500Hz 推送 -> 250Hz 有效采样率, 窗口恢复1.0s
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
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

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

/** 从输出解析异常置信度 (模型输出层自带 softmax, 反量化后直接取概率) */
static float parse_output_confidence(void) {
    float val_a;
    if (g_output->type == kTfLiteInt8) {
        float scale = g_output->params.scale;
        int32_t zp = g_output->params.zero_point;
        val_a = (g_output->data.int8[1] - zp) * scale;
    } else {
        val_a = g_output->data.f[1];
    }
    /* M3 验证 (2026-08-05): 模型输出层自带 softmax, TFLite INT8 输出 = 概率量化。
     * 二次 softmax 把概率动态范围压缩至 [0.270, 0.730] (T0-1 发现), 导致阈值语义
     * 漂移; 反量化后直接取异常类概率 = FP32 语义, INFERENCE_THRESHOLD 直接生效。 */
    return val_a;
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
            /* 2026-08-08: 板上实测单次推理 ~910ms (ResNet-L INT8 参考实现),
             * 推理间隔 1s (AI_STRIDE=250)。推理后主动让出 CPU:
             * ① IDLE0 有运行机会, 避免 Task WDT 触发 (IDLE0 饿死 abort);
             * ② Core 0 上 BLE 协议栈任务可抢占/运行, 避免 BLE 饿死。 */
            vTaskDelay(pdMS_TO_TICKS(50));
        }
    }
}


/* ======================== 公开 API ======================== */

bool ai_inference_init(void) {
    g_mutex = xSemaphoreCreateMutex();
    g_data_ready_sem = xSemaphoreCreateBinary();
    g_result_queue = xQueueCreate(AI_MAX_RESULTS, sizeof(ai_result_t));
    if (!g_mutex || !g_data_ready_sem || !g_result_queue) {
        Serial.println("[AI] init fail: sem/queue create");
        return false;
    }

    tflite::InitializeTarget();

    g_model = tflite::GetModel(ecg_model_data);
    Serial.print("[AI] model bytes: ");
    Serial.println((int)sizeof(ecg_model_data));
    Serial.print("[AI] model schema ver: ");
    Serial.print((int)g_model->version());
    Serial.print(" vs lib TFLITE_SCHEMA_VERSION: ");
    Serial.println((int)TFLITE_SCHEMA_VERSION);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[AI] init fail: schema version mismatch");
        return false;
    }

    static tflite::MicroErrorReporter error_reporter;
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interp(
        g_model, resolver, g_tensor_arena, TENSOR_ARENA_SIZE, &error_reporter
    );
    g_interpreter = &static_interp;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[AI] init fail: AllocateTensors (arena maybe too small)");
        return false;
    }
    Serial.print("[AI] arena used: ");
    Serial.println((int)g_interpreter->arena_used_bytes());

    g_input  = g_interpreter->input(0);
    g_output = g_interpreter->output(0);

    if (xTaskCreatePinnedToCore(
            inference_task, "ai_inference", AI_STACK_SIZE,
            nullptr, AI_TASK_PRIO, &g_inference_task, AI_CORE_ID
        ) != pdPASS) {
        Serial.println("[AI] init fail: task create");
        return false;
    }

    Serial.println("[AI] init OK: task created");
    return true;
}

void ai_inference_push(float value) {
    if (!g_inference_enabled) return;

    #if AI_INPUT_DECIMATION > 1
        /* 2:1 抽取: 仅保留偶数样本 (500Hz->250Hz), 使250点窗口=1.0s与训练一致 */
        static uint8_t s_decim_ctr = 0;
        if ((s_decim_ctr++ % AI_INPUT_DECIMATION) != 0) return;
    #endif

    if (xSemaphoreTake(g_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        g_sample_buffer[g_buffer_idx % AI_WINDOW_SIZE] = value;
        g_buffer_idx++;

        /* 群延迟补偿 (T1-3 P0, 2026-08-06): 因果部署链 (梳状+HP0.5+LP40+2:1抽取)
         * 群延迟 ~6 样本 @250Hz (24ms), 使 R 峰在窗口内滞后 6 样本。触发时刻后移
         * AI_TRIGGER_OFFSET 样本 (idx%125==6, 等效评估侧 δ=+6 窗口重提取语义),
         * 令 R 峰回到训练窗口位置, 零成本抵消群延迟错位 (FINAL_RESULTS 表5)。 */
        if ((g_buffer_idx % AI_STRIDE) == AI_TRIGGER_OFFSET
            && g_buffer_idx >= AI_WINDOW_SIZE) {
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

