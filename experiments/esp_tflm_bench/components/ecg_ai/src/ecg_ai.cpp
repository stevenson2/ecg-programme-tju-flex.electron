#include "ecg_ai.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <new>
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr int kMaxConfirmN = 64;
constexpr int kDefaultWindow = 250;

/* AI 输入链因果 0.5Hz @250Hz 系数（与 include/filter/filter.h 完全一致） */
constexpr double kAiHpA1 = -1.9822289297925284;
constexpr double kAiHpA2 =  0.98238545061412508;
constexpr double kAiHpB0 =  0.99115359510166301;
constexpr double kAiHpB1 = -1.982307190203326;
constexpr double kAiHpB2 =  0.99115359510166301;

struct AiState {
    bool initialized = false;

    ecg_ai_config_t cfg{};

    /* TFLite */
    const tflite::Model *model = nullptr;
    tflite::MicroMutableOpResolver<16> resolver;
    bool resolver_ready = false;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;
    uint8_t *arena = nullptr;
    size_t arena_size = 0;

    /* 流式预处理 */
    uint32_t decim_ctr = 0;
    uint32_t sample_count = 0;   /* 抽取后样本数（1-based 计数） */
    uint32_t write_idx = 0;
    float buffer[kDefaultWindow] = {0};
    double hp_w1 = 0.0;
    double hp_w2 = 0.0;

    /* 策略/历史 */
    bool history[kMaxConfirmN] = {false};
    uint32_t history_count = 0;
    uint32_t history_idx = 0;
    uint32_t cooldown_remaining = 0;

    /* 回调/队列 */
    QueueHandle_t queue = nullptr;
    ecg_ai_result_cb_t callback = nullptr;

    uint32_t total_inferences = 0;
    uint32_t total_confirmed = 0;

    uint32_t next_sample_index = 0;
};

AiState g_ai;

inline double biquad(double x, double b0, double b1, double b2,
                     double a1, double a2, double *w1, double *w2) {
    double w = (double)x - a1 * (*w1) - a2 * (*w2);
    double y = b0 * w + b1 * (*w1) + b2 * (*w2);
    *w2 = *w1;
    *w1 = w;
    return y;
}

void zscore(float *buf, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += buf[i];
    float mean = sum / (float)n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = buf[i] - mean;
        var += d * d;
    }
    float std = sqrtf(var / (float)n);
    if (std < 1e-6f) std = 1.0f;
    for (int i = 0; i < n; i++) {
        buf[i] = (buf[i] - mean) / std;
    }
}

bool fill_int8_and_invoke(const float *window) {
    if (!g_ai.initialized || !g_ai.input || !g_ai.output) return false;
    const float scale = g_ai.input->params.scale;
    const int32_t zp = g_ai.input->params.zero_point;
    for (int i = 0; i < kDefaultWindow; i++) {
        int32_t q = (int32_t)(window[i] / scale + 0.5f) + zp;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        g_ai.input->data.int8[i] = (int8_t)q;
    }
    uint32_t t0 = (uint32_t)esp_timer_get_time();
    TfLiteStatus st = g_ai.interpreter->Invoke();
    uint32_t dt = (uint32_t)(esp_timer_get_time() - t0);
    if (st != kTfLiteOk) return false;
    g_ai.next_sample_index = g_ai.sample_count;
    float out_scale = g_ai.output->params.scale;
    int32_t out_zp = g_ai.output->params.zero_point;
    float p0 = (float)((g_ai.output->data.int8[0] - out_zp) * out_scale);
    float p1 = (float)((g_ai.output->data.int8[1] - out_zp) * out_scale);
    (void)p0;

    ecg_ai_result_t r{};
    r.confidence = p1;
    r.raw_abnormal = (p1 > g_ai.cfg.threshold) ? 1 : 0;
    r.sample_index = g_ai.next_sample_index;
    r.latency_us = dt;

    /* 确认策略 */
    g_ai.history[g_ai.history_idx] = (r.raw_abnormal != 0);
    g_ai.history_idx = (g_ai.history_idx + 1) % (uint32_t)g_ai.cfg.confirm_n;
    if (g_ai.history_count < (uint32_t)g_ai.cfg.confirm_n) g_ai.history_count++;

    bool candidate = false;
    if (g_ai.cfg.confirm_mode == ECG_AI_CONFIRM_ONE_OF_N) {
        for (uint32_t i = 0; i < g_ai.history_count; i++) {
            if (g_ai.history[i]) { candidate = true; break; }
        }
    } else {
        uint32_t cnt = 0;
        for (uint32_t i = 0; i < g_ai.history_count; i++) {
            if (g_ai.history[i]) cnt++;
        }
        candidate = (cnt >= (uint32_t)g_ai.cfg.confirm_k);
    }

    if (g_ai.cooldown_remaining > 0) {
        g_ai.cooldown_remaining--;
        r.confirmed = 0;
    } else if (candidate) {
        r.confirmed = 1;
        g_ai.cooldown_remaining = (uint32_t)g_ai.cfg.cooldown_beats;
    } else {
        r.confirmed = 0;
    }

    g_ai.total_inferences++;
    if (r.confirmed) g_ai.total_confirmed++;

    if (g_ai.callback) g_ai.callback(&r);
    if (g_ai.queue) {
        xQueueSend(g_ai.queue, &r, 0);
    }
    return true;
}

bool build_current_window(float *out) {
    if (g_ai.sample_count < kDefaultWindow) return false;
    for (int i = 0; i < kDefaultWindow; i++) {
        out[i] = g_ai.buffer[(g_ai.write_idx + i) % kDefaultWindow];
    }
    return true;
}

} /* namespace */

void ecg_ai_config_default(ecg_ai_config_t *cfg) {
    if (!cfg) return;
    cfg->threshold = 0.60f;
    cfg->confirm_mode = ECG_AI_CONFIRM_ONE_OF_N;
    cfg->confirm_n = 5;
    cfg->confirm_k = 5;
    cfg->cooldown_beats = 5;
    cfg->stride = 250;
    cfg->trigger_offset = 6;
    cfg->arena_size = 128 * 1024;
    cfg->use_psram = true;
    cfg->queue_length = 8;
}

bool ecg_ai_init(const uint8_t *model_data, size_t model_size,
                 const ecg_ai_config_t *cfg) {
    if (!model_data || !cfg) return false;
    if (g_ai.initialized) ecg_ai_reset();

    g_ai.cfg = *cfg;
    if (g_ai.cfg.confirm_n < 1) g_ai.cfg.confirm_n = 1;
    if (g_ai.cfg.confirm_n > kMaxConfirmN) g_ai.cfg.confirm_n = kMaxConfirmN;
    if (g_ai.cfg.confirm_k < 1) g_ai.cfg.confirm_k = 1;
    if (g_ai.cfg.confirm_k > g_ai.cfg.confirm_n) g_ai.cfg.confirm_k = g_ai.cfg.confirm_n;
    if (g_ai.cfg.stride <= 0) g_ai.cfg.stride = 250;
    if (g_ai.cfg.trigger_offset < 0) g_ai.cfg.trigger_offset = 0;
    if (g_ai.cfg.arena_size < 64 * 1024) g_ai.cfg.arena_size = 64 * 1024;

    if (g_ai.cfg.use_psram) {
        g_ai.arena = (uint8_t *)heap_caps_malloc_prefer(
            g_ai.cfg.arena_size, 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT,
            MALLOC_CAP_8BIT);
    } else {
        g_ai.arena = (uint8_t *)heap_caps_malloc(g_ai.cfg.arena_size, MALLOC_CAP_8BIT);
    }
    if (!g_ai.arena) {
        printf("[ecg_ai] arena alloc failed (%d bytes)\n", g_ai.cfg.arena_size);
        return false;
    }
    g_ai.arena_size = g_ai.cfg.arena_size;

    tflite::InitializeTarget();
    g_ai.model = tflite::GetModel(model_data);
    if (g_ai.model->version() != TFLITE_SCHEMA_VERSION) {
        printf("[ecg_ai] model schema mismatch\n");
        return false;
    }

    if (!g_ai.resolver_ready) {
        g_ai.resolver.AddExpandDims();
        g_ai.resolver.AddConv2D();
        g_ai.resolver.AddReshape();
        g_ai.resolver.AddDepthwiseConv2D();
        g_ai.resolver.AddMean();
        g_ai.resolver.AddFullyConnected();
        g_ai.resolver.AddLogistic();
        g_ai.resolver.AddShape();
        g_ai.resolver.AddStridedSlice();
        g_ai.resolver.AddPack();
        g_ai.resolver.AddMul();
        g_ai.resolver.AddAdd();
        g_ai.resolver.AddSoftmax();
        g_ai.resolver_ready = true;
    }

    g_ai.interpreter = new (std::nothrow) tflite::MicroInterpreter(
        g_ai.model, g_ai.resolver, g_ai.arena, g_ai.arena_size);
    if (!g_ai.interpreter) {
        printf("[ecg_ai] interpreter alloc failed\n");
        return false;
    }
    if (g_ai.interpreter->AllocateTensors() != kTfLiteOk) {
        printf("[ecg_ai] AllocateTensors failed\n");
        return false;
    }
    g_ai.input = g_ai.interpreter->input(0);
    g_ai.output = g_ai.interpreter->output(0);

    if (g_ai.cfg.queue_length > 0) {
        g_ai.queue = xQueueCreate(g_ai.cfg.queue_length, sizeof(ecg_ai_result_t));
        if (!g_ai.queue) {
            printf("[ecg_ai] queue create failed\n");
            return false;
        }
    }

    g_ai.initialized = true;
    printf("[ecg_ai] init OK, arena used=%zu, in_scale=%.8f zp=%d\n",
           g_ai.interpreter->arena_used_bytes(),
           g_ai.input->params.scale, (int)g_ai.input->params.zero_point);
    return true;
}

void ecg_ai_reset(void) {
    if (g_ai.interpreter) {
        delete g_ai.interpreter;
        g_ai.interpreter = nullptr;
    }
    if (g_ai.arena) {
        heap_caps_free(g_ai.arena);
        g_ai.arena = nullptr;
    }
    if (g_ai.queue) {
        vQueueDelete(g_ai.queue);
        g_ai.queue = nullptr;
    }
    /* 注意：不能 memset 整个 AiState，因为 MicroMutableOpResolver 是带构造的
     * C++ 对象；这里只清理运行期字段，resolver 保留构造状态。 */
    g_ai.initialized = false;
    g_ai.model = nullptr;
    g_ai.input = nullptr;
    g_ai.output = nullptr;
    g_ai.arena = nullptr;
    g_ai.arena_size = 0;
    g_ai.queue = nullptr;
    g_ai.callback = nullptr;
    g_ai.decim_ctr = 0;
    g_ai.sample_count = 0;
    g_ai.write_idx = 0;
    g_ai.hp_w1 = 0.0;
    g_ai.hp_w2 = 0.0;
    g_ai.history_count = 0;
    g_ai.history_idx = 0;
    g_ai.cooldown_remaining = 0;
    g_ai.total_inferences = 0;
    g_ai.total_confirmed = 0;
    g_ai.next_sample_index = 0;
    for (int i = 0; i < kDefaultWindow; i++) g_ai.buffer[i] = 0.0f;
    for (int i = 0; i < kMaxConfirmN; i++) g_ai.history[i] = false;
}

bool ecg_ai_is_initialized(void) {
    return g_ai.initialized;
}

void ecg_ai_feed_sample(float sample_500hz) {
    if (!g_ai.initialized) return;
    /* 2:1 抽取：保留偶数样本 */
    if ((g_ai.decim_ctr++ & 1) != 0) return;
    /* 抽取后 250Hz 上的因果 0.5Hz 高通 */
    float v = (float)biquad(sample_500hz, kAiHpB0, kAiHpB1, kAiHpB2,
                            kAiHpA1, kAiHpA2, &g_ai.hp_w1, &g_ai.hp_w2);
    g_ai.buffer[g_ai.write_idx] = v;
    g_ai.write_idx = (g_ai.write_idx + 1) % kDefaultWindow;
    g_ai.sample_count++;
    if ((g_ai.sample_count % (uint32_t)g_ai.cfg.stride) == (uint32_t)g_ai.cfg.trigger_offset &&
        g_ai.sample_count >= kDefaultWindow) {
        float win[kDefaultWindow];
        if (build_current_window(win)) {
            zscore(win, kDefaultWindow);
            fill_int8_and_invoke(win);
        }
    }
}

bool ecg_ai_run_preprocessed(const float *window_250) {
    if (!window_250) return false;
    return fill_int8_and_invoke(window_250);
}

bool ecg_ai_run_window_raw(const float *window_250) {
    if (!window_250) return false;
    float buf[kDefaultWindow];
    memcpy(buf, window_250, sizeof(buf));
    zscore(buf, kDefaultWindow);
    return fill_int8_and_invoke(buf);
}

bool ecg_ai_run_int8(const int8_t *input_250) {
    if (!g_ai.initialized || !input_250 || !g_ai.input) return false;
    memcpy(g_ai.input->data.int8, input_250, kDefaultWindow);
    uint32_t t0 = (uint32_t)esp_timer_get_time();
    TfLiteStatus st = g_ai.interpreter->Invoke();
    uint32_t dt = (uint32_t)(esp_timer_get_time() - t0);
    if (st != kTfLiteOk) return false;

    float out_scale = g_ai.output->params.scale;
    int32_t out_zp = g_ai.output->params.zero_point;
    ecg_ai_result_t r{};
    r.confidence = (float)((g_ai.output->data.int8[1] - out_zp) * out_scale);
    r.raw_abnormal = (r.confidence > g_ai.cfg.threshold) ? 1 : 0;
    r.sample_index = g_ai.next_sample_index;
    r.latency_us = dt;
    /* 简化：直接反馈原始拍，不修改确认状态（测试用） */
    r.confirmed = r.raw_abnormal;
    g_ai.total_inferences++;
    if (r.confirmed) g_ai.total_confirmed++;
    if (g_ai.callback) g_ai.callback(&r);
    if (g_ai.queue) xQueueSend(g_ai.queue, &r, 0);
    return true;
}

void ecg_ai_set_result_callback(ecg_ai_result_cb_t cb) {
    g_ai.callback = cb;
}

bool ecg_ai_pop_result(ecg_ai_result_t *out) {
    if (!g_ai.queue || !out) return false;
    return xQueueReceive(g_ai.queue, out, 0) == pdTRUE;
}

uint32_t ecg_ai_total_inferences(void) {
    return g_ai.total_inferences;
}

uint32_t ecg_ai_total_confirmed(void) {
    return g_ai.total_confirmed;
}
