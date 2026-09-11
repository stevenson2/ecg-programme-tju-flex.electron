/*
 * ecg_ai_dual.cpp - v2 双模型 AND 门控（M4 门控固件集成规范落地, 2026-09-07）
 *
 * beat 模型（int8, 250 点单拍, 逐拍 z-score 输入）+ record 模型（fp16 权重,
 * 1250 点 = 5×250 逐拍 z-score 窗按时间序连接）：
 *   p_b[i] = beat 模型对拍 i 的异常概率
 *   p_r[i] = record 模型对"以 i 结尾的 record_window_beats 拍窗"的异常概率
 *   hbw[i] = mean(p_b 最近 hbw_n 拍)
 *   pos[i] = (p_r[i] >= threshold_r) AND (hbw[i] >= threshold_b)
 *   cand[i] = (pos 最近 confirm_n 拍计数) >= confirm_k
 *   alarm  = cand 为真且距上次报警 >= cooldown_beats
 *
 * 与 ai_inference.cpp 生产单模型路径完全独立（独立解释器/arena/状态）。
 * 输入约定：与统一基准评测的部署链形态逐位同构（逐拍 z-score 单位尺度）。
 */
#include "ai_inference/ecg_ai_dual.h"

#include <string.h>
#include <math.h>
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

namespace {

constexpr int kDualRing = 64;          /* p_b / pos 历史容量 */
constexpr int kDualMaxBeats = 8;       /* record 窗最大拍数 */
constexpr int kDualWinPts = 250;

struct DualState {
    bool initialized = false;

    ecg_ai_dual_config_t cfg{};

    const tflite::Model *b_model = nullptr;
    tflite::AllOpsResolver b_resolver;
    tflite::MicroInterpreter *b_interp = nullptr;
    TfLiteTensor *b_in = nullptr;
    TfLiteTensor *b_out = nullptr;
    uint8_t *b_arena = nullptr;
    size_t b_arena_size = 0;

    const tflite::Model *r_model = nullptr;
    tflite::AllOpsResolver r_resolver;
    tflite::MicroInterpreter *r_interp = nullptr;
    TfLiteTensor *r_in = nullptr;
    TfLiteTensor *r_out = nullptr;
    uint8_t *r_arena = nullptr;
    size_t r_arena_size = 0;

    /* 拍窗环（逐拍 z-score 后的窗, 时间序） */
    float ring[kDualMaxBeats][kDualWinPts] = {{0}};
    uint32_t ring_idx = 0;
    uint32_t ring_count = 0;

    /* p_b 历史（hbw） */
    float pb_hist[kDualRing] = {0};
    uint32_t pb_idx = 0;
    uint32_t pb_count = 0;

    /* pos 历史 + 冷却 */
    bool pos_hist[kDualRing] = {false};
    uint32_t pos_idx = 0;
    uint32_t pos_count = 0;
    uint32_t cooldown = 0;

    uint32_t feed_count = 0;
    uint32_t total_confirmed = 0;
};

DualState g_dual;

static tflite::MicroErrorReporter g_dual_reporter;

void dual_fill_beat_input(const float *window) {
    const float scale = g_dual.b_in->params.scale;
    const int32_t zp = g_dual.b_in->params.zero_point;
    for (int i = 0; i < kDualWinPts; i++) {
        int32_t q = (int32_t)(window[i] / scale + 0.5f) + zp;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        g_dual.b_in->data.int8[i] = (int8_t)q;
    }
}

void dual_fill_record_input(const float *concat) {
    const int n = g_dual.cfg.record_window_beats * kDualWinPts;
    /* fp16 转换模型 I/O 保持 float32; 非 f32 输入在 init 时拒绝 */
    memcpy(g_dual.r_in->data.f, concat, sizeof(float) * n);
}

float dual_parse_beat_prob() {
    const float os = g_dual.b_out->params.scale;
    const int32_t oz = g_dual.b_out->params.zero_point;
    return (float)((g_dual.b_out->data.int8[1] - oz) * os);
}

float dual_parse_record_prob() {
    if (g_dual.r_out->type == kTfLiteInt8) {
        const float os = g_dual.r_out->params.scale;
        const int32_t oz = g_dual.r_out->params.zero_point;
        return (float)((g_dual.r_out->data.int8[1] - oz) * os);
    }
    return g_dual.r_out->data.f[1];
}

} /* namespace */

void ecg_ai_dual_config_default(ecg_ai_dual_config_t *cfg) {
    if (!cfg) return;
    cfg->threshold_r = 0.50f;      /* v1-min（最保守） */
    cfg->threshold_b = 0.15f;
    cfg->confirm_mode = 1;         /* 1-of-n */
    cfg->confirm_n = 5;
    cfg->confirm_k = 1;
    cfg->cooldown_beats = 20;
    cfg->hbw_n = 5;
    cfg->record_window_beats = 5;
    cfg->arena_size = 1024 * 1024;
    cfg->use_psram = true;
}

bool ecg_ai_dual_init(const uint8_t *beat_data, size_t beat_len,
                      const uint8_t *record_data, size_t record_len,
                      const ecg_ai_dual_config_t *cfg) {
    (void)beat_len;
    (void)record_len;
    if (!beat_data || !record_data || !cfg) return false;
    ecg_ai_dual_reset();

    g_dual.cfg = *cfg;
    if (g_dual.cfg.confirm_n < 1) g_dual.cfg.confirm_n = 1;
    if (g_dual.cfg.confirm_n > kDualRing) g_dual.cfg.confirm_n = kDualRing;
    if (g_dual.cfg.confirm_k < 1) g_dual.cfg.confirm_k = 1;
    if (g_dual.cfg.confirm_k > g_dual.cfg.confirm_n)
        g_dual.cfg.confirm_k = g_dual.cfg.confirm_n;
    if (g_dual.cfg.cooldown_beats < 0) g_dual.cfg.cooldown_beats = 0;
    if (g_dual.cfg.hbw_n < 1) g_dual.cfg.hbw_n = 1;
    if (g_dual.cfg.hbw_n > kDualRing) g_dual.cfg.hbw_n = kDualRing;
    if (g_dual.cfg.record_window_beats < 1)
        g_dual.cfg.record_window_beats = 1;
    if (g_dual.cfg.record_window_beats > kDualMaxBeats)
        g_dual.cfg.record_window_beats = kDualMaxBeats;
    if (g_dual.cfg.arena_size < 128 * 1024)
        g_dual.cfg.arena_size = 128 * 1024;

    tflite::InitializeTarget();

    /* ---- beat (int8) ---- */
    g_dual.b_model = tflite::GetModel(beat_data);
    if (g_dual.b_model->version() != TFLITE_SCHEMA_VERSION) {
        return false;
    }
    g_dual.b_arena_size = 128 * 1024;
    g_dual.b_arena = (uint8_t *)heap_caps_malloc_prefer(
        g_dual.b_arena_size, 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT,
        MALLOC_CAP_8BIT);
    if (!g_dual.b_arena) {
        g_dual.b_arena = (uint8_t *)malloc(g_dual.b_arena_size);
    }
    if (!g_dual.b_arena) return false;
    g_dual.b_interp = new (std::nothrow) tflite::MicroInterpreter(
        g_dual.b_model, g_dual.b_resolver, g_dual.b_arena,
        g_dual.b_arena_size, &g_dual_reporter);
    if (!g_dual.b_interp || g_dual.b_interp->AllocateTensors() != kTfLiteOk) {
        return false;
    }
    g_dual.b_in = g_dual.b_interp->input(0);
    g_dual.b_out = g_dual.b_interp->output(0);

    /* ---- record (fp16 权重, f32 I/O) ---- */
    g_dual.r_model = tflite::GetModel(record_data);
    if (g_dual.r_model->version() != TFLITE_SCHEMA_VERSION) {
        return false;
    }
    g_dual.r_arena = (uint8_t *)heap_caps_malloc_prefer(
        g_dual.cfg.arena_size, 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT,
        MALLOC_CAP_8BIT);
    if (!g_dual.r_arena) {
        g_dual.r_arena = (uint8_t *)malloc(g_dual.cfg.arena_size);
    }
    if (!g_dual.r_arena) return false;
    g_dual.r_arena_size = g_dual.cfg.arena_size;
    g_dual.r_interp = new (std::nothrow) tflite::MicroInterpreter(
        g_dual.r_model, g_dual.r_resolver, g_dual.r_arena,
        g_dual.r_arena_size, &g_dual_reporter);
    if (!g_dual.r_interp || g_dual.r_interp->AllocateTensors() != kTfLiteOk) {
        return false;
    }
    g_dual.r_in = g_dual.r_interp->input(0);
    g_dual.r_out = g_dual.r_interp->output(0);
    if (g_dual.r_in->type != kTfLiteFloat32) {
        return false;
    }

    g_dual.initialized = true;
    return true;
}

void ecg_ai_dual_reset(void) {
    if (g_dual.b_interp) { delete g_dual.b_interp; g_dual.b_interp = nullptr; }
    if (g_dual.r_interp) { delete g_dual.r_interp; g_dual.r_interp = nullptr; }
    if (g_dual.b_arena) { heap_caps_free(g_dual.b_arena); g_dual.b_arena = nullptr; }
    if (g_dual.r_arena) { heap_caps_free(g_dual.r_arena); g_dual.r_arena = nullptr; }
    g_dual.initialized = false;
    g_dual.b_model = nullptr;
    g_dual.r_model = nullptr;
    g_dual.b_in = nullptr;
    g_dual.b_out = nullptr;
    g_dual.r_in = nullptr;
    g_dual.r_out = nullptr;
    g_dual.b_arena = nullptr;
    g_dual.r_arena = nullptr;
    g_dual.ring_idx = 0;
    g_dual.ring_count = 0;
    g_dual.pb_idx = 0;
    g_dual.pb_count = 0;
    g_dual.pos_idx = 0;
    g_dual.pos_count = 0;
    g_dual.cooldown = 0;
    g_dual.feed_count = 0;
    g_dual.total_confirmed = 0;
    memset(g_dual.ring, 0, sizeof(g_dual.ring));
    memset(g_dual.pb_hist, 0, sizeof(g_dual.pb_hist));
    memset(g_dual.pos_hist, 0, sizeof(g_dual.pos_hist));
}

bool ecg_ai_dual_is_initialized(void) {
    return g_dual.initialized;
}

void ecg_ai_dual_gate_reset(void) {
    g_dual.ring_idx = 0;
    g_dual.ring_count = 0;
    g_dual.pb_idx = 0;
    g_dual.pb_count = 0;
    g_dual.pos_idx = 0;
    g_dual.pos_count = 0;
    g_dual.cooldown = 0;
    memset(g_dual.ring, 0, sizeof(g_dual.ring));
    memset(g_dual.pb_hist, 0, sizeof(g_dual.pb_hist));
    memset(g_dual.pos_hist, 0, sizeof(g_dual.pos_hist));
}

ecg_ai_dual_result_t ecg_ai_dual_feed_beat(const float *window_250) {
    ecg_ai_dual_result_t r{};
    if (!g_dual.initialized || !window_250) return r;
    uint32_t t0 = (uint32_t)esp_timer_get_time();

    /* 1. beat 推理 */
    dual_fill_beat_input(window_250);
    g_dual.b_interp->Invoke();
    const float p_b = dual_parse_beat_prob();

    /* 2. 拍窗环更新 */
    memcpy(g_dual.ring[g_dual.ring_idx], window_250,
           sizeof(float) * kDualWinPts);
    g_dual.ring_idx = (g_dual.ring_idx + 1) %
                      (uint32_t)g_dual.cfg.record_window_beats;
    if (g_dual.ring_count < (uint32_t)g_dual.cfg.record_window_beats)
        g_dual.ring_count++;

    /* 3. p_b 历史 + hbw */
    g_dual.pb_hist[g_dual.pb_idx] = p_b;
    g_dual.pb_idx = (g_dual.pb_idx + 1) % kDualRing;
    if (g_dual.pb_count < (uint32_t)g_dual.cfg.hbw_n)
        g_dual.pb_count++;
    float hbw = 0.0f;
    for (uint32_t i = 0; i < g_dual.pb_count; i++) {
        hbw += g_dual.pb_hist[(g_dual.pb_idx + kDualRing - 1 - i) % kDualRing];
    }
    hbw /= (float)g_dual.pb_count;

    /* 4. record 推理（凑满窗才跑） */
    float p_r = 0.0f;
    bool have_r = false;
    if (g_dual.ring_count >= (uint32_t)g_dual.cfg.record_window_beats) {
        static float concat[kDualMaxBeats * kDualWinPts];
        for (uint32_t i = 0; i < (uint32_t)g_dual.cfg.record_window_beats; i++) {
            uint32_t src = (g_dual.ring_idx + i) %
                           (uint32_t)g_dual.cfg.record_window_beats;
            memcpy(concat + i * kDualWinPts, g_dual.ring[src],
                   sizeof(float) * kDualWinPts);
        }
        dual_fill_record_input(concat);
        g_dual.r_interp->Invoke();
        p_r = dual_parse_record_prob();
        have_r = true;
    }

    /* 5. AND 门控 + k-of-n + 冷却 */
    r.p_b = p_b;
    r.p_r = have_r ? p_r : 0.0f;
    r.hbw = hbw;
    r.have_r = have_r ? 1 : 0;
    r.gate_pos = (have_r && p_r >= g_dual.cfg.threshold_r &&
                  hbw >= g_dual.cfg.threshold_b) ? 1 : 0;
    r.sample_index = ++g_dual.feed_count;
    r.latency_us = (uint32_t)(esp_timer_get_time() - t0);

    g_dual.pos_hist[g_dual.pos_idx] = (r.gate_pos != 0);
    g_dual.pos_idx = (g_dual.pos_idx + 1) % (uint32_t)g_dual.cfg.confirm_n;
    if (g_dual.pos_count < (uint32_t)g_dual.cfg.confirm_n)
        g_dual.pos_count++;

    bool candidate = false;
    if (g_dual.cfg.confirm_mode == 1) {   /* 1-of-n */
        for (uint32_t i = 0; i < g_dual.pos_count; i++) {
            if (g_dual.pos_hist[i]) { candidate = true; break; }
        }
    } else {                              /* k-of-n */
        uint32_t cnt = 0;
        for (uint32_t i = 0; i < g_dual.pos_count; i++) {
            if (g_dual.pos_hist[i]) cnt++;
        }
        candidate = (cnt >= (uint32_t)g_dual.cfg.confirm_k);
    }

    if (g_dual.cooldown > 0) {
        g_dual.cooldown--;
        r.confirmed = 0;
    } else if (candidate) {
        r.confirmed = 1;
        g_dual.cooldown = (uint32_t)g_dual.cfg.cooldown_beats;
    }
    if (r.confirmed) g_dual.total_confirmed++;

    return r;
}

bool ecg_ai_dual_bench_beat(const float *window_250, float *p_b) {
    if (!g_dual.initialized || !window_250 || !p_b) return false;
    dual_fill_beat_input(window_250);
    g_dual.b_interp->Invoke();
    *p_b = dual_parse_beat_prob();
    return true;
}

bool ecg_ai_dual_bench_record(const float *window_1250, float *p_r) {
    if (!g_dual.initialized || !window_1250 || !p_r) return false;
    dual_fill_record_input(window_1250);
    g_dual.r_interp->Invoke();
    *p_r = dual_parse_record_prob();
    return true;
}

uint32_t ecg_ai_dual_total_inferences(void) {
    return g_dual.feed_count;
}

uint32_t ecg_ai_dual_total_confirmed(void) {
    return g_dual.total_confirmed;
}
