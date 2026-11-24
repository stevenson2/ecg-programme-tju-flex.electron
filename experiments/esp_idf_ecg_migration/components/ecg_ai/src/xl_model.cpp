#include "xl_model.h"

#include <stdio.h>
#include <string.h>
#include <new>
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

struct Slot {
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interp = nullptr;
    uint8_t *arena = nullptr;
    size_t arena_size = 0;
    TfLiteTensor *in_ecg = nullptr;
    TfLiteTensor *in_feat = nullptr;
    TfLiteTensor *out = nullptr;
};

Slot g_slot[2];
/* One global inference mutex (v0.2.1). The async inference task and the UART
 * XL_D1 diagnostic both call xl_model_infer(). Without a lock they could run
 * concurrently and corrupt each other (observed: LoadStoreError panic inside
 * reduce/Mean, and garbage D1 outputs). A per-model lock is not enough: the
 * ESP-NN conv/depthwise kernels keep a process-wide scratch-buffer pointer
 * (esp_nn_set_conv_scratch_buf), so no two Invoke() calls may overlap at all.
 * Statically allocated, created once, never deleted. */
StaticSemaphore_t g_lock_buf;
SemaphoreHandle_t g_lock = nullptr;
void ensure_locks() {
    if (!g_lock) g_lock = xSemaphoreCreateMutexStatic(&g_lock_buf);
}
/* One shared resolver: op list = ptq_info.json "ops" (13 builtins), no Softmax (output is int8 logits). */
tflite::MicroMutableOpResolver<13> *g_resolver = nullptr;

bool build_resolver() {
    if (g_resolver) return true;
    g_resolver = new (std::nothrow) tflite::MicroMutableOpResolver<13>();
    if (!g_resolver) return false;
    TfLiteStatus s = kTfLiteOk;
    s = (TfLiteStatus)(s | g_resolver->AddAdd());
    s = (TfLiteStatus)(s | g_resolver->AddConcatenation());
    s = (TfLiteStatus)(s | g_resolver->AddConv2D());
    s = (TfLiteStatus)(s | g_resolver->AddDepthwiseConv2D());
    s = (TfLiteStatus)(s | g_resolver->AddExpandDims());
    s = (TfLiteStatus)(s | g_resolver->AddFullyConnected());
    s = (TfLiteStatus)(s | g_resolver->AddLogistic());
    s = (TfLiteStatus)(s | g_resolver->AddMean());
    s = (TfLiteStatus)(s | g_resolver->AddMul());
    s = (TfLiteStatus)(s | g_resolver->AddPack());
    s = (TfLiteStatus)(s | g_resolver->AddReshape());
    s = (TfLiteStatus)(s | g_resolver->AddShape());
    s = (TfLiteStatus)(s | g_resolver->AddStridedSlice());
    return s == kTfLiteOk;
}

bool init_slot(int m, const unsigned char *data, bool use_psram) {
    Slot &sl = g_slot[m];
    sl.model = tflite::GetModel(data);
    if (sl.model->version() != TFLITE_SCHEMA_VERSION) {
        printf("[xl] model %d schema mismatch\n", m);
        return false;
    }
    /* Arena size is searched upward; the first size that AllocateTensors accepts is kept. */
    static const size_t kTry[] = {64 * 1024, 96 * 1024, 128 * 1024, 192 * 1024, 256 * 1024};
    for (size_t sz : kTry) {
        sl.arena = use_psram
            ? (uint8_t *)heap_caps_malloc(sz, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
            : (uint8_t *)heap_caps_malloc(sz, MALLOC_CAP_8BIT);
        if (!sl.arena && use_psram) sl.arena = (uint8_t *)heap_caps_malloc(sz, MALLOC_CAP_8BIT);
        if (!sl.arena) { printf("[xl] model %d arena %u alloc failed\n", m, (unsigned)sz); return false; }
        sl.arena_size = sz;
        sl.interp = new (std::nothrow) tflite::MicroInterpreter(sl.model, *g_resolver, sl.arena, sz);
        if (sl.interp && sl.interp->AllocateTensors() == kTfLiteOk) break;
        delete sl.interp; sl.interp = nullptr;
        heap_caps_free(sl.arena); sl.arena = nullptr;
    }
    if (!sl.interp) { printf("[xl] model %d AllocateTensors failed up to 256KB\n", m); return false; }
    if (sl.interp->inputs_size() != 2 || sl.interp->outputs_size() != 1) {
        printf("[xl] model %d unexpected io count %u/%u\n", m,
               (unsigned)sl.interp->inputs_size(), (unsigned)sl.interp->outputs_size());
        return false;
    }
    for (int i = 0; i < 2; i++) {
        TfLiteTensor *t = sl.interp->input(i);
        if (t->type != kTfLiteInt8) { printf("[xl] model %d input %d not int8\n", m, i); return false; }
        if (t->dims->size == 3) sl.in_ecg = t; else sl.in_feat = t;
    }
    sl.out = sl.interp->output(0);
    if (!sl.in_ecg || !sl.in_feat || sl.out->type != kTfLiteInt8) {
        printf("[xl] model %d tensor layout mismatch\n", m);
        return false;
    }
    printf("[xl] model %d ready: arena=%u used=%u in_ecg scale=%.9f zp=%d feat scale=%.9f zp=%d out scale=%.9f zp=%d\n",
           m, (unsigned)sl.arena_size, (unsigned)sl.interp->arena_used_bytes(),
           (double)sl.in_ecg->params.scale, (int)sl.in_ecg->params.zero_point,
           (double)sl.in_feat->params.scale, (int)sl.in_feat->params.zero_point,
           (double)sl.out->params.scale, (int)sl.out->params.zero_point);
    return true;
}

}  // namespace

bool xl_model_init(const unsigned char *const models[2], bool use_psram) {
    ensure_locks();
    xl_model_deinit();
    tflite::InitializeTarget();
    if (!build_resolver()) { printf("[xl] resolver failed\n"); return false; }
    for (int m = 0; m < 2; m++) {
        if (!init_slot(m, models[m], use_psram)) { xl_model_deinit(); return false; }
    }
    return true;
}

void xl_model_deinit(void) {
    ensure_locks();
    xSemaphoreTake(g_lock, portMAX_DELAY);
    for (int m = 0; m < 2; m++) {
        Slot &sl = g_slot[m];
        delete sl.interp;
        if (sl.arena) heap_caps_free(sl.arena);
        sl = Slot();
    }
    xSemaphoreGive(g_lock);
}

int xl_model_infer(int m, const int8_t *win, const int8_t *feat, int8_t out[4]) {
    if (m < 0 || m > 1 || !g_lock) return -1;
    xSemaphoreTake(g_lock, portMAX_DELAY);
    Slot &sl = g_slot[m];
    int rc = -1;
    if (sl.interp) {
        memcpy(sl.in_ecg->data.int8, win, 250);
        memcpy(sl.in_feat->data.int8, feat, 4);
        int64_t t0 = esp_timer_get_time();
        if (sl.interp->Invoke() == kTfLiteOk) {
            rc = (int)(esp_timer_get_time() - t0);
            memcpy(out, sl.out->data.int8, 4);
        }
    }
    xSemaphoreGive(g_lock);
    return rc;
}

size_t xl_model_arena_used(int m) { return (m >= 0 && m < 2 && g_slot[m].interp) ? g_slot[m].interp->arena_used_bytes() : 0; }
size_t xl_model_arena_size(int m) { return (m >= 0 && m < 2) ? g_slot[m].arena_size : 0; }
