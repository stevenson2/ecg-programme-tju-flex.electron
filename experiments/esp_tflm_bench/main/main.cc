#include <stdio.h>
#include <math.h>
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include <string.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "model.h"
#include "test_vectors.h"
#include "ecg_ai.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

static constexpr int kTensorArenaSize = 128 * 1024;
static uint8_t* tensor_arena = nullptr;

extern "C" void app_main(void) {
  tflite::InitializeTarget();
  if (tensor_arena == nullptr) {
    tensor_arena = (uint8_t*)heap_caps_malloc_prefer(kTensorArenaSize, 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT, MALLOC_CAP_8BIT);
  }
  if (tensor_arena == nullptr) {
    printf("Arena alloc failed\n");
    return;
  }
  const tflite::Model* model = tflite::GetModel(models_ecg_model_exp7c_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("Model schema mismatch\n");
    return;
  }
  static tflite::MicroMutableOpResolver<16> resolver;
  resolver.AddExpandDims(); resolver.AddConv2D(); resolver.AddReshape(); resolver.AddDepthwiseConv2D(); resolver.AddMean(); resolver.AddFullyConnected(); resolver.AddLogistic(); resolver.AddShape(); resolver.AddStridedSlice(); resolver.AddPack(); resolver.AddMul(); resolver.AddAdd(); resolver.AddSoftmax();
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("AllocateTensors failed\n");
    return;
  }
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  printf("arena used: %zu\n", interpreter.arena_used_bytes());
  printf("input dims: %d %d %d %d\n", input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->size);
  printf("input type: %d quant: %f zp:%d\n", (int)input->type, input->params.scale, (int)input->params.zero_point);
  printf("output type: %d\n", output->type);
    // ---- PC consistency: run the fixed 200-beat vector set ----
    printf("CONSISTENCY_BEGIN\n");
    for (int w = 0; w < 3; w++) interpreter.Invoke();  // warmup
    float out_scale = output->params.scale;
    int32_t out_zp = output->params.zero_point;
    printf("out_quant scale=%.9f zp=%ld\n", out_scale, (long)out_zp);
    for (int i = 0; i < kTestCount; i++) {
      memcpy(input->data.int8, kTestInputs[i], 250);
      TfLiteStatus st = interpreter.Invoke();
      if (st != kTfLiteOk) {
        printf("CONSISTENCY_FAIL,%d\n", i);
        continue;
      }
      int8_t q0 = output->data.int8[0];
      int8_t q1 = output->data.int8[1];
      float p0 = ((float)q0 - (float)out_zp) * out_scale;
      float p1 = ((float)q1 - (float)out_zp) * out_scale;
      float mx = p0 > p1 ? p0 : p1;
      float e0 = expf(p0 - mx);
      float e1 = expf(p1 - mx);
      float p1_double = e1 / (e0 + e1);
      printf("RESULT,%d,%d,%d,%.6f,%.6f\n", i, (int)q0, (int)q1, p1, p1_double);
          vTaskDelay(pdMS_TO_TICKS(1));
    }
    printf("CONSISTENCY_END\n");
    // ---- ecg_ai component smoke test ----
    {
      ecg_ai_config_t cfg;
      ecg_ai_config_default(&cfg);
      cfg.threshold = 0.60f;
      cfg.confirm_mode = ECG_AI_CONFIRM_ONE_OF_N;
      cfg.confirm_n = 5;
      cfg.cooldown_beats = 5;
      if (ecg_ai_init(models_ecg_model_exp7c_int8_tflite,
                      models_ecg_model_exp7c_int8_tflite_len, &cfg)) {
        printf("ECG_AI_COMPONENT_BEGIN\n");
        int comp_n = 10 < kTestCount ? 10 : kTestCount;
        for (int i = 0; i < comp_n; i++) {
          ecg_ai_result_t r;
          if (ecg_ai_run_int8(kTestInputs[i]) && ecg_ai_pop_result(&r)) {
            printf("ECG_AI_COMPONENT,%d,%.6f,%d\n", i, r.confidence, (int)r.confirmed);
          }
          vTaskDelay(pdMS_TO_TICKS(1));
        }
        printf("ECG_AI_COMPONENT_END,inf=%lu,confirmed=%lu\n",
               (unsigned long)ecg_ai_total_inferences(),
               (unsigned long)ecg_ai_total_confirmed());
        ecg_ai_reset();
      } else {
        printf("ECG_AI_COMPONENT_INIT_FAIL\n");
      }
    }
  // dummy input
  for (int i = 0; i < 250; i++) {
    float x = (float)((i * 37) % 100) / 100.0f - 0.5f;
    int32_t q = (int32_t)(x / input->params.scale + 0.5f) + input->params.zero_point;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    input->data.int8[i] = (int8_t)q;
  }
  // warmup
  for (int i = 0; i < 5; i++) interpreter.Invoke();
  const int N = 20;
  uint32_t total = 0, min = UINT32_MAX, max = 0;
  for (int i = 0; i < N; i++) {
    uint32_t t0 = esp_timer_get_time();
    interpreter.Invoke();
    uint32_t dt = (uint32_t)(esp_timer_get_time() - t0);
    total += dt;
    if (dt < min) min = dt;
    if (dt > max) max = dt;
  }
  printf("BENCH,%d,%lu,%lu,%lu\n", N, (unsigned long)(total / N), (unsigned long)min, (unsigned long)max);
  vTaskDelay(pdMS_TO_TICKS(1000));
}



