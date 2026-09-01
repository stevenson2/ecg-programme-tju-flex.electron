#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "driver/gpio.h"
#include "model.h"
#include "ecg_ai.h"
#include "storage/ecg_recorder.h"
#include "bluetooth/ble.h"
#include "wifi/ecg_wifi.h"
#include "adc_afe/afe_hal.h"
#include "filter/filter.h"
#include "signal_generator/ecg_simulator.h"
#include "signal_generator/ecg_replay.h"
#include "heartrate/heartrate.h"
#include "rhythm_safety/rhythm_safety.h"
#include "af_detect/af_detect.h"
#include "vf_detect/vf_detect.h"

#define DC_OFFSET_REMOVE 1.65f
#define COMB_TAPS 10
#define BUTTON_DEBOUNCE_MS 250

typedef enum {
    SOURCE_SIMULATOR = 0,
    SOURCE_REPLAY_NORMAL,
    SOURCE_REPLAY_ABNORMAL,
    SOURCE_AFE_REAL
} InputSource;

static int s_mode = SOURCE_SIMULATOR;
static uint32_t s_lastButtonMs = 0;
static bool s_buttonWasLow = false;
static bool s_secondAbnormal = false;   /* 本秒内是否出现 AI 确认异常 (录制位图用) */

static float s_combBuf1[COMB_TAPS] = {0};
static int s_combIdx1 = 0;
static float s_combSum1 = 0.0f;
static float s_combBuf2[COMB_TAPS] = {0};
static int s_combIdx2 = 0;
static float s_combSum2 = 0.0f;

static float applyCombFilter(float x) {
    s_combSum1 -= s_combBuf1[s_combIdx1];
    s_combBuf1[s_combIdx1] = x;
    s_combSum1 += x;
    s_combIdx1 = (s_combIdx1 + 1) % COMB_TAPS;
    float y1 = s_combSum1 / (float)COMB_TAPS;

    s_combSum2 -= s_combBuf2[s_combIdx2];
    s_combBuf2[s_combIdx2] = y1;
    s_combSum2 += y1;
    s_combIdx2 = (s_combIdx2 + 1) % COMB_TAPS;
    return s_combSum2 / (float)COMB_TAPS;
}

static const char *modeName(int mode) {
    switch (mode) {
        case SOURCE_SIMULATOR: return "SIMULATOR";
        case SOURCE_REPLAY_NORMAL: return "REPLAY_NORMAL";
        case SOURCE_REPLAY_ABNORMAL: return "REPLAY_ABNORMAL";
        case SOURCE_AFE_REAL: return "AFE_REAL";
        default: return "UNKNOWN";
    }
}

static void switchMode(int newMode) {
    s_mode = newMode;
    printf("[main] mode -> %s\n", modeName(newMode));
    filterReset();
    aiFilterReset();
    hrFullReset();
    rsReset();
    afReset();
    vfReset();
    if (newMode == SOURCE_REPLAY_NORMAL) ecgReplaySetSegment(0);
    if (newMode == SOURCE_REPLAY_ABNORMAL) ecgReplaySetSegment(1);
    if (newMode == SOURCE_REPLAY_NORMAL || newMode == SOURCE_REPLAY_ABNORMAL) {
        ecgReplayReset();
    }
    if (newMode == SOURCE_AFE_REAL) {
        AFE_HAL_Config cfg = AFE_HAL_DEFAULT_3V3;
        cfg.oversample = 1;
        afeHalInit(&cfg);
    }
}

static void checkButton(void) {
    uint32_t now = (uint32_t)(esp_timer_get_time() / 1000);
    bool low = gpio_get_level(GPIO_NUM_0) == 0;
    if (low && !s_buttonWasLow && (now - s_lastButtonMs) >= BUTTON_DEBOUNCE_MS) {
        s_lastButtonMs = now;
        int next;
        switch (s_mode) {
            case SOURCE_SIMULATOR: next = SOURCE_AFE_REAL; break;
            case SOURCE_AFE_REAL: next = SOURCE_REPLAY_NORMAL; break;
            case SOURCE_REPLAY_NORMAL: next = SOURCE_REPLAY_ABNORMAL; break;
            default: next = SOURCE_SIMULATOR; break;
        }
        switchMode(next);
    }
    s_buttonWasLow = low;
}

static void storage_init_task(void *arg) {
    (void)arg;
    if (!ecgRecorderInit()) {
        printf("[storage] ecgRecorderInit failed\n");
        vTaskDelete(NULL);
        return;
    }
    /* 异常触发自动录制: 异常秒 -> 启动, 连续 N 秒正常 -> 停止。
     * IDF 线无 BLE/串口 REC_* 命令通道, 故默认启用 auto-record 保证录制功能可用。 */
    ecgRecorderSetAutoRecord(true);
    printf("[storage] recorder init OK, auto-record enabled\n");
    vTaskDelete(NULL);
}

extern "C" void app_main(void) {
    ecg_ai_config_t cfg;
    ecg_ai_config_default(&cfg);
    cfg.threshold = 0.50f;
    cfg.confirm_mode = ECG_AI_CONFIRM_ONE_OF_N;
    cfg.confirm_n = 5;
    cfg.cooldown_beats = 5;
    if (!ecg_ai_init(models_ecg_model_exp7c_int8_tflite,
                     models_ecg_model_exp7c_int8_tflite_len, &cfg)) {
        printf("[main] ecg_ai init failed\n");
        return;
    }

    ecgSimulatorInit();
    ecgReplayInit();
    hrInit();
    rsInit();
    afInit();
    vfInit();
    filterInit();
    aiFilterInit();

    gpio_config_t io = {
        .pin_bit_mask = 1ULL << GPIO_NUM_0,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io);

    /* 先把 BLE/WiFi 启起来，避免 SPIFFS 首次格式化阻塞导致手机扫不到设备。 */
    initBLE();
    ecgWifiInit();
    ecgWifiStart();

    xTaskCreate(storage_init_task, "storage_init", 4096, NULL, 2, NULL);

    uint32_t frame = 0;
    float last_conf = 0.0f;
    int last_abnormal = 0;
    printf("[main] ESP-IDF ECG demo start, mode=%s\n", modeName(s_mode));

    while (true) {
        checkButton();

        float noisySample, cleanSample;
        if (s_mode == SOURCE_SIMULATOR) {
            noisySample = generateECGSample();
            cleanSample = getCleanECGValue();
        } else if (s_mode == SOURCE_REPLAY_NORMAL || s_mode == SOURCE_REPLAY_ABNORMAL) {
            noisySample = ecgReplayNextSample();
            cleanSample = noisySample;
        } else {
            noisySample = afeHalReadSample();
            cleanSample = noisySample - 1.65f;
        }

        float noisyNoDC = noisySample - DC_OFFSET_REMOVE;
        float combOut = applyCombFilter(noisyNoDC);
        float displaySample = applyDisplayFilter(combOut);
        float filteredSample = applyFilter(combOut);

        float ai_in = applyFilterAI(combOut);
        ecg_ai_feed_sample(ai_in);

        HR_Result hr = hrProcess(filteredSample);
        RS_Result rs = rsProcess(&hr);
        AF_Result af = afProcess(&hr);
        (void)rs; (void)af;

        ecg_ai_result_t r;
        while (ecg_ai_pop_result(&r)) {
            last_conf = r.confidence;
            last_abnormal = (int)r.confirmed;
            if (r.confirmed) s_secondAbnormal = true;
            printf("AI_RESULT,%.4f,%u,%u\n", r.confidence, (unsigned)r.raw_abnormal, (unsigned)r.confirmed);
        }

        /* ECG 录制: 2:1 抽取 (500->250Hz) 喂 int16 样本, 每秒更新异常位图。
         * 与 Arduino 链同参数: 记录 cleanSample (去偏置原始), scale 8000.0 (V->int16)。 */
        if ((frame % 2) == 0) {
            ecgRecorderPushSample((int16_t)(cleanSample * 8000.0f));
        }
        {
            uint32_t nowSec = (uint32_t)(esp_timer_get_time() / 1000000ULL);
            static uint32_t s_lastRecSec = 0;
            if (nowSec != s_lastRecSec) {
                s_lastRecSec = nowSec;
                ecgRecorderSetSecondAbnormal(s_secondAbnormal);
                s_secondAbnormal = false;
            }
        }

        if ((frame % 4) == 0 && isBLEConnected()) {
            char ble_line[160];
            uint8_t trueBPM = (s_mode == SOURCE_SIMULATOR) ? ecgSimulatorGetTrueBPM() : 0;
            snprintf(ble_line, sizeof(ble_line),
                     "%.3f,%.3f,%.3f,%u,%u,%.2f,0,%d,%.3f;",
                     cleanSample, noisyNoDC, displaySample,
                     (unsigned)hr.bpm, (unsigned)trueBPM, hr.sqi,
                     last_abnormal, last_conf);
            sendBLEMessage(ble_line);
        }

        if (frame % 500 == 0) {
            printf("TICK,%lu,src=%s,bpm=%u,sqi=%.3f,disp=%.4f,comb=%.4f\n",
                   (unsigned long)frame, modeName(s_mode),
                   (unsigned)hr.bpm, hr.sqi, displaySample, combOut);
        }
        frame++;
        vTaskDelay(pdMS_TO_TICKS(2));
    }
}
