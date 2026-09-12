#include <stdio.h>
#include <string.h>
#include <strings.h>
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
/* BLE 批量发送：攒 BLE_BATCH_SIZE 帧再 Notify 一次，降低射频占空比/发热。 */
#define BLE_BATCH_SIZE   2
#define BLE_BATCH_BUF_SIZE (160 * BLE_BATCH_SIZE + 1)
static char s_bleBatchBuf[BLE_BATCH_BUF_SIZE];
static int  s_bleBatchLen   = 0;
static int  s_bleBatchCount = 0;
static volatile bool s_storageReady = false;

/* 热路径日志开关: 1=打印 AI_RESULT/TICK (调试), 0=关闭 (release/功耗实测)。
 * 两行/秒 @460800 对功耗影响极小, 默认保持开启。 */
#define ECG_HOT_LOG 1

/* 功耗/性能观测: 每轮处理耗时累计, TICK 打 busy% 与 2ms 节拍超时次数。
 * 处理一旦超 2ms, vTaskDelay 节拍会被拉长, 采样率静默下降 — ovr 必须为 0。 */
static uint64_t s_busyAccumUs = 0;
static uint64_t s_lastTickUs = 0;
static uint32_t s_loopOverruns = 0;
static bool s_bleWasConnected = false;
/* REC_SCHEDULE <间隔秒> <时长秒>：上电秒数调度（无 RTC）。 */
static uint32_t s_schedInterval = 0;
static uint32_t s_schedDuration = 0;
static uint32_t s_schedNextStart = 0;
static bool s_schedActiveRec = false;


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

static bool strEqualsIgnoreCase(const char *a, const char *b) {
    return strcasecmp(a, b) == 0;
}

static bool strStartsWithIgnoreCase(const char *s, const char *prefix) {
    while (*prefix) {
        if (*s == '\0') return false;
        char cs = (*s >= 'a' && *s <= 'z') ? (char)(*s - 'a' + 'A') : *s;
        char cp = (*prefix >= 'a' && *prefix <= 'z') ? (char)(*prefix - 'a' + 'A') : *prefix;
        if (cs != cp) return false;
        s++; prefix++;
    }
    return true;
}

static void sendBleReply(const char *reply) {
    if (reply && reply[0]) sendBLEMessage(reply);
}

/* BLE NUS RX 命令：与 Arduino 线 REC_ 前缀与 WIFI_ 前缀命令对齐。
 * 此前队列有入无出，App 录制指令全丢（2026-09-11 修复）。
 * 注意：注释里不要写 "REC_*\/WIFI_*" —— 其中的 *\/ 会提前终结块注释，
 * 该写法曾导致 2e759e4 提交后固件无法编译（2026-09-12 构建验证时发现）。 */
static void handleBleCommands(void) {
    char cmd[64];
    char reply[96];
    while (bleCommandQueueTake(cmd, sizeof(cmd))) {
        if (!s_storageReady) {
            sendBleReply("REC_ERR not_ready");
            continue;
        }
        if (strEqualsIgnoreCase(cmd, "REC_STOP")) {
            uint32_t dur = ecgRecorderCurrentDurationSec();
            bool ok = ecgRecorderStop();
            s_schedActiveRec = false;
            if (s_schedInterval > 0) {
                s_schedNextStart = (uint32_t)(esp_timer_get_time() / 1000000ULL) + s_schedInterval;
            }
            snprintf(reply, sizeof(reply), "REC_STOP %s %lus", ok ? "ok" : "fail",
                     (unsigned long)dur);
            sendBleReply(reply);
            continue;
        }
        if (strEqualsIgnoreCase(cmd, "REC_START")) {
            bool ok = ecgRecorderStart();
            s_schedActiveRec = false;
            snprintf(reply, sizeof(reply), "REC_START %s", ok ? "ok" : "fail");
            sendBleReply(reply);
            continue;
        }
        if (strEqualsIgnoreCase(cmd, "REC_STATUS")) {
            snprintf(reply, sizeof(reply), "REC_STATUS rec=%d auto=%d count=%lu",
                     ecgRecorderIsRecording() ? 1 : 0,
                     ecgRecorderAutoRecordEnabled() ? 1 : 0,
                     (unsigned long)ecgRecorderRecordCount());
            sendBleReply(reply);
            continue;
        }
        if (strEqualsIgnoreCase(cmd, "REC_LIST")) {
            char listBuf[512];
            int n = ecgRecorderList(listBuf, (int)sizeof(listBuf));
            if (n <= 0 || listBuf[0] == '\0') {
                sendBleReply("REC_LIST empty");
            } else {
                sendBleReply("REC_LIST ok");
                char *line = listBuf;
                while (line && *line) {
                    char *nl = strchr(line, '\n');
                    if (nl) *nl = '\0';
                    if (line[0]) sendBleReply(line);
                    line = nl ? (nl + 1) : NULL;
                }
            }
            continue;
        }
        if (strEqualsIgnoreCase(cmd, "REC_AUTO 0")) {
            ecgRecorderSetAutoRecord(false);
            sendBleReply("REC_AUTO 0 ok");
            continue;
        }
        if (strEqualsIgnoreCase(cmd, "REC_AUTO 1")) {
            ecgRecorderSetAutoRecord(true);
            sendBleReply("REC_AUTO 1 ok");
            continue;
        }
        if (strStartsWithIgnoreCase(cmd, "REC_SCHEDULE")) {
            const char *arg = cmd + 12;
            while (*arg == ' ') arg++;
            if (strEqualsIgnoreCase(arg, "OFF")) {
                s_schedInterval = 0;
                s_schedDuration = 0;
                if (s_schedActiveRec) {
                    ecgRecorderStop();
                    s_schedActiveRec = false;
                }
                sendBleReply("REC_SCHEDULE OFF ok");
                continue;
            }
            unsigned long iv = 0, dur = 0;
            if (sscanf(arg, "%lu %lu", &iv, &dur) == 2 && iv >= 10 && dur >= 5) {
                s_schedInterval = (uint32_t)iv;
                s_schedDuration = (uint32_t)dur;
                s_schedNextStart = (uint32_t)(esp_timer_get_time() / 1000000ULL) + (uint32_t)iv;
                snprintf(reply, sizeof(reply), "REC_SCHEDULE ok %lus %lus", iv, dur);
                sendBleReply(reply);
            } else {
                sendBleReply("REC_SCHEDULE fail");
            }
            continue;
        }
        if (strEqualsIgnoreCase(cmd, "WIFI_ON")) {
            bool ok = ecgWifiStart();
            snprintf(reply, sizeof(reply), "WIFI_ON %s", ok ? "ok" : "fail");
            sendBleReply(reply);
            continue;
        }
        if (strEqualsIgnoreCase(cmd, "WIFI_OFF")) {
            ecgWifiStop();
            sendBleReply("WIFI_OFF ok");
            continue;
        }
    }
}

static void processRecSchedule(void) {
    if (s_schedInterval == 0 || !s_storageReady) return;
    uint32_t nowSec = (uint32_t)(esp_timer_get_time() / 1000000ULL);
    if (!s_schedActiveRec && nowSec >= s_schedNextStart) {
        if (ecgRecorderStart()) {
            s_schedActiveRec = true;
        }
        s_schedNextStart = nowSec + s_schedInterval;
    }
    if (s_schedActiveRec && ecgRecorderCurrentDurationSec() >= s_schedDuration) {
        ecgRecorderStop();
        s_schedActiveRec = false;
    }
}

static void storage_init_task(void *arg) {
    (void)arg;
    if (!ecgRecorderInit()) {
        printf("[storage] ecgRecorderInit failed\n");
        vTaskDelete(NULL);
        return;
    }
    /* 异常触发自动录制: 异常秒 -> 启动, 连续 N 秒正常 -> 停止。
     * BLE REC_* 命令已接入；auto-record 仍默认开，保证无 App 时也能落盘。 */
    ecgRecorderSetAutoRecord(true);
    s_storageReady = true;
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
    if (!ecg_ai_init(models_ecg_model_v3a_int8_tflite,
                     models_ecg_model_v3a_int8_tflite_len, &cfg)) {
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
        uint64_t tWork0 = esp_timer_get_time();
        checkButton();
        handleBleCommands();
        processRecSchedule();

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
        /* 显示滤波只喂 BLE 波形与 TICK: 未连接时跳过 (省 ~2 个双精度 biquad/样本),
         * TICK 的 disp 列降级为梳状输出; 重连时清一次状态消瞬态。 */
        bool bleNowConnected = isBLEConnected();
        float displaySample;
        if (bleNowConnected) {
            if (!s_bleWasConnected) displayFilterReset();
            displaySample = applyDisplayFilter(combOut);
        } else {
            displaySample = combOut;
        }
        s_bleWasConnected = bleNowConnected;
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
#if ECG_HOT_LOG
            printf("AI_RESULT,%.4f,%u,%u\n", r.confidence, (unsigned)r.raw_abnormal, (unsigned)r.confirmed);
#endif
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
            /* 可穿戴低功耗：攒 2 帧再发一次 Notify，减少 BLE 射频活动。
             * 对 App 仍等效 125 样本/s（每次 Notify 2 个分号帧）。 */
            int lineLen = (int)strlen(ble_line);
            if (lineLen > 0 && s_bleBatchLen + lineLen < (int)sizeof(s_bleBatchBuf)) {
                memcpy(s_bleBatchBuf + s_bleBatchLen, ble_line, lineLen);
                s_bleBatchLen += lineLen;
                s_bleBatchCount++;
            }
            if (s_bleBatchCount >= BLE_BATCH_SIZE) {
                s_bleBatchBuf[s_bleBatchLen] = '\0';
                sendBLEMessage(s_bleBatchBuf);
                s_bleBatchLen = 0;
                s_bleBatchCount = 0;
            }
        }

        if (frame % 500 == 0) {
            uint64_t nowUs = esp_timer_get_time();
#if ECG_HOT_LOG
            uint32_t busyPct = s_lastTickUs
                ? (uint32_t)((s_busyAccumUs * 100) / (nowUs - s_lastTickUs)) : 0;
            printf("TICK,%lu,src=%s,bpm=%u,sqi=%.3f,disp=%.4f,comb=%.4f,busy=%u%%,ovr=%lu\n",
                   (unsigned long)frame, modeName(s_mode),
                   (unsigned)hr.bpm, hr.sqi, displaySample, combOut,
                   (unsigned)busyPct, (unsigned long)s_loopOverruns);
#endif
            s_busyAccumUs = 0;
            s_loopOverruns = 0;
            s_lastTickUs = nowUs;
        }
        frame++;
        {
            uint64_t dtWorkUs = (uint64_t)esp_timer_get_time() - tWork0;
            s_busyAccumUs += dtWorkUs;
            if (dtWorkUs > 2000) s_loopOverruns++;
        }
        vTaskDelay(pdMS_TO_TICKS(2));
    }
}
