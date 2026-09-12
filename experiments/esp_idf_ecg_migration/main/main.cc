#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "driver/gpio.h"
#include "driver/uart.h"
#ifdef CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG_ENABLED
#include "driver/usb_serial_jtag.h"
#endif
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

/* ================= M1 报警擎住状态机 (2026-09-12) =================
 * 背景: App 冻结不可改, 报警持续性必须在固件侧。原链路 BLE 第 8 列跟随
 * ecg_ai 的 confirmed (1-of-5 + 冷却 5 拍), 持续异常下呈 1s 亮 / 5s 灭闪断
 * (§104 基线: 90s 连续病理信号, 报警位从未持续超过 1s)。
 * 策略: 任何生理报警源触发 -> abnormal 位擎住 >= 30s; 解除需信号质量恢复 +
 * 最近窗 raw 密度回落 + 规则源静默。AI 内部冷却只用于事件计数, 不再压制
 * 持续 (AI_RESULT 的 confirmed 列保持原语义, 供对照/调参)。
 * 注意: 注释中禁止出现 "某词加斜杠星号" 组合, 会提前终结块注释 (2e759e4)。 */
#define ALARM_MIN_HOLD_S     30      /* 最短擎住时长 (秒) */
#define ALARM_AI_WIN         30      /* AI 进入判据: 最近 30 窗中 raw 异常 >= ALARM_AI_TH */
#define ALARM_AI_TH          10      /* (v3-A replay_normal raw 密度实测后再定稿) */
#define ALARM_REL_WIN        10      /* 解除判据: 最近 10 窗 raw 异常 <= ALARM_REL_TH */
#define ALARM_REL_TH         1
#define ALARM_REL_SQI_MIN    0.50f   /* 解除判据: SQI 需恢复到此线 */
#define ALARM_ASYSTOLE_MS    4000u   /* 距最近心拍超时 -> 时间停搏 (电极脱落等效) */
#define ALARM_VF_GATE_MS     2500u   /* VF 互锁: 距最近拍小于此值不放行 (压正常窦律误报) */
#define ALARM_SRC_AI         0x01u   /* AI 持续密度判据 */
#define ALARM_SRC_RS         0x02u   /* 规则: RR 停搏 / 30s 窗过缓 / 过速 */
#define ALARM_SRC_FLAT       0x04u   /* 时间停搏 (无拍 >= 4s) */
#define ALARM_SRC_VF         0x08u   /* VF/VT 两窗确认 */

typedef enum {
    SOURCE_SIMULATOR = 0,
    SOURCE_REPLAY_NORMAL,
    SOURCE_REPLAY_ABNORMAL,
    SOURCE_AFE_REAL
} InputSource;

static int s_mode = SOURCE_SIMULATOR;
static uint32_t s_lastButtonMs = 0;
static bool s_buttonWasLow = false;
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

/* ---- M1 报警擎住状态机状态 ---- */
static bool     s_alarmLatched = false;    /* 擎住中的报警位 (BLE 第 8 列来源) */
static uint32_t s_alarmLastTrigSec = 0;    /* 最近一次触发时刻 (擎住计时锚点) */
static uint8_t  s_alarmSrc = 0;            /* 擎住期间累计的触发源位图 */
static uint8_t  s_ruleHitSec = 0;          /* 本秒内命中过的规则源 (逐帧累积, 秒边界清零) */
static bool     s_aiConfirmedSec = false;  /* 本秒内出现过 AI confirmed (录制位图用) */
static uint8_t  s_aiRawHist[ALARM_AI_WIN]; /* 最近 N 窗 raw 异常环形历史 */
static int      s_aiRawHistLen = 0;
static int      s_aiRawHistIdx = 0;
static uint32_t s_lastBeatSeenMs = 0;      /* 主循环侧独立跟踪最近拍 (hr 内部软复位会刷新自己的时间戳) */
static bool     s_beatEverSeen = false;    /* 本模式会话内是否见过心拍 (时间停搏的武装条件) */
static float    s_sqiSec = 0.0f;           /* 秒边界 SQI 快照 (解除判据用) */


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

/* ---- M1 报警擎住状态机 ---- */
static void alarmReset(void) {
    s_alarmLatched = false;
    s_alarmSrc = 0;
    s_alarmLastTrigSec = 0;
    memset(s_aiRawHist, 0, sizeof(s_aiRawHist));
    s_aiRawHistLen = 0;
    s_aiRawHistIdx = 0;
    s_ruleHitSec = 0;
    s_aiConfirmedSec = false;
    s_lastBeatSeenMs = 0;
    s_beatEverSeen = false;
    s_sqiSec = 0.0f;
}

static void alarmPushAiRaw(uint8_t raw) {
    s_aiRawHist[s_aiRawHistIdx] = raw;
    s_aiRawHistIdx = (s_aiRawHistIdx + 1) % ALARM_AI_WIN;
    if (s_aiRawHistLen < ALARM_AI_WIN) s_aiRawHistLen++;
}

/* 最近 win 窗 (不超过已填数) 中 raw=1 的个数 */
static int alarmAiDensity(int win) {
    if (win > s_aiRawHistLen) win = s_aiRawHistLen;
    int cnt = 0;
    for (int k = 1; k <= win; k++) {
        int idx = (s_aiRawHistIdx - k + ALARM_AI_WIN) % ALARM_AI_WIN;
        cnt += s_aiRawHist[idx];
    }
    return cnt;
}

/* 秒边界推进: ruleHits = 本秒规则源命中位图 (传入后本函数负责消费) */
static void alarmSecondTick(uint32_t nowSec, uint8_t ruleHits) {
    uint8_t trig = ruleHits;
    int dens = alarmAiDensity(ALARM_AI_WIN);
    if (dens >= ALARM_AI_TH) trig |= ALARM_SRC_AI;
    if (trig != 0) {
        if (!s_alarmLatched) {
            printf("[ALARM] LATCH src=0x%02x t=%lus dens=%d\n",
                   trig, (unsigned long)nowSec, dens);
        } else if (trig & ~s_alarmSrc) {
            printf("[ALARM] EXTEND +0x%02x t=%lus\n",
                   (unsigned char)(trig & ~s_alarmSrc), (unsigned long)nowSec);
        }
        s_alarmLatched = true;
        s_alarmSrc |= trig;
        s_alarmLastTrigSec = nowSec;   /* 持续触发 -> 擎住窗顺延 */
    } else if (s_alarmLatched
               && (nowSec - s_alarmLastTrigSec) >= ALARM_MIN_HOLD_S
               && s_sqiSec >= ALARM_REL_SQI_MIN
               && alarmAiDensity(ALARM_REL_WIN) <= ALARM_REL_TH) {
        printf("[ALARM] CLEAR held=%lus src=0x%02x t=%lus\n",
               (unsigned long)(nowSec - s_alarmLastTrigSec),
               s_alarmSrc, (unsigned long)nowSec);
        s_alarmLatched = false;
        s_alarmSrc = 0;
    }
}

static void switchMode(int newMode) {
    s_mode = newMode;
    printf("[main] mode -> %s\n", modeName(newMode));
    alarmReset();
    filterReset();
    aiFilterReset();
    hrFullReset();
    rsReset();
    afReset();
    vfReset();
    if (newMode == SOURCE_REPLAY_NORMAL) ecgReplaySetSegment(ECG_REPLAY_SEG_NORMAL);
    if (newMode == SOURCE_REPLAY_ABNORMAL) ecgReplaySetSegment(ECG_REPLAY_SEG_ABNORMAL);
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

static void cmdReply(bool fromUart, const char *msg) {
    if (fromUart) {
        printf("CMD %s\n", msg);
    } else {
        sendBleReply(msg);
    }
}

/* 命令处理器 (M1): BLE NUS 与 UART0 RX 共用同一命令集。
 * - fromUart=1: 应答走串口 (CMD 前缀), pc_tools 全自动驱动;
 * - fromUart=0: 应答走 BLE, 与 App 的旧行为逐字节兼容。
 * 新增自动化命令 (两个通道均可发): MODE / REPLAY / STATUS / PING。
 * REC 与 WIFI 前缀命令集合保持不变 (App 冻结, 逐字节兼容)。
 * 注意: 注释里不要写 "REC 前缀加斜杠星号" 之类组合, 会提前终结块注释。 */
static void processCommand(const char *cmd, bool fromUart) {
    char reply[128];

    /* ---- 自动化命令 (不依赖存储就绪) ---- */
    if (strEqualsIgnoreCase(cmd, "PING")) {
        snprintf(reply, sizeof(reply), "PONG %lu",
                 (unsigned long)(esp_timer_get_time() / 1000));
        cmdReply(fromUart, reply);
        return;
    }
    if (strEqualsIgnoreCase(cmd, "STATUS")) {
        snprintf(reply, sizeof(reply),
                 "STATUS mode=%s seg=%u alarm=%u asrc=0x%02x rec=%d ai_conf=%lu",
                 modeName(s_mode), (unsigned)ecgReplayGetSegment(),
                 s_alarmLatched ? 1u : 0u, s_alarmSrc,
                 ecgRecorderIsRecording() ? 1 : 0,
                 (unsigned long)ecg_ai_total_confirmed());
        cmdReply(fromUart, reply);
        return;
    }
    if (strStartsWithIgnoreCase(cmd, "MODE ")) {
        const char *m = cmd + 5;
        int src = -1, seg = -1;
        if (strEqualsIgnoreCase(m, "sim") || strEqualsIgnoreCase(m, "simulator")) {
            src = SOURCE_SIMULATOR;
        } else if (strEqualsIgnoreCase(m, "replay_normal")) {
            src = SOURCE_REPLAY_NORMAL; seg = ECG_REPLAY_SEG_NORMAL;
        } else if (strEqualsIgnoreCase(m, "replay_abnormal")) {
            src = SOURCE_REPLAY_ABNORMAL; seg = ECG_REPLAY_SEG_ABNORMAL;
        } else if (strEqualsIgnoreCase(m, "replay_flat")) {
            /* 平线验收段: 正常 15s -> 平线 45s 循环 (M1) */
            src = SOURCE_REPLAY_NORMAL; seg = ECG_REPLAY_SEG_FLAT;
        } else if (strEqualsIgnoreCase(m, "afe") || strEqualsIgnoreCase(m, "afe_real")) {
            src = SOURCE_AFE_REAL;
        }
        if (src >= 0) {
            switchMode(src);
            if (seg >= 0) ecgReplaySetSegment((uint8_t)seg);
            snprintf(reply, sizeof(reply), "MODE ok %s", modeName(s_mode));
        } else {
            snprintf(reply, sizeof(reply), "MODE fail '%s'", m);
        }
        cmdReply(fromUart, reply);
        return;
    }
    if (strStartsWithIgnoreCase(cmd, "REPLAY ")) {
        int n = atoi(cmd + 7);
        if (n >= 0 && n <= ECG_REPLAY_SEG_FLAT) {
            ecgReplaySetSegment((uint8_t)n);
            snprintf(reply, sizeof(reply), "REPLAY ok %d", n);
        } else {
            snprintf(reply, sizeof(reply), "REPLAY fail (0..%d)", ECG_REPLAY_SEG_FLAT);
        }
        cmdReply(fromUart, reply);
        return;
    }

    /* ---- 存储/网络命令: 需要文件系统就绪 (与旧 BLE 行为一致) ---- */
    if (!s_storageReady) {
        cmdReply(fromUart, "REC_ERR not_ready");
        return;
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
        cmdReply(fromUart, reply);
        return;
    }
    if (strEqualsIgnoreCase(cmd, "REC_START")) {
        bool ok = ecgRecorderStart();
        s_schedActiveRec = false;
        snprintf(reply, sizeof(reply), "REC_START %s", ok ? "ok" : "fail");
        cmdReply(fromUart, reply);
        return;
    }
    if (strEqualsIgnoreCase(cmd, "REC_STATUS")) {
        snprintf(reply, sizeof(reply), "REC_STATUS rec=%d auto=%d count=%lu",
                 ecgRecorderIsRecording() ? 1 : 0,
                 ecgRecorderAutoRecordEnabled() ? 1 : 0,
                 (unsigned long)ecgRecorderRecordCount());
        cmdReply(fromUart, reply);
        return;
    }
    if (strEqualsIgnoreCase(cmd, "REC_LIST")) {
        char listBuf[512];
        int n = ecgRecorderList(listBuf, (int)sizeof(listBuf));
        if (n <= 0 || listBuf[0] == '\0') {
            cmdReply(fromUart, "REC_LIST empty");
        } else {
            cmdReply(fromUart, "REC_LIST ok");
            char *line = listBuf;
            while (line && *line) {
                char *nl = strchr(line, '\n');
                if (nl) *nl = '\0';
                if (line[0]) cmdReply(fromUart, line);
                line = nl ? (nl + 1) : NULL;
            }
        }
        return;
    }
    if (strEqualsIgnoreCase(cmd, "REC_AUTO 0")) {
        ecgRecorderSetAutoRecord(false);
        cmdReply(fromUart, "REC_AUTO 0 ok");
        return;
    }
    if (strEqualsIgnoreCase(cmd, "REC_AUTO 1")) {
        ecgRecorderSetAutoRecord(true);
        cmdReply(fromUart, "REC_AUTO 1 ok");
        return;
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
            cmdReply(fromUart, "REC_SCHEDULE OFF ok");
            return;
        }
        unsigned long iv = 0, dur = 0;
        if (sscanf(arg, "%lu %lu", &iv, &dur) == 2 && iv >= 10 && dur >= 5) {
            s_schedInterval = (uint32_t)iv;
            s_schedDuration = (uint32_t)dur;
            s_schedNextStart = (uint32_t)(esp_timer_get_time() / 1000000ULL) + (uint32_t)iv;
            snprintf(reply, sizeof(reply), "REC_SCHEDULE ok %lus %lus", iv, dur);
            cmdReply(fromUart, reply);
        } else {
            cmdReply(fromUart, "REC_SCHEDULE fail");
        }
        return;
    }
    if (strEqualsIgnoreCase(cmd, "WIFI_ON")) {
        bool ok = ecgWifiStart();
        snprintf(reply, sizeof(reply), "WIFI_ON %s", ok ? "ok" : "fail");
        cmdReply(fromUart, reply);
        return;
    }
    if (strEqualsIgnoreCase(cmd, "WIFI_OFF")) {
        ecgWifiStop();
        cmdReply(fromUart, "WIFI_OFF ok");
        return;
    }
}

static void handleBleCommands(void) {
    char cmd[64];
    while (bleCommandQueueTake(cmd, sizeof(cmd))) {
        processCommand(cmd, false);
    }
}

/* UART0 RX 命令通道 (M1): 串口 460800, 行缓冲 (换行结束)。
 * console 的 TX 路径不受影响 (只装 RX 侧驱动, 不切换 VFS)。
 * 另: 板子常经原生 USB-Serial-JTAG 口连接 (Windows 枚举为 "USB 串行设备"),
 * USJ 的 RX FIFO 极小, 固件不读则主机写必堵 —— 所以命令通道同时挂 USJ。 */
static char s_uartLine[96];
static int  s_uartLineLen = 0;
#ifdef CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG_ENABLED
static char s_usjLine[96];
static int  s_usjLineLen = 0;
#endif

static void feedCommandChar(char c, char *line, int *len, int cap) {
    if (c == '\n' || c == '\r') {
        if (*len > 0) {
            line[*len] = '\0';
            processCommand(line, true);
            *len = 0;
        }
    } else if (*len < cap - 1) {
        line[(*len)++] = c;
    } else {
        *len = 0;   /* 超长行丢弃, 防粘包 */
    }
}

static void uartCommandInit(void) {
    uart_driver_install(UART_NUM_0, 512, 0, 0, NULL, 0);
#ifdef CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG_ENABLED
    usb_serial_jtag_driver_config_t usjCfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    usb_serial_jtag_driver_install(&usjCfg);
#endif
}

static void uartCommandPoll(void) {
    uint8_t buf[64];
    int n = uart_read_bytes(UART_NUM_0, buf, sizeof(buf), 0);
    for (int i = 0; i < n; i++) {
        feedCommandChar((char)buf[i], s_uartLine, &s_uartLineLen, (int)sizeof(s_uartLine));
    }
#ifdef CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG_ENABLED
    n = usb_serial_jtag_read_bytes(buf, sizeof(buf), 0);
    for (int i = 0; i < n; i++) {
        feedCommandChar((char)buf[i], s_usjLine, &s_usjLineLen, (int)sizeof(s_usjLine));
    }
#endif
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
    uartCommandInit();

    uint32_t frame = 0;
    float last_conf = 0.0f;
    printf("[main] ESP-IDF ECG demo start, mode=%s\n", modeName(s_mode));

    while (true) {
        uint64_t tWork0 = esp_timer_get_time();
        checkButton();
        uartCommandPoll();
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
        (void)af;   /* Lane A 研究线 (AF 头) 精度未达部署线, 暂不入报警通道 */

        /* ---- M1: 规则通道接线 (此前 rs 结果被丢弃, vfDetect 只 init 从未调用) ---- */
        uint32_t nowMs = (uint32_t)(esp_timer_get_time() / 1000);
        if (rs.asystole || rs.bradycardia || rs.tachycardia) {
            s_ruleHitSec |= ALARM_SRC_RS;
        }
        if (hr.beatDetected) {
            s_lastBeatSeenMs = nowMs;
            s_beatEverSeen = true;
        }
        /* 时间停搏 (电极脱落等效): 主循环侧独立跟踪最近拍时刻。
         * 不能用 hrGetLastBeatMillis —— hr 内部 3s 超时软复位会把它刷新为
         * 当前时间 (heartrate.cpp hrReset), 持续无拍时永远测不出 4s 间隔。 */
        if (s_beatEverSeen && (nowMs - s_lastBeatSeenMs) >= ALARM_ASYSTOLE_MS) {
            s_ruleHitSec |= ALARM_SRC_FLAT;
        }
        /* VF/VT: 250Hz 喂入 (2:1 抽取, 5s 窗按 250Hz 标定), mV 域。
         * 换算沿用 Arduino v2 校准: AFE/SIM 为 V 域乘 0.763; IDF 回放数据
         * 本身就是 mV 域 (mV 乘 1, 2026-08-08 再生成), 不再乘 0.001。
         * 互锁: 距最近拍 < 2.5s 不放行 (正常窦律 VF 误报抑制, AFE 实测教训)。 */
        if ((frame % 2) == 0) {
            float vfIn = filteredSample;
            if (s_mode != SOURCE_REPLAY_NORMAL && s_mode != SOURCE_REPLAY_ABNORMAL) {
                vfIn *= 0.763f;
            }
            VF_Result vf = vfProcess(vfIn);
            if (vf.vfAlarm && s_beatEverSeen
                && (nowMs - s_lastBeatSeenMs) > ALARM_VF_GATE_MS) {
                s_ruleHitSec |= ALARM_SRC_VF;
            }
        }

        ecg_ai_result_t r;
        while (ecg_ai_pop_result(&r)) {
            last_conf = r.confidence;
            if (r.confirmed) s_aiConfirmedSec = true;
            alarmPushAiRaw(r.raw_abnormal);
#if ECG_HOT_LOG
            printf("AI_RESULT,%.4f,%u,%u,%lu\n", r.confidence,
                   (unsigned)r.raw_abnormal, (unsigned)r.confirmed,
                   (unsigned long)r.latency_us);
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
                uint8_t ruleHits = s_ruleHitSec;
                s_sqiSec = hr.sqi;
                alarmSecondTick(nowSec, ruleHits);
                /* 录制位图 (M1): 擎住位 OR 本秒规则命中 OR 本秒 AI confirmed。
                 * 平线/停搏现在会触发自动录制 (修复前只有 AI confirmed 能触发)。 */
                bool abnormalSec = s_alarmLatched || (ruleHits != 0) || s_aiConfirmedSec;
                ecgRecorderSetSecondAbnormal(abnormalSec);
                s_aiConfirmedSec = false;
            }
        }

        if ((frame % 4) == 0 && isBLEConnected()) {
            char ble_line[160];
            uint8_t trueBPM = (s_mode == SOURCE_SIMULATOR) ? ecgSimulatorGetTrueBPM() : 0;
            /* 第 8 列 (M1 起) = 固件侧擎住的报警位 (持续异常持续为 1);
             * 第 9 列 = 最近一次推理置信度 (不变)。帧格式逐字节兼容。 */
            snprintf(ble_line, sizeof(ble_line),
                     "%.3f,%.3f,%.3f,%u,%u,%.2f,0,%d,%.3f;",
                     cleanSample, noisyNoDC, displaySample,
                     (unsigned)hr.bpm, (unsigned)trueBPM, hr.sqi,
                     s_alarmLatched ? 1 : 0, last_conf);
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
            printf("TICK,%lu,src=%s,bpm=%u,sqi=%.3f,disp=%.4f,comb=%.4f,busy=%u%%,ovr=%lu,alarm=%u,asrc=0x%02x,seg=%u\n",
                   (unsigned long)frame, modeName(s_mode),
                   (unsigned)hr.bpm, hr.sqi, displaySample, combOut,
                   (unsigned)busyPct, (unsigned long)s_loopOverruns,
                   s_alarmLatched ? 1u : 0u, s_alarmSrc,
                   (unsigned)ecgReplayGetSegment());
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
