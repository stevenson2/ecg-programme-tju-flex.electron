#include <Arduino.h>
#include <esp_task_wdt.h>
#include "esp_timer.h"
#include "filter/filter.h"
#include "bluetooth/ble.h"
#include "signal_generator/ecg_simulator.h"
#include "signal_generator/ecg_replay.h"
#include "adc_afe/afe_hal.h"
#include "heartrate/heartrate.h"
#include "rhythm_safety/rhythm_safety.h"
#include "af_detect/af_detect.h"
#include "vf_detect/vf_detect.h"
#include "thermal/thermal.h"
#include "ai_inference/ai_inference.h"
#include "storage/ecg_recorder.h"
#include "wifi/ecg_wifi.h"

/**
 * @file main.cpp
 * @brief ESP32-S3 心电采集系统 - 主程序入口
 *
 * == 双输入模式 ==
 *   ① 模拟模式 (SOURCE_SIMULATOR): ecg_simulator 生成临床级心电
 *   ② 真实模式 (SOURCE_AFE_REAL):   ADC 采集自研 PCB 模拟前端信号
 *
 *   通过 GPIO0 按键 (BOOT按钮) 或串口 'm' 指令实时切换,
 *   无需重新编译烧录。
 *
 * == 按键接线 (自选) ==
 *   默认使用 GPIO0 (ESP32-S3-DevKitM-1 板载 BOOT 按钮):
 *     - 按下 → GND (默认上拉 3.3V)
 *     - 每次按下切换输入源
 *
 *   如需外接按键:
 *     GPIO___ → 按键 → GND
 *     ↑ 改 BUTTON_PIN 宏即可
 *
 * == 数据流 ==
 *   [模拟发生器 / 真实AFE] → 梳状滤波(50/100Hz) → HP 0.5Hz → LP 40Hz → 心率检测 → BLE发送(NUS) + 串口
 *
 * == v3.0 优化 ==
 *   - CPU 240MHz 性能模式 (原80MHz)
 *   - 滤波链精简: 移除独立50/100Hz陷波器，由梳状统一处理
 *   - BLE 4帧批量打包发送，吞吐量提升4×
 *
 * == 信号电平说明 ==
 *   - clean  ：纯净心电波形 / 去偏置ADC信号, ±1.2V
 *   - noisy  ：含噪声原始信号 (偏置已去除), 与clean同基准
 *   - filtered：数字滤波后的信号
 *
 * 三通道同基准显示，方便手机 App 叠加对比
 */

/* ======================== 常量定义 ======================== */
#define SAMPLE_INTERVAL_MS  2   /* 500Hz 采样间隔 */
#define SAMPLE_INTERVAL_US  2000 /* 500Hz 采样间隔 (esp_timer 微秒) */
#define DC_OFFSET_REMOVE    1.65f  /* 去除 ADC 直流偏置，统一显示基准 */

/* ======================== 开发板适配 ======================== */
/*
 * ESP32-S3-SUPERMINI (ESP32S3FH4R2) 适配:
 *   LED: GPIO48 RGB 共阳极, LOW=亮, HIGH=灭
 *   USB: 内置 USB-Serial-JTAG, 需要等待枚举
 */
#define LED_ACTIVE_LEVEL   LOW   /* 共阳极: LOW 点亮 */

/* ======================== 按键配置 ======================== */
#define BUTTON_PIN          GPIO_NUM_0   /* 板载 BOOT 按钮 (低电平有效) */
#define BUTTON_DEBOUNCE_MS  200          /* 消抖延时 (毫秒) */

/* ======================== AFE 引脚配置 ======================== */
/* ★ 重要: 根据你的 PCB 改这里 ★ */
#define AFE_ADC_PIN         GPIO_NUM_4   /* AFE 输出接在哪个 GPIO */
#define AFE_DC_BIAS         1.65f        /* PCB 输出的直流偏置 (V) */
#define AFE_VREF            3.3f         /* ESP32 ADC 参考电压 */
#define AFE_OVERSAMPLE      4            /* 过采样次数 (500Hz下降为4x以节省时间) */

/* ======================== AFE 导联脱落检测引脚 ======================== */
/* AD8232 LOD+/LOD- 为推挽输出 (VOH≈2.9V/VOL≈0.05V, 无需上拉):
 *   高 = 对应导联脱落, 低 = 连接正常 (DC 模式, 三电极)
 * LO+ 接 IO5 (检测 +IN 脱落), LO- 接 IO6 (检测 -IN 脱落) */
#define AFE_LOD_P_PIN       GPIO_NUM_5   /* LOD+ (+IN 导联脱落检测) */
#define AFE_LOD_N_PIN       GPIO_NUM_6   /* LOD- (-IN 导联脱落检测) */

/* ======================== 输入模式枚举 ======================== */
typedef enum {
    SOURCE_SIMULATOR = 0,    /**< 模拟发生器模式 (默认, 无硬件也可运行) */
    SOURCE_REPLAY    = 1,    /**< 数据库回放模式 (MIT-BIH 正常/异常段, 2026-08-08) */
    SOURCE_AFE_REAL  = 2     /**< 真实 AFE 采集模式 */
} InputSource;

/* ======================== BLE 批量打包 ======================== */
/* 2026-08-10: 9 列帧后 4 帧/包达 25KB/s 超 BLE 4.2 实际吞吐 → 波形错乱;
 * 改为 2 帧/包 (12.5KB/s 安全区), 保留 9 列报警链路 */
#define BLE_BATCH_SIZE   2            /* 每2帧合包一次 */
#define BLE_BUF_SIZE     (64 * BLE_BATCH_SIZE + 4)  /* 约 132 字节 buffer */
static char s_bleBuf[BLE_BUF_SIZE];
static int  s_bleBufLen = 0;

/* BLE 通知分频 (DIAG NOTIFY 可调, 2026-08-10 引入):
 * 2=250Hz(原行为), 4=125Hz(默认, 2026-08-14 BLE 阶梯感根治)
 * 250Hz notify ≈11KB/s; 连接间隔退化到 30ms 时有效吞吐仅 ~6KB/s → 必丢帧/阶梯。
 * 125Hz ≈5.5KB/s, 30ms 间隔 + MTU185 下仍可承载; App 时间轴同步 125Hz。
 * DIAG NOTIFY 2 仅用于 PC/串口诊断; App 时间轴已按 125Hz 编译, App 联调勿切回。 */
static int s_bleNotifyDivider = 4;

/* ======================== 全局变量 ======================== */
static unsigned long lastSampleTime = 0;
static unsigned long frameCount = 0;

/* esp_timer 硬件采样节拍 (2026-08-14 根治 336Hz 帧率漂移): 500Hz 固定节拍, 帧率无关 */
static esp_timer_handle_t s_sampleTimer = NULL;
static volatile uint32_t s_sampleTick = 0;   /* 定时器回调递增 */
static uint32_t s_lastSampleTick = 0;        /* loop 已消费的节拍 */
static uint32_t s_sampleTickDrops = 0;         /* loop 跟不上 500Hz 时被丢弃的节拍 (诊断) */

/* 报警锁存 (2026-08-08): AI 报警触发后 abnormal 列持续 5 秒, 防止一闪而过
 * s_alarmHold: 剩余输出周期数 (@100Hz 串口输出), 500 = 5 秒 */
#define ALARM_HOLD_OUTS  500
static int32_t s_alarmHold = 0;
static float  s_alarmHoldConf = 0.0f;

/* 最近一次 AI 原始置信度 (诊断用, 2026-08-13): 独立于 LOD/flatline 强制报警,
 * 用于判断 AI 模型在真实 ECG 上的真实输出 (是否误报/漏报)。 */
static float  s_lastAiConf = 0.0f;
static uint8_t s_lastAiAbn  = 0;

/* 串口多字符命令行缓冲 (REC_* 指令) */
static char s_serialLine[48] = {0};
static int  s_serialLineLen     = 0;

/* 定时录制调度 (REC_SCHEDULE, 2026-08-10)
 * 基于上电秒数 (millis/1000), 不依赖 RTC/NTP —— 固件当前仅 AP 模式无外网,
 * 真实时钟待 STA 阶段; 记录文件 startUnix 仍为上电秒数 (见 TH §33/§34) */
static uint32_t s_schedInterval  = 0;  /* 0 = 调度关闭 */
static uint32_t s_schedDuration  = 0;  /* 每次录制时长(秒) */
static uint32_t s_schedNextStart = 0;  /* 下次开始的上电秒 */
static uint32_t s_schedStartSec  = 0;  /* 本次调度录制开始的秒 */
static bool     s_schedActiveRec = false; /* 当前录制是否由调度启动 */

/* 停搏/无信号检测 (2026-08-10, 用户真机反馈: 低电压直线 AI 不报异常)
 * 模型训练分布不含停搏场景; 临床监护停搏必须报警 →
 * filtered 信号连续 3 秒峰峰值 < 20mV 判定无信号/停搏, abnormal_flag 合并置 1 */
#define FLATLINE_SECS     3
#define FLATLINE_PP_V     0.02f
static float s_secMin = 1e9f, s_secMax = -1e9f;  /* 当前秒内 filtered 极值 */
static int   s_flatCount = 0;   /* 连续平坦秒计数 */
static bool  s_flatline = false; /* 停搏/无信号标志 */

/* 输入模式与按键状态 */
static InputSource  s_inputMode     = SOURCE_SIMULATOR;
static unsigned long s_lastBtnPress = 0;
static bool         s_btnLastState  = HIGH;   /* 上拉, 默认高电平 */

/*
 * ========== 50Hz 梳状滤波器 (v3.0 500Hz 适配) ==========
 * 两级滑动平均级联，500Hz/50Hz=10 → 10抽头:
 *   第1级: y1[n] = (x[n] + ... + x[n-9]) / 10
 *   第2级: y[n]  = (y1[n] + ... + y1[n-9]) / 10
 *
 * 利用 500Hz/50Hz = 10 的精确比，在 50Hz/100Hz 处精确陷零。
 * 双级级联效果:
 *   - 50Hz 衰减: -59.6dB + (-59.6dB) = -119.2dB
 *   - 有效阻带宽度提升 2 倍 (应对电网频率 ±0.5Hz 漂移)
 *   - QRS 10Hz 增益: -1.2dB (可接受)
 *   - 群延迟: 20ms (远小于RR间期800ms)
 *
 * 重要: 只能放在 main.cpp 且每帧只调用一次。
 */
#define COMB_TAPS   10
/* 第1级梳状 */
static float s_combBuf1[COMB_TAPS] = {0};
static int   s_combIdx1 = 0;
static float s_combSum1 = 0.0f;
/* 第2级梳状 */
static float s_combBuf2[COMB_TAPS] = {0};
static int   s_combIdx2 = 0;
static float s_combSum2 = 0.0f;

/**
 * @brief 双级级联梳状滤波器 (50Hz/100Hz 精确陷零)
 * @param x 输入样本
 * @return 滤波后样本
 */
static inline float applyCombFilter(float x)
{
    /* 第1级: 5抽头滑动平均 */
    s_combSum1 -= s_combBuf1[s_combIdx1];
    s_combBuf1[s_combIdx1] = x;
    s_combSum1 += x;
    s_combIdx1 = (s_combIdx1 + 1) % COMB_TAPS;
    float y1 = s_combSum1 / (float)COMB_TAPS;
    
    /* 第2级: 对第1级输出再做一次5抽头滑动平均 */
    s_combSum2 -= s_combBuf2[s_combIdx2];
    s_combBuf2[s_combIdx2] = y1;
    s_combSum2 += y1;
    s_combIdx2 = (s_combIdx2 + 1) % COMB_TAPS;
    return s_combSum2 / (float)COMB_TAPS;
}

/* ======================== 大小写无关字符串比较 ======================== */
static bool strEqualsIgnoreCase(const char* a, const char* b)
{
    while (*a && *b) {
        char ca = (*a >= 'a' && *a <= 'z') ? (*a - 'a' + 'A') : *a;
        char cb = (*b >= 'a' && *b <= 'z') ? (*b - 'a' + 'A') : *b;
        if (ca != cb) return false;
        a++; b++;
    }
    return *a == *b;
}

/** @brief 大小写无关前缀匹配 (供 REC_SCHEDULE <args> 类带参命令使用) */
static bool strStartsWithIgnoreCase(const char* s, const char* prefix)
{
    while (*prefix) {
        if (*s == '\0') return false;   /* 短命令防御: 避免读越界 (2026-08-14) */
        char cs = (*s >= 'a' && *s <= 'z') ? (*s - 'a' + 'A') : *s;
        char cp = (*prefix >= 'a' && *prefix <= 'z') ? (*prefix - 'a' + 'A') : *prefix;
        if (cs != cp) return false;
        s++; prefix++;
    }
    return true;
}

/* ======================== 命令解析器 (BLE + Serial 共享) ======================== */
/**
 * @brief 解析 REC_* / WIFI_* 命令并填充回复字符串
 * @param cmd      以 null 结尾的命令串
 * @param reply    输出回复缓冲区
 * @param replyLen 缓冲区大小
 * @return true 命令已识别并处理
 */
/* 串口发起的录制占用标记 (2026-08-13): 手机 App 定时录制经 BLE 周期性下发
 * REC_START/REC_STOP, 会与串口发起的微调数据录制互相打架 (实测 180s 录制 59s 即被
 * App 的 REC_STOP 掐断)。串口 REC_START 置位后, BLE 的 REC_START/REC_STOP 一律拒绝,
 * 直到串口 REC_STOP 清除。 */
static bool s_serialRecOwned = false;

static bool parseRecorderCommand(const char* cmd, char* reply, size_t replyLen,
                                 bool fromSerial)
{
    if (strEqualsIgnoreCase(cmd, "REC_STOP")) {
        if (!fromSerial && s_serialRecOwned && ecgRecorderIsRecording()) {
            snprintf(reply, replyLen, "REC_STOP busy (serial-owned)");
            return true;
        }
        uint32_t dur = ecgRecorderCurrentDurationSec();
        bool ok = ecgRecorderStop();
        if (fromSerial) s_serialRecOwned = false;
        /* 手动停止: 调度接管标记清除 + 下一轮推迟, 避免立即重启 */
        s_schedActiveRec = false;
        if (s_schedInterval > 0) {
            s_schedNextStart = (uint32_t)(millis() / 1000) + s_schedInterval;
        }
        snprintf(reply, replyLen, "REC_STOP %s %lus", ok ? "ok" : "fail",
                 (unsigned long)dur);
        return true;
    }
    if (strEqualsIgnoreCase(cmd, "REC_START")) {
        if (!fromSerial && s_serialRecOwned) {
            snprintf(reply, replyLen, "REC_START busy (serial-owned)");
            return true;
        }
        bool ok = ecgRecorderStart();
        if (ok && fromSerial) s_serialRecOwned = true;
        /* 手动开始: 当前录制不受调度自动停止干预 */
        s_schedActiveRec = false;
        snprintf(reply, replyLen, "REC_START %s", ok ? "ok" : "fail");
        return true;
    }
    if (strEqualsIgnoreCase(cmd, "REC_STATUS")) {
        snprintf(reply, replyLen, "REC_STATUS rec=%d auto=%d count=%lu",
                 ecgRecorderIsRecording() ? 1 : 0,
                 ecgRecorderAutoRecordEnabled() ? 1 : 0,
                 (unsigned long)ecgRecorderRecordCount());
        return true;
    }
    if (strEqualsIgnoreCase(cmd, "REC_LIST")) {
        /* 调用者负责处理多行输出: reply 仅返回标记 */
        snprintf(reply, replyLen, "REC_LIST");
        return true;
    }
    if (strEqualsIgnoreCase(cmd, "REC_AUTO 0")) {
        ecgRecorderSetAutoRecord(false);
        snprintf(reply, replyLen, "REC_AUTO 0 ok");
        return true;
    }
    if (strEqualsIgnoreCase(cmd, "REC_AUTO 1")) {
        ecgRecorderSetAutoRecord(true);
        snprintf(reply, replyLen, "REC_AUTO 1 ok");
        return true;
    }
    /* REC_SCHEDULE <间隔秒> <时长秒> | REC_SCHEDULE OFF
     * 定时录制: 每隔 <间隔秒> 自动录 <时长秒>, 基于上电秒数 (无 RTC) */
    if (strStartsWithIgnoreCase(cmd, "REC_SCHEDULE")) {
        const char* arg = cmd + 12;  /* strlen("REC_SCHEDULE") */
        while (*arg == ' ') arg++;
        if (strEqualsIgnoreCase(arg, "OFF")) {
            s_schedInterval = 0;
            s_schedDuration = 0;
            /* 关闭时同时停止正在进行的调度录制 */
            if (s_schedActiveRec) {
                ecgRecorderStop();
                s_schedActiveRec = false;
            }
            snprintf(reply, replyLen, "REC_SCHEDULE OFF ok");
            return true;
        }
        uint32_t iv = 0, dur = 0;
        if (sscanf(arg, "%lu %lu", &iv, &dur) == 2 && iv >= 10 && dur >= 5) {
            s_schedInterval = iv;
            s_schedDuration = dur;
            s_schedNextStart = (uint32_t)(millis() / 1000) + iv;
            snprintf(reply, replyLen, "REC_SCHEDULE ok %lus %lus",
                     (unsigned long)iv, (unsigned long)dur);
            return true;
        }
        snprintf(reply, replyLen, "REC_SCHEDULE fail (用法: REC_SCHEDULE <间隔秒> <时长秒> | OFF)");
        return true;
    }
    if (strEqualsIgnoreCase(cmd, "WIFI_ON")) {
        bool ok = ecgWifiStart();
        snprintf(reply, replyLen, "WIFI_ON %s", ok ? "ok" : "fail");
        return true;
    }
    if (strEqualsIgnoreCase(cmd, "WIFI_OFF")) {
        ecgWifiStop();
        snprintf(reply, replyLen, "WIFI_OFF ok");
        return true;
    }
    /* DIAG 诊断命令 (WiFi beacon 专项, 2026-08-10) — 一次烧录, 运行时切换单变量:
     *   DIAG              → 打印当前诊断配置
     *   DIAG TXP <v>      → AP 发射功率 (0=跳过 setTxPower, 34=8.5dBm, 60=15dBm, 78=19.5dBm)
     *   DIAG CH <1|6|11>  → AP 信道 (下次 WIFI_ON 生效)
     *   DIAG SEQ <0|1>    → AP 启动序列 (1=PR#1865 式慢速 OFF→AP 切换 + setSleep(false))
     *   DIAG NOTIFY <2|4> → BLE 通知分频 (4=125Hz 默认, 2=250Hz 原行为)
     *   DIAG AI <0|1>     → AI 推理开关
     * 证据: WiFiManager PR#1865 (2.0.17 世代 S3 实测) / ESP-IDF#13508 / ESPHome#6456 */
    if (strStartsWithIgnoreCase(cmd, "DIAG")) {
        const char* arg = cmd + 4;
        while (*arg == ' ') arg++;
        int v = 0;
        if (strEqualsIgnoreCase(arg, "") || strEqualsIgnoreCase(arg, "STATUS")) {
            char ip[24];
            ecgWifiDiagStaIp(ip, sizeof(ip));
            snprintf(reply, replyLen, "DIAG txp=%d ch=%d seq=%d notify=%d ai=%d mode=%d sta=%d ip=%s",
                     ecgWifiDiagGetTxPower(), ecgWifiDiagGetChannel(),
                     ecgWifiDiagGetSeqSlow() ? 1 : 0, s_bleNotifyDivider,
                     ai_inference_is_enabled() ? 1 : 0,
                     ecgWifiDiagGetMode(), ecgWifiDiagStaStatus(), ip);
            return true;
        }
        /* DIAG STA <ssid> <pass> — STA 连接测试 (候选D, 2026-08-10): AP_STA 共存,
         * 不停止 AP; 状态由 DIAG (sta=/ip=) 查询 */
        if (strStartsWithIgnoreCase(arg, "STA ") && strlen(arg) > 4) {
            char ssid[33] = {0}, pass[65] = {0};
            if (sscanf(arg + 4, "%32s %64s", ssid, pass) == 2) {
                bool ok = ecgWifiDiagStaConnect(ssid, pass);
                snprintf(reply, replyLen, "DIAG STA %s (ssid=%s, 状态用 DIAG 查询)",
                         ok ? "connecting" : "fail", ssid);
            } else {
                snprintf(reply, replyLen, "DIAG STA fail (用法: DIAG STA <ssid> <pass>)");
            }
            return true;
        }
        if (strEqualsIgnoreCase(arg, "STAOFF")) {
            ecgWifiDiagStaDisconnect();
            snprintf(reply, replyLen, "DIAG STAOFF ok (回到纯 AP)");
            return true;
        }
        if (strStartsWithIgnoreCase(arg, "TXP") && sscanf(arg + 3, "%d", &v) == 1) {
            if (v == 0 || v == 34 || v == 60 || v == 78) {
                ecgWifiDiagSetTxPower(v);
                snprintf(reply, replyLen, "DIAG TXP %d ok (下次 WIFI_ON 生效)", v);
            } else {
                snprintf(reply, replyLen, "DIAG TXP fail (0|34|60|78)");
            }
            return true;
        }
        if (strStartsWithIgnoreCase(arg, "CH") && sscanf(arg + 2, "%d", &v) == 1) {
            if (v == 1 || v == 6 || v == 11) {
                ecgWifiDiagSetChannel(v);
                snprintf(reply, replyLen, "DIAG CH %d ok (下次 WIFI_ON 生效)", v);
            } else {
                snprintf(reply, replyLen, "DIAG CH fail (1|6|11)");
            }
            return true;
        }
        if (strStartsWithIgnoreCase(arg, "SEQ") && sscanf(arg + 3, "%d", &v) == 1) {
            if (v == 0 || v == 1) {
                ecgWifiDiagSetSeqSlow(v == 1);
                snprintf(reply, replyLen, "DIAG SEQ %d ok (下次 WIFI_ON 生效)", v);
            } else {
                snprintf(reply, replyLen, "DIAG SEQ fail (0|1)");
            }
            return true;
        }
        if (strStartsWithIgnoreCase(arg, "NOTIFY") && sscanf(arg + 6, "%d", &v) == 1) {
            if (v == 2 || v == 4) {
                s_bleNotifyDivider = v;
                snprintf(reply, replyLen, "DIAG NOTIFY %d ok (%dHz)", v, 500 / v);
            } else {
                snprintf(reply, replyLen, "DIAG NOTIFY fail (2|4)");
            }
            return true;
        }
        if (strStartsWithIgnoreCase(arg, "AI") && sscanf(arg + 2, "%d", &v) == 1) {
            if (v == 0 || v == 1) {
                ai_inference_set_enabled(v == 1);
                snprintf(reply, replyLen, "DIAG AI %d ok", v);
            } else {
                snprintf(reply, replyLen, "DIAG AI fail (0|1)");
            }
            return true;
        }
        /* DIAG LPF <4|40> — 显示链低通截止频率切换 (2026-08-14,
         * 用户按 ADI 视频要求试验 4Hz 镜面平滑; 40Hz 恢复形态保真) */
        if (strStartsWithIgnoreCase(arg, "LPF") && sscanf(arg + 3, "%d", &v) == 1) {
            if (v == 4 || v == 40) {
                displaySetLpCutoff(v);
                snprintf(reply, replyLen, "DIAG LPF %d ok", v);
            } else {
                snprintf(reply, replyLen, "DIAG LPF fail (4|40)");
            }
            return true;
        }
        snprintf(reply, replyLen, "DIAG fail (TXP|CH|SEQ|NOTIFY|AI|LPF)");
        return true;
    }
    return false;
}

/* ======================== esp_timer 500Hz 采样节拍 ======================== */

/* 定时器回调 (ISR 上下文, 仅递增计数器, 快速无阻塞) */
static void sampleTimerCb(void *arg)
{
    (void)arg;
    s_sampleTick++;
}

static void sampleTimerStart(void)
{
    esp_timer_create_args_t cfg = {
        .callback = &sampleTimerCb,
        .arg = NULL,
        .dispatch_method = ESP_TIMER_TASK,
        .name = "ecg_sample",
    };
    if (esp_timer_create(&cfg, &s_sampleTimer) != ESP_OK) {
        Serial.println("[系统] esp_timer 创建失败, 回退 millis 软定时");
        return;
    }
    if (esp_timer_start_periodic(s_sampleTimer, SAMPLE_INTERVAL_US) != ESP_OK) {
        Serial.println("[系统] esp_timer 启动失败, 回退 millis 软定时");
        s_sampleTimer = NULL;
        return;
    }
    s_lastSampleTick = s_sampleTick;
    Serial.println("[系统] esp_timer 500Hz 采样节拍已启动 (根治 336Hz 帧率漂移)");
}

void setup()
{
    Serial.begin(460800);
#if ARDUINO_USB_CDC_ON_BOOT
    /* USB-Serial-JTAG 枚举较慢, 等待就绪 (最长 3 秒) */
    unsigned long usbStart = millis();
    while (!Serial && (millis() - usbStart) < 3000) {
        delay(10);
    }
#else
    delay(100);
#endif
    Serial.println();
    Serial.println("========================================");
    Serial.println(" ESP32-ECG-MONITOR 心电采集系统 v1.0");
    Serial.println(" 广播名称: ESP32-ECG");
    Serial.println(" 模式：软件验证模式（无外部电路）");
    Serial.println(" 串口格式：clean,noisy,filtered,bpm");
    Serial.println(" 板上心率:  简化 Pan-Tompkins QRS 检测");
    Serial.println("========================================");

    /* 初始化板载 LED (SUPERMINI 共阳极, LOW=亮) */
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, !LED_ACTIVE_LEVEL);  /* 初始熄灭 */

    /* 初始化按键 (内部上拉, 按下为 LOW) */
    pinMode(BUTTON_PIN, INPUT_PULLUP);

    /* 初始化各模块 */
    ecgSimulatorInit();
    Serial.println("[系统] 心电信号生成器已初始化");
    ecgReplayInit();   /* 数据库回放模式 (MIT-BIH) */

    /* 初始化真实 AFE 模块 (即使当前是模拟模式, 也准备好) */
    AFE_HAL_Config afeCfg = {
        .adcPin     = AFE_ADC_PIN,
        .dcBias     = AFE_DC_BIAS,
        .vRef       = AFE_VREF,
        .oversample = AFE_OVERSAMPLE,
        .enableCal  = true
    };
    afeHalInit(&afeCfg);

    /* AD8232 导联脱落检测引脚 (LOD+/LOD- 推挽输出, 无需上拉) */
    pinMode(AFE_LOD_P_PIN, INPUT);
    pinMode(AFE_LOD_N_PIN, INPUT);

    filterInit();
    filterWarmup(0.0f);  /* 预热滤波器，消除启动瞬态 */
    aiFilterInit();      /* AI 输入链 0.5Hz 高通 (2026-08-10, 匹配训练分布) */
    Serial.println("[系统] 数字滤波器已初始化");

    hrInit();
    Serial.println("[系统] 心率监测器已启动");

    rsInit();   /* T4-8 模块1: 心律安全逻辑 */
    afInit();   /* T4-8 模块3: AF RR 不规则度检测 */
    vfInit();   /* T4-9 模块2: VF/VT 检测 */
    Serial.println("[系统] T4-8/T4-9 心律安全 + AF + VF 检测已初始化");
    Serial.print("[系统] 模拟器真实心率: ");
    Serial.print(ecgSimulatorGetTrueBPM());
    Serial.println(" BPM");

    initBLE();

    /* 初始化 ECG 录制模块 (SPIFFS 挂载 + 扫描修复 + 索引重建) */
    if (ecgRecorderInit()) {
        Serial.println("[REC] 录制模块初始化完成");
    } else {
        Serial.println("[REC] 录制模块初始化失败 (SPIFFS?) — 继续运行");
    }

    /* 初始化 WiFi AP 传输模块 (注册路由, 不启动 AP)
     * 必须在 SPIFFS 挂载后、loop() 前调用 */
    ecgWifiInit();

    /* 2026-08-10 修复: 上电自动启动 AP (阶段B 产品行为)。
     * 背景: 串口打开/关闭触发 USB-Serial-JTAG 复位 (arduino-esp32 USB CDC
     * DTR 传统行为), 而 AP 为命令启动 → 串口一关设备即复位、AP 消失,
     * 手机/电脑无法连接 (用户实测"只有串口开着 WiFi 才能用")。
     * 改为 boot 自动启动后, 串口开关/设备重启均不影响 AP 可用性。
     * 保留 WIFI_OFF 命令可手动关闭 (省电场景)。 */
    if (!ecgWifiStart()) {
        Serial.println("[WiFi] 自动启动 AP 失败, 可用 WIFI_ON 手动启动");
    }

    /* 初始化 AI 推理模块 (Core 0) */
    if (ai_inference_init()) {
        Serial.println("[AI] 异常检测推理引擎已启动 (Core 0)");
    } else {
        Serial.println("[AI] 推理引擎初始化失败, 继续运行");
    }

    /* 任务看门狗诊断 (2026-08-14): 设备偶发卡死时主循环静默挂起 (无 panic —
     * 默认 TWDT 只监控 CPU0 idle, loopTask 在 CPU1 不受监控, 见 sdkconfig)。
     * 把 loopTask 挂进 TWDT (10s 超时): 卡死即 panic + 回溯 (USB-JTAG 控制台
     * 可读, 定位卡死点后按需移除本段); 同时让设备从"静默死"变为自愈重启。 */
    esp_task_wdt_init(10, true);
    esp_task_wdt_add(NULL);

    /* 性能模式: 240MHz 确保 4ms 帧内完成所有处理和 BLE 通知 */
    setCpuFrequencyMhz(240);
    Serial.print("[系统] CPU 频率: ");
    Serial.print(getCpuFrequencyMhz());
    Serial.println(" MHz (性能模式)");

    /* 初始化温度监测模块 */
    thermalInit();

    /* 默认启动模式: 无外接 AFE 时用模拟器, 方便调试 */
    s_inputMode = SOURCE_SIMULATOR;
    Serial.println("[系统] 当前输入: 模拟发生器");
    Serial.println("[系统] 按 BOOT 键或发 'm' 切换真实/模拟输入");

    lastSampleTime = millis();   /* 回退软定时基准 (esp_timer 失败时) */
    sampleTimerStart();
    Serial.println("[系统] 系统启动完成，开始采集...");
}

/* ======================== 输入模式切换逻辑 ======================== */

/**
 * @brief 检测按键是否被按下 (下降沿触发 + 消抖)
 *
 * 使用 BUTTON_PIN (默认 GPIO0, 板载 BOOT 按钮),
 * 内部上拉, 按下时读取 LOW。
 *
 * @return true  按键刚被按下 (仅返回一次)
 */
static bool isButtonPressed(void)
{
    bool currentState = digitalRead(BUTTON_PIN);
    unsigned long now = millis();

    /* 检测下降沿: 之前 HIGH → 现在 LOW, 且超过消抖时间 */
    if (s_btnLastState == HIGH && currentState == LOW
        && (now - s_lastBtnPress) > BUTTON_DEBOUNCE_MS)
    {
        s_lastBtnPress = now;
        s_btnLastState = LOW;
        return true;
    }

    /* 更新状态 (释放时恢复) */
    if (currentState == HIGH) {
        s_btnLastState = HIGH;
    }

    return false;
}

/**
 * @brief 切换输入模式
 *
 * 在 SOURCE_SIMULATOR ↔ SOURCE_AFE_REAL 之间切换。
 * 切换时自动复位滤波器, 消除模式切换导致的瞬态。
 * 打印当前模式到串口, LED 闪烁 3 次指示切换成功。
 */
static void toggleInputMode(void)
{
    if (s_inputMode == SOURCE_SIMULATOR) {
        s_inputMode = SOURCE_REPLAY;
        ecgReplayInit();
        Serial.println("\n>>> 切换至: 数据库回放模式 (MIT-BIH) <<<");
        Serial.println("    'n'=正常段(100)  'a'=异常段(106)");
    } else if (s_inputMode == SOURCE_REPLAY) {
        s_inputMode = SOURCE_AFE_REAL;
        Serial.println("\n>>> 切换至: 真实 AFE 采集模式 <<<");
    } else {
        s_inputMode = SOURCE_SIMULATOR;
        ecgSimulatorReset();  /* 重置发生器, 从新周期开始 */
        Serial.println("\n>>> 切换至: 模拟发生器模式 <<<");
    }

    /* 切换后复位滤波器与心率检测器, 消除瞬态 */
    filterReset();
    aiFilterReset();   /* AI 输入链 0.5Hz 高通同步重置 (2026-08-10) */
    hrFullReset();
    ai_inference_reset();  /* 重置 AI 推理缓冲 */
    
    /* v2.2: 复位双级梳状滤波器状态 */
    for (int i = 0; i < COMB_TAPS; i++) {
        s_combBuf1[i] = 0.0f;
        s_combBuf2[i] = 0.0f;
    }
    s_combIdx1 = 0;
    s_combSum1 = 0.0f;
    s_combIdx2 = 0;
    s_combSum2 = 0.0f;

    /* LED 闪烁 3 次指示切换 (SUPERMINI 共阳极) */
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_BUILTIN, LED_ACTIVE_LEVEL);     /* 亮 */
        delay(50);
        digitalWrite(LED_BUILTIN, !LED_ACTIVE_LEVEL);    /* 灭 */
        delay(50);
    }

    Serial.print("[系统] 当前输入模式: ");
    if (s_inputMode == SOURCE_SIMULATOR) {
        Serial.println("模拟");
    } else if (s_inputMode == SOURCE_REPLAY) {
        Serial.print("回放 (");
        Serial.print(ecgReplayGetSegment() == 0 ? "正常段 100" : "异常段 106");
        Serial.println(")");
    } else {
        Serial.println("真实AFE");
    }
    Serial.println("---");
}

void loop()
{
    esp_task_wdt_reset();   /* 喂狗: 主循环卡死 >10s 触发 panic 回溯 (诊断) */
    unsigned long currentTime = millis();

    /* 500Hz 固定采样节拍 (2026-08-14 esp_timer 硬件定时, 根治 336Hz 帧率漂移):
     * 每 2ms 定时器回调递增 s_sampleTick, loop 每节拍处理一帧; esp_timer
     * 创建失败时回退原 millis 软定时。 */
    bool sampleDue = false;
    if (s_sampleTimer != NULL) {
        if (s_sampleTick != s_lastSampleTick) {
            uint32_t missed = s_sampleTick - s_lastSampleTick;
            if (missed > 1) s_sampleTickDrops += (missed - 1);
            s_lastSampleTick = s_sampleTick;
            sampleDue = true;
        }
    } else if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
        lastSampleTime = currentTime;
        sampleDue = true;
    }

    if (sampleDue)
    {
        frameCount++;

        /* ---- 按键检测 (在帧处理前, 确保及时响应) ---- */
        if (isButtonPressed()) {
            toggleInputMode();
        }

        /* ---- LED 心跳指示 (慢闪省电) ---- */
        /* 有手机连接时：每 20 帧翻转一次 (12.5Hz) */
        /* 无连接时：每 250 帧翻转一次 (1Hz) */
        if (frameCount % 20 == 0 && isBLEConnected())
        {
            digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
        }
        else if (!isBLEConnected() && frameCount % 250 == 0)
        {
            digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
        }

        /* ======== 步骤1：获取原始样本 ======== */
        /* 根据当前输入模式选择数据源 */
        float noisySample;      /* 含 DC 偏置的原始信号 */
        float cleanSample;      /* 纯净/去偏置信号 */

        if (s_inputMode == SOURCE_SIMULATOR) {
            /* 模拟模式: ecg_simulator 生成 */
            noisySample = generateECGSample();     /* 含 1.65V DC */
            cleanSample = getCleanECGValue();      /* 无偏置, ±1.2V */
        } else if (s_inputMode == SOURCE_REPLAY) {
            /* 回放模式: MIT-BIH 数据库段 (2026-08-08) */
            noisySample = ecgReplayNextSample();   /* 真实心电, ±2V */
            cleanSample = noisySample;
        } else {
            /* 真实模式: ADC 采集 — 单次读取 (2026-08-14 根治 336Hz):
             * 原 afeHalReadSample + afeHalReadECG 双重读取 = 每帧 2×4=8 次
             * analogRead, 拖慢主循环致 ~336Hz; 改为单次读取, clean 由同一
             * 采样值减 dcBias 导出 (与 afeHalReadECG 语义等价, 且 noisy/clean
             * 严格同源更正确)。 */
            noisySample = afeHalReadSample();        /* 含 dcBias */
            cleanSample = noisySample - AFE_DC_BIAS; /* 已去偏置 */
        }

        /* ======== 步骤2：去除直流偏置，统一显示基准 ======== */
        /* 先去除偏置再滤波，避免 HPF 启动瞬态，且串口/BLE 三通道同基准 */
        float noisyNoDC = noisySample - DC_OFFSET_REMOVE;

        /* ======== 步骤2.5: 50Hz 梳状滤波 (v3.0 500Hz) ======== */
        /* 利用 500Hz/50Hz=10 的精确比, 10抽头滑动平均精确陷零50Hz */
        /* 注意: 只能在此处调一次! 不在 afe_hal 里调是因每帧调两次ADC */
        noisyNoDC = applyCombFilter(noisyNoDC);

        /* ======== 步骤2.6: 显示链 (2026-08-14) ======== */
        /* 用户验收反馈基线"斜+毛糙": 显示列改为中值基线去除 (0.2s/0.6s,
         * de Chazal 2004, 无高通相位失真) + LP40 平滑。仅供显示, 不影响
         * 心率/VF 链 (filteredSample) 与 AI 链 (applyFilterAI)。 */
        float displaySample = applyDisplayFilter(noisyNoDC);

        /* ======== 步骤3：数字滤波 ======== */
        float filteredSample = applyFilter(noisyNoDC);

        /* ======== 步骤3.5：心率检测 ======== */
        HR_Result hr = hrProcess(filteredSample);

        /* ======== T4-8 模块1+3: 心律安全 + AF 检测 (每帧, 消费 hr.rrInterval) ======== */
        /* 编译级集成; 报警通过调试串口输出, 不改变既有 CSV 格式 (输出集成待硬件阶段) */
        RS_Result rs = rsProcess(&hr);
        AF_Result af = afProcess(&hr);
        if (rs.asystole) {
            Serial.println("[SAFETY] ASYSTOLE detected (RR >= 4s)");
        }
        if (rs.bradycardia) {
            Serial.println("[SAFETY] BRADYCARDIA (30s HR < 40bpm)");
        }
        if (rs.tachycardia) {
            Serial.println("[SAFETY] TACHYCARDIA (30s HR > 180bpm)");
        }
        if (af.windowReady && af.label == 1) {
            Serial.println("[AF] AF suspected (CV/entropy window)");
        }

        /* T4-9 模块2: VF/VT 检测 — 特征链按 250Hz 标定 (5s 窗 1250 点)
         * 2026-08-14 修复: 原每帧 (~500Hz) 喂入致采样率错配 (5s 窗实为 ~2.8s,
         * 带通 8-20Hz 标定漂移, PROJECT_SUMMARY 已识别问题); 与 AI 链一致
         * 2:1 抽取后喂入 (250Hz 设计速率)。 */
        if ((frameCount % 2) == 0) {
            VF_Result vf = vfProcess(filteredSample);
            if (vf.vfAlarm) {
                Serial.println("[VF] VF/VT ALARM (2-window confirmed)");
            }
        }

        /* ======== 步骤3.6：AI 异常检测推理 (推送样本到 Core 0) ======== */
        /* 2026-08-13 (P0-2 解耦): AI 输入链独立于显示链 — AI 用 applyFilterAI
         * (HP 0.05 + LP 40, 与训练侧 exp7 复刻链一致), 显示链 filtered 用 HP 0.5
         * (基线稳定)。改显示链 HP 不再影响 AI 输入, 避免 train/deploy 失配。 */
        ai_inference_push(applyFilterAI(noisyNoDC));

        /* ---- 停搏/无信号检测: 当前秒内 filtered 极值跟踪 ---- */
        if (filteredSample < s_secMin) s_secMin = filteredSample;
        if (filteredSample > s_secMax) s_secMax = filteredSample;

        /* ======== 步骤3.7：ECG 录制 — 2:1 抽取 (500Hz→250Hz) 喂入 int16 样本 ======== */
        /* scale：±2V → ±16000 (int16 满量程 ±32767, 余量 ~2× headroom)
         *   replay 片段 ±2V, 模拟器 ±1.2V, AFE ~2Vpp; 统一 scale=8000.0
         *   例: 1.0V → 8000, -1.5V → -12000 */
        #define REC_SCALE_V_TO_INT16  8000.0f
        if ((frameCount % 2) == 0) {
            /* 2026-08-13 微调准备 (TH §40 B): 记录源 filteredSample → cleanSample
             * (去偏置原始), PC 侧按训练链 (梳状5抽头+HP0.05+LP40+因果0.5Hz@250Hz)
             * 预处理后用于 exp7c 真实数据微调, 消除域迁移误报 */
            ecgRecorderPushSample((int16_t)(cleanSample * REC_SCALE_V_TO_INT16));
        }

        /* ======== 步骤3.8：BLE 命令轮询 (消费 RxCallbacks 投递的队列, 非阻塞) ======== */
        {
            char bleCmd[32];
            char reply[128];
            while (bleCommandQueueTake(bleCmd, sizeof(bleCmd))) {
                if (parseRecorderCommand(bleCmd, reply, sizeof(reply), false)) {
                    if (strcmp(reply, "REC_LIST") == 0) {
                        /* REC_LIST: 多行输出, BLE 逐行发送 (ESP32 BLE 栈自动分片) */
                        char listBuf[512];
                        int n = ecgRecorderList(listBuf, sizeof(listBuf));
                        if (n > 0) {
                            sendBLEMessage("REC_LIST ok");
                            char* saveptr;
                            char* line = strtok_r(listBuf, "\n", &saveptr);
                            while (line) {
                                if (strlen(line) > 0) sendBLEMessage(line);
                                line = strtok_r(NULL, "\n", &saveptr);
                            }
                        } else {
                            sendBLEMessage("REC_LIST empty");
                        }
                    } else {
                        sendBLEMessage(reply);
                    }
                }
            }
        }

        /* ======== 步骤3.9：WiFi HTTP 请求轮询 (handleClient 空闲时 μs 级, 每迭代调用) ======== */
        ecgWifiProcess();

        /* ======== 步骤4：通过 BLE 发送 (默认 125Hz, 1帧/Notify; DIAG NOTIFY 可调) ======== */
        /* 每帧格式: clean,noisy,filtered,bpm,true_bpm,sqi,motion,abnormal,confidence;
         * 与串口 9 列一致 (2026-08-10 修复: 原仅 5 列, App 收不到 abnormal 致报警永不触发);
         * abnormal 取报警锁存值 (AI 新结果由 100Hz 块更新锁存), 锁存 5 秒与串口语义一致
         * 2026-08-10 修复2: 发送率 500Hz→250Hz (每 2 帧发 1 帧)。
         *   根因: 9 列解析修复后 App 每包解析全部帧, 数据率变 500Hz,
         *   而 App 缓冲/时间轴按 250Hz 设计 (timeWindow*250) → 速率错配致波形变形
 * 2026-08-14 修复3: 默认 250Hz→125Hz (s_bleNotifyDivider=4), App 时间轴同步 125Hz;
 *   250Hz notify≈11KB/s 在连接间隔退化到 30ms 时必丢帧致波形阶梯 (TH §57/§58)。 */
        if (frameCount % s_bleNotifyDivider == 0) {
            /* 2026-08-14 修复: true_bpm/motion 原硬编码 0 (串口路径正确, BLE 未同步)
             * → 与串口 9 列语义一致 */
            uint8_t trueBPM = (s_inputMode == SOURCE_SIMULATOR)
                              ? ecgSimulatorGetTrueBPM() : 0;
            int len = snprintf(s_bleBuf, sizeof(s_bleBuf),
                               "%.3f,%.3f,%.3f,%u,%u,%.2f,%u,%u,%.2f;",
                               cleanSample, noisyNoDC, displaySample,
                               hr.bpm, (unsigned)trueBPM, hr.sqi,
                               hr.motionActive ? 1u : 0u,
                               (s_alarmHold > 0) ? 1u : 0u, s_alarmHoldConf);
            if (len > 0) {
                sendBLEMessage(s_bleBuf);
            }
        }

        /* ======== 步骤5：串口输出（PC 绘图仪使用） ======== */
        /* 2026-08-08: 输出率 25Hz→100Hz (每 5 帧) — 提升波形分辨率
         * (25Hz 下 QRS 峰仅 2 采样点, 锯齿感强)。带宽: 55B×100Hz = 44kbps
         * < 115200 ✅。原 25Hz 注释: 降低 USB PHY 功耗 (100Hz 仍远低于上限) */
        /* 格式: clean,noisy,filtered,bpm,true_bpm,sqi,motion,abnormal_flag,confidence */
        if (frameCount % 5 == 0)
        {
            uint8_t trueBPM = (s_inputMode == SOURCE_SIMULATOR)
                              ? ecgSimulatorGetTrueBPM() : 0;
            /* 检查 AI 推理结果 */
            uint8_t abnormFlag = 0;
            float abnormConf = 0.0f;
            ai_result_t aiResult;
            if (ai_inference_pop_result(&aiResult)) {
                abnormFlag = aiResult.is_abnormal;
                abnormConf = aiResult.confidence;
                s_lastAiConf = aiResult.confidence;   /* 记录原始置信度 (诊断) */
                s_lastAiAbn  = aiResult.is_abnormal;
            }

            /* 报警锁存: 触发后 abnormal 列保持 1 共 5 秒 (防一闪而过, 2026-08-08) */
            if (abnormFlag) {
                s_alarmHold = ALARM_HOLD_OUTS;
                s_alarmHoldConf = abnormConf;
            }
            if (s_alarmHold > 0) {
                abnormFlag = 1;
                abnormConf = s_alarmHoldConf;
                s_alarmHold--;
            }

            /* 停搏/无信号合并: AI 未报但低电压直线持续 3 秒 → 强制报警 (2026-08-10)
             * 覆盖模型训练分布外的停搏/导联脱落场景 */
            if (s_flatline) {
                abnormFlag = 1;
                abnormConf = 0.99f;
                s_alarmHold = ALARM_HOLD_OUTS;   /* 维持锁存 (flatline 期间持续报警) */
                s_alarmHoldConf = 0.99f;
            }

            /* 导联脱落检测 (AD8232 LOD+/LOD-, 2026-08-13): 已禁用。
             * 用户硬件未接 LO+/LO- 线 (IO5/IO6 浮空), 浮空引脚随机读 HIGH 导致
             * 正常心电被随机强制 abnormal=1/conf=0.99 (污染 CSV 与录制位图)。
             * 导联脱落由 flatline 软件检测兜底; 将来接好 LOD 线后重新启用即可。
            if (digitalRead(AFE_LOD_P_PIN) == HIGH
                || digitalRead(AFE_LOD_N_PIN) == HIGH) {
                abnormFlag = 1;
                abnormConf = 0.99f;
                s_alarmHold = ALARM_HOLD_OUTS;
                s_alarmHoldConf = 0.99f;
            }
            */

            Serial.print(cleanSample, 4);
            Serial.print(",");
            Serial.print(noisyNoDC, 4);
            Serial.print(",");
            Serial.print(displaySample, 4);   /* 显示链: 中值基线去除+LP40 (2026-08-14,
                                                基线平直无相位失真; AI 链独立不受影响) */
            Serial.print(",");
            Serial.print(hr.bpm);
            Serial.print(",");
            Serial.print(trueBPM);
            Serial.print(",");
            Serial.print(hr.sqi, 3);
            Serial.print(",");
            Serial.print(hr.motionActive ? 1 : 0);
            Serial.print(",");
            Serial.print(abnormFlag);
            Serial.print(",");
            Serial.println(abnormConf, 3);
        }

        /* ======== 步骤6：实时削顶预警 (仅真实模式) ======== */
        if (s_inputMode == SOURCE_AFE_REAL && afeHalIsClipping()) {
            static unsigned long lastClipWarn = 0;
            if (currentTime - lastClipWarn > 2000) {  /* 每2秒打印一次 */
                lastClipWarn = currentTime;
                Serial.println("[警告] ADC 信号削顶! 请减小 AFE 增益");
            }
        }

        /* ======== 步骤7：温度监测 + BPM 状态打印 (每250帧≈1秒) ======== */
        if (frameCount % 250 == 0) {
            /* ---- ECG 录制: 标记当前秒异常状态 (CSV 同源锁存值)
               注: frameCount%250 在 500Hz 主循环下为 2Hz, 用 millis 秒去重
               保证真 1Hz, 否则 durationSec/位图 翻倍 ---- */
            static uint32_t s_lastRecSec = 0;
            uint32_t nowSec = (uint32_t)(millis() / 1000);
            if (nowSec != s_lastRecSec) {
                s_lastRecSec = nowSec;
                if (s_sampleTickDrops > 0) {
                    Serial.printf("[SAMPLE] 500Hz tick backlog dropped=%u (loop <500Hz)\n",
                                  (unsigned)s_sampleTickDrops);
                    s_sampleTickDrops = 0;
                }
                ecgRecorderSetSecondAbnormal(s_alarmHold > 0);

                /* ---- 停搏/无信号判定: 本秒峰峰值 < 阈值 连续 3 秒 ---- */
                float secPP = s_secMax - s_secMin;
                s_secMin = 1e9f;
                s_secMax = -1e9f;
                if (secPP < FLATLINE_PP_V) {
                    s_flatCount++;
                    if (s_flatCount >= FLATLINE_SECS) s_flatline = true;
                } else {
                    s_flatCount = 0;
                    s_flatline = false;
                }

                /* ---- 定时录制调度 (REC_SCHEDULE) ----
                 * 仅在调度激活且当前无录制时启动; 只自动停止调度自己启动的录制,
                 * 手动 REC_START 的会话由手动 REC_STOP 结束 (s_schedActiveRec=false) */
                if (s_schedInterval > 0) {
                    if (!ecgRecorderIsRecording() && nowSec >= s_schedNextStart) {
                        if (ecgRecorderStart()) {
                            s_schedActiveRec = true;
                            s_schedStartSec = nowSec;
                            Serial.println("[SCHED] 定时录制开始");
                        }
                    } else if (ecgRecorderIsRecording() && s_schedActiveRec &&
                               (nowSec - s_schedStartSec) >= s_schedDuration) {
                        ecgRecorderStop();
                        s_schedActiveRec = false;
                        s_schedNextStart = nowSec + s_schedInterval;
                        Serial.println("[SCHED] 定时录制结束");
                    }
                }
            }

            /* ---- 温度监测 ---- */
            ThermalState ts = thermalUpdate();

            /* 温度恢复正常后恢复 240MHz */
            static bool s_wasOverheated = false;
            if (ts.avg < 55.0f && s_wasOverheated) {
                s_wasOverheated = false;
                setCpuFrequencyMhz(240);
                Serial.print("[温度] ✅ 温度已降至 ");
                Serial.print(ts.avg, 1);
                Serial.println("°C, 恢复 240MHz");
            }

            if (ts.alertLevel >= THERMAL_WARN) {
                Serial.print("[温度] ⚠ ");
                Serial.print(ts.avg, 1);
                Serial.print("°C | ");
                Serial.println(thermalGetAlertString());
            }
            if (ts.alertLevel >= THERMAL_CRITICAL) {
                s_wasOverheated = true;
                /* 降频至 60MHz (BLE 在此频率仍可正常工作) */
                Serial.println("[温度] 🔥 过热! 自动降频至 60MHz...");
                setCpuFrequencyMhz(60);
            }
            /* 每 30 秒打印一次详细信息 */
            if (frameCount % (250 * 30) == 0) {
                thermalPrintStatus();
            }
            /* 运动警告 */
            if (hr.motionActive) {
                static bool wasInMotion = false;
                if (!wasInMotion) {
                    Serial.println("[运动] ⚠ 检测到运动干扰, 心率冻结中...");
                    wasInMotion = true;
                }
            } else {
                /* 隐式重置 wasInMotion (静态局部变量保持) */
            }

            if (hr.beatCount > 0) {
                Serial.print("[心率] ");
                /* 2026-08-14: 按 bpm 判定显示 (原按 confidence≥0.3 因 HOLD 衰减
                 * 在 1Hz 采样下闪烁 "检测/学习中", 且 "需 5" 文案在 beats>5 时误导) */
                if (hr.bpm > 0) {
                    Serial.print("检测 ");
                    Serial.print(hr.bpm);
                    Serial.print(" BPM");
                    /* 模拟模式下对比真实心率 */
                    if (s_inputMode == SOURCE_SIMULATOR) {
                        Serial.print(" | 真实 ");
                        Serial.print(ecgSimulatorGetTrueBPM());
                        Serial.print(" BPM");
                    }
                    Serial.print(" | 心拍: ");
                    Serial.print(hr.beatCount);
                    Serial.print(" | RR: ");
                    Serial.print(hr.rrInterval * 1000.0f, 1);
                    Serial.print(" ms");
                    Serial.print(" | SQI: ");
                    Serial.print(hr.sqi, 2);
                    if (hr.motionActive) {
                        Serial.print(" [运动中]");
                    }
                    Serial.print(" | 置信度: ");
                    Serial.println(hr.confidence, 2);
                } else {
                    Serial.print("学习中... (心拍: ");
                    Serial.print(hr.beatCount);
                    Serial.print(" / 需 5) | SQI: ");
                    Serial.println(hr.sqi, 2);
                }
            } else {
                Serial.print("[心率] 等待心拍... | SQI: ");
                Serial.println(hr.sqi, 2);
            }
        }

        /* ======== 串口指令处理 ======== */
        if (Serial.available() > 0)
        {
            char cmd = Serial.read();

            /* ---- 多字符命令行累加器 (REC_* 指令, BLE 同解析器) ---- */
            if (cmd == '\n' || cmd == '\r') {
                if (s_serialLineLen > 1) {
                    s_serialLine[s_serialLineLen] = '\0';
                    char reply[128];
                    if (parseRecorderCommand(s_serialLine, reply, sizeof(reply), true)) {
                        if (strcmp(reply, "REC_LIST") == 0) {
                            Serial.println("REC_LIST ok");
                            char listBuf[512];
                            int n = ecgRecorderList(listBuf, sizeof(listBuf));
                            if (n > 0) {
                                Serial.print(listBuf);
                            } else {
                                Serial.println("(empty)");
                            }
                        } else {
                            Serial.println(reply);
                        }
                    }
                }
                s_serialLineLen = 0;
            } else if (s_serialLineLen < (int)(sizeof(s_serialLine) - 1)) {
                s_serialLine[s_serialLineLen++] = cmd;
            }

            /* ---- 单字符快捷指令 (原有, 保持不变) ---- */
            switch (cmd)
            {
                case 'r':
                case 'R':
                    filterReset();
                    Serial.println("[调试] 滤波器已重置");
                    break;

                case 's':
                case 'S':
                    ecgSimulatorReset();
                    Serial.println("[调试] 信号发生器已重置");
                    break;

                case 'm':
                case 'M':
                    toggleInputMode();
                    break;

                case 'n':
                case 'N':
                    if (s_inputMode == SOURCE_REPLAY) {
                        ecgReplaySetSegment(0);
                        Serial.println(">>> 回放: 正常段 (MIT-BIH 100) <<<");
                    } else {
                        Serial.println("[提示] 当前非回放模式 ('m' 切换)");
                    }
                    break;

                case 'e':
                case 'E':
                    if (s_inputMode == SOURCE_REPLAY) {
                        ecgReplaySetSegment(1);
                        Serial.println(">>> 回放: 异常段 (MIT-BIH 106, VEB) <<<");
                    } else {
                        Serial.println("[提示] 当前非回放模式 ('m' 切换)");
                    }
                    break;

                case 't':
                case 'T':
                    thermalPrintStatus();
                    break;

                case 'a':
                case 'A':
                    {
                        uint32_t ti, ta, al;
                        ai_inference_stats(&ti, &ta, &al);
                        Serial.print("[AI] 统计 | 总次数: ");
                        Serial.print(ti);
                        Serial.print(" | 异常: ");
                        Serial.print(ta);
                        Serial.print(" | 平均延迟: ");
                        Serial.print(al);
                        Serial.print(" us | 开关: ");
                        Serial.print(ai_inference_is_enabled() ? "开" : "关");
                        Serial.print(" | 最近置信度: ");
                        Serial.print(s_lastAiConf, 3);
                        Serial.print(" (判异常=");
                        Serial.print(s_lastAiAbn);
                        Serial.println(")");
                    }
                    break;

                case 'c':
                case 'C':
                    Serial.print("[系统] CPU 当前频率: ");
                    Serial.print(getCpuFrequencyMhz());
                    Serial.println(" MHz");
                    break;

                case 'l':
                case 'L':
                    {
                        int lop = digitalRead(AFE_LOD_P_PIN);
                        int lon = digitalRead(AFE_LOD_N_PIN);
                        Serial.print("[LOD] IO5(LO+)=");
                        Serial.print(lop == HIGH ? "HIGH(脱落)" : "LOW(连接)");
                        Serial.print("  IO6(LO-)=");
                        Serial.println(lon == HIGH ? "HIGH(脱落)" : "LOW(连接)");
                    }
                    break;

                default:
                    break;
            }
        }
    }
}
