#include <Arduino.h>
#include "filter/filter.h"
#include "bluetooth/ble.h"
#include "signal_generator/ecg_simulator.h"
#include "adc_afe/afe_hal.h"
#include "heartrate/heartrate.h"

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
 *   [模拟发生器 / 真实AFE] → 三级数字滤波 → BLE发送(NUS) + 串口
 *
 * == 信号电平说明 ==
 *   - clean  ：纯净心电波形 / 去偏置ADC信号, ±1.2V
 *   - noisy  ：含噪声原始信号 (偏置已去除), 与clean同基准
 *   - filtered：数字滤波后的信号
 *
 * 三通道同基准显示，方便手机 App 叠加对比
 */

/* ======================== 常量定义 ======================== */
#define SAMPLE_INTERVAL_MS  4   /* 250Hz 采样间隔 */
#define DC_OFFSET_REMOVE    1.65f  /* 去除 ADC 直流偏置，统一显示基准 */

/* ======================== 按键配置 ======================== */
#define BUTTON_PIN          GPIO_NUM_0   /* 板载 BOOT 按钮 (低电平有效) */
#define BUTTON_DEBOUNCE_MS  200          /* 消抖延时 (毫秒) */

/* ======================== AFE 引脚配置 ======================== */
/* ★ 重要: 根据你的 PCB 改这里 ★ */
#define AFE_ADC_PIN         GPIO_NUM_4   /* AFE 输出接在哪个 GPIO */
#define AFE_DC_BIAS         1.65f        /* PCB 输出的直流偏置 (V) */
#define AFE_VREF            3.3f         /* ESP32 ADC 参考电压 */
#define AFE_OVERSAMPLE      8            /* 过采样次数 */

/* ======================== 输入模式枚举 ======================== */
typedef enum {
    SOURCE_SIMULATOR = 0,    /**< 模拟发生器模式 (默认, 无硬件也可运行) */
    SOURCE_AFE_REAL  = 1     /**< 真实 AFE 采集模式 */
} InputSource;

/* ======================== 全局变量 ======================== */
static unsigned long lastSampleTime = 0;
static unsigned long frameCount = 0;

/* 输入模式与按键状态 */
static InputSource  s_inputMode     = SOURCE_SIMULATOR;
static unsigned long s_lastBtnPress = 0;
static bool         s_btnLastState  = HIGH;   /* 上拉, 默认高电平 */

void setup()
{
    Serial.begin(115200);
    delay(100);
    Serial.println();
    Serial.println("========================================");
    Serial.println(" ESP32-ECG-MONITOR 心电采集系统 v1.0");
    Serial.println(" 广播名称: ESP32-ECG");
    Serial.println(" 模式：软件验证模式（无外部电路）");
    Serial.println(" 串口格式：clean,noisy,filtered,bpm");
    Serial.println(" 板上心率:  简化 Pan-Tompkins QRS 检测");
    Serial.println("========================================");

    /* 初始化板载 LED */
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    /* 初始化按键 (内部上拉, 按下为 LOW) */
    pinMode(BUTTON_PIN, INPUT_PULLUP);

    /* 初始化各模块 */
    ecgSimulatorInit();
    Serial.println("[系统] 心电信号生成器已初始化");

    /* 初始化真实 AFE 模块 (即使当前是模拟模式, 也准备好) */
    AFE_HAL_Config afeCfg = {
        .adcPin     = AFE_ADC_PIN,
        .dcBias     = AFE_DC_BIAS,
        .vRef       = AFE_VREF,
        .oversample = AFE_OVERSAMPLE,
        .enableCal  = true
    };
    afeHalInit(&afeCfg);

    filterInit();
    Serial.println("[系统] 数字滤波器已初始化");

    hrInit();
    Serial.println("[系统] 心率监测器已启动");
    Serial.print("[系统] 模拟器真实心率: ");
    Serial.print(ecgSimulatorGetTrueBPM());
    Serial.println(" BPM");

    initBLE();

    /* 默认启动模式: 无外接 AFE 时用模拟器, 方便调试 */
    s_inputMode = SOURCE_SIMULATOR;
    Serial.println("[系统] 当前输入: 模拟发生器");
    Serial.println("[系统] 按 BOOT 键或发 'm' 切换真实/模拟输入");

    lastSampleTime = millis();
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
        s_inputMode = SOURCE_AFE_REAL;
        Serial.println("\n>>> 切换至: 真实 AFE 采集模式 <<<");
    } else {
        s_inputMode = SOURCE_SIMULATOR;
        ecgSimulatorReset();  /* 重置发生器, 从新周期开始 */
        Serial.println("\n>>> 切换至: 模拟发生器模式 <<<");
    }

    /* 切换后复位滤波器与心率检测器, 消除瞬态 */
    filterReset();
    hrReset();

    /* LED 闪烁 3 次指示切换 */
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_BUILTIN, HIGH);
        delay(50);
        digitalWrite(LED_BUILTIN, LOW);
        delay(50);
    }

    Serial.print("[系统] 当前输入模式: ");
    Serial.println(s_inputMode == SOURCE_SIMULATOR ? "模拟" : "真实AFE");
    Serial.println("---");
}

void loop()
{
    unsigned long currentTime = millis();

    /* 精确定时 4ms 采样间隔 */
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS)
    {
        lastSampleTime = currentTime;
        frameCount++;

        /* ---- 按键检测 (在帧处理前, 确保及时响应) ---- */
        if (isButtonPressed()) {
            toggleInputMode();
        }

        /* ---- LED 心跳指示 ---- */
        /* 有手机连接时：每个样本翻转一次（快闪） */
        /* 无连接时：每 25 帧翻转一次（慢闪，指示运行中） */
        if (isBLEConnected())
        {
            digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
        }
        else if (frameCount % 25 == 0)
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
        } else {
            /* 真实模式: ADC 采集 */
            noisySample = afeHalReadSample();       /* 含 dcBias */
            cleanSample = afeHalReadECG();          /* 已去偏置 */
        }

        /* ======== 步骤2：数字滤波 ======== */
        float filteredSample = applyFilter(noisySample);

        /* ======== 步骤2.5：心率检测 ======== */
        HR_Result hr = hrProcess(filteredSample);

        /* ======== 步骤3：去除直流偏置，统一显示基准 ======== */
        /* 手机 App 三通道叠加显示，需同一电平基准 */
        float noisyNoDC = noisySample - DC_OFFSET_REMOVE;

        /* ======== 步骤4：通过 BLE 发送 ======== */
        /* 格式：clean,noisy_no_dc,filtered,bpm */
        char csvLine[48];
        snprintf(csvLine, sizeof(csvLine),
                 "%.3f,%.3f,%.3f,%u\r\n",
                 cleanSample, noisyNoDC, filteredSample, hr.bpm);
        sendBLEMessage(csvLine);

        /* ======== 步骤5：串口输出（PC 绘图仪使用） ======== */
        /* 格式: clean,noisy,filtered,bpm,true_bpm */
        /* 串口保留原始含偏置的 noisy，方便调试 */
        uint8_t trueBPM = (s_inputMode == SOURCE_SIMULATOR)
                          ? ecgSimulatorGetTrueBPM() : 0;
        Serial.print(cleanSample, 4);
        Serial.print(",");
        Serial.print(noisySample, 4);  /* 保留原始含 DC 的 noisy */
        Serial.print(",");
        Serial.print(filteredSample, 4);
        Serial.print(",");
        Serial.print(hr.bpm);
        Serial.print(",");
        Serial.println(trueBPM);

        /* ======== 步骤6：实时削顶预警 (仅真实模式) ======== */
        if (s_inputMode == SOURCE_AFE_REAL && afeHalIsClipping()) {
            static unsigned long lastClipWarn = 0;
            if (currentTime - lastClipWarn > 2000) {  /* 每2秒打印一次 */
                lastClipWarn = currentTime;
                Serial.println("[警告] ADC 信号削顶! 请减小 AFE 增益");
            }
        }

        /* ======== 步骤7：BPM 状态打印 (每250帧≈1秒) ======== */
        if (frameCount % 250 == 0 && hr.beatCount > 0) {
            Serial.print("[心率] ");
            if (hr.confidence >= 0.3f) {
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
                Serial.print(" | 置信度: ");
                Serial.println(hr.confidence, 2);
            } else {
                Serial.print("学习中... (心拍: ");
                Serial.print(hr.beatCount);
                Serial.println(" / 需 5)");
            }
        }

        /* ======== 串口指令处理 ======== */
        if (Serial.available() > 0)
        {
            char cmd = Serial.read();
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

                default:
                    break;
            }
        }
    }
}
