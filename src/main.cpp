#include <Arduino.h>
#include "filter/filter.h"
#include "bluetooth/ble.h"
#include "signal_generator/ecg_simulator.h"

/**
 * @file main.cpp
 * @brief ESP32-S3 心电采集系统 - 主程序入口
 *
 * 系统工作流程（软件验证模式）：
 * 1. ecg_simulator 生成带噪声的模拟心电信号
 * 2. filter 对信号进行带通滤波 + 50Hz 陷波
 * 3. BLE 将三路数据（clean/noisy/filtered）发送到手机 App
 * 4. 串口输出供 PC 绘图仪使用
 *
 * 采样率：250Hz（每 4ms 一个样本）
 *
 * 信号电平说明：
 * - clean  ：纯净心电波形，范围约 -1.2V ~ +1.2V
 * - noisy  ：含噪声的原始ADC值，已去除1.65V直流偏置，与clean同基准
 * - filtered：数字滤波后的信号，范围与clean一致
 * 三通道同基准显示，方便手机 App 叠加对比
 */

/* ======================== 常量定义 ======================== */
#define SAMPLE_INTERVAL_MS  4   /* 250Hz 采样间隔 */
#define DC_OFFSET_REMOVE    1.65f  /* 去除 ADC 直流偏置，统一显示基准 */

/* ======================== 全局变量 ======================== */
static unsigned long lastSampleTime = 0;
static unsigned long frameCount = 0;

void setup()
{
    Serial.begin(115200);
    delay(100);
    Serial.println();
    Serial.println("========================================");
    Serial.println(" ESP32-ECG-MONITOR 心电采集系统 v1.0");
    Serial.println(" 广播名称: ESP32-ECG");
    Serial.println(" 模式：软件验证模式（无外部电路）");
    Serial.println(" 串口格式：clean,noisy,filtered");
    Serial.println("========================================");

    /* 初始化板载 LED */
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    /* 初始化各模块 */
    ecgSimulatorInit();
    Serial.println("[系统] 心电信号生成器已初始化");

    filterInit();
    Serial.println("[系统] 数字滤波器已初始化");

    initBLE();

    lastSampleTime = millis();
    Serial.println("[系统] 系统启动完成，开始采集...");
}

void loop()
{
    unsigned long currentTime = millis();

    /* 精确定时 4ms 采样间隔 */
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS)
    {
        lastSampleTime = currentTime;
        frameCount++;

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

        /* ======== 步骤1：生成带噪声的模拟心电样本 ======== */
        float noisySample = generateECGSample();  /* 含 1.65V 直流偏置 */

        /* ======== 步骤2：获取纯净心电信号 ======== */
        float cleanSample = getCleanECGValue();   /* 无偏置，±1.2V */

        /* ======== 步骤3：数字滤波 ======== */
        float filteredSample = applyFilter(noisySample);

        /* ======== 步骤4：去除直流偏置，统一显示基准 ======== */
        /* 手机 App 三通道叠加显示，需同一电平基准 */
        float noisyNoDC = noisySample - DC_OFFSET_REMOVE;

        /* ======== 步骤5：通过 BLE 发送 ======== */
        /* 格式：clean,noisy_no_dc,filtered */
        char csvLine[32];
        snprintf(csvLine, sizeof(csvLine),
                 "%.3f,%.3f,%.3f\r\n",
                 cleanSample, noisyNoDC, filteredSample);
        sendBLEMessage(csvLine);

        /* ======== 步骤6：串口输出（PC 绘图仪使用） ======== */
        /* 串口保留原始含偏置的 noisy，方便调试 */
        Serial.print(cleanSample, 4);
        Serial.print(",");
        Serial.print(noisySample, 4);  /* 保留原始含 DC 的 noisy */
        Serial.print(",");
        Serial.println(filteredSample, 4);

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

                default:
                    break;
            }
        }
    }
}
