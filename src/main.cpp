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
 * 3. BLE 将滤波后的数据发送到连接的客户端
 * 4. 串口输出三路数据（无干扰,带干扰,滤波后），供 PC 绘图仪使用
 *
 * 采样率：250Hz（每 4ms 一个样本）
 */

/* 全局变量 */
static unsigned long lastSampleTime = 0;
const unsigned long sampleInterval = 4;

void setup()
{
    Serial.begin(115200);
    delay(100);
    Serial.println();
    Serial.println("========================================");
    Serial.println(" ESP32-ECG-MONITOR 心电采集系统 v1.0");
    Serial.println(" 模式：软件验证模式（无外部电路）");
    Serial.println(" 串口格式：无干扰信号,带干扰信号,滤波后信号");
    Serial.println("========================================");

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

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

    if (currentTime - lastSampleTime >= sampleInterval)
    {
        lastSampleTime = currentTime;

        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));

        /* 步骤1：生成带噪声的模拟心电样本 */
        float noisySample = generateECGSample();

        /* 步骤2：获取无干扰纯净心电信号 */
        float cleanSample = getCleanECGValue();

        /* 步骤3：对带噪声信号进行数字滤波 */
        float filteredSample = applyFilter(noisySample);

        /* 步骤4：通过 BLE 发送滤波后的数据 */
        sendECGData(filteredSample);

        /* 步骤5：串口输出三路数据（格式：无干扰,带干扰,滤波后） */
        Serial.print(cleanSample, 4);
        Serial.print(",");
        Serial.print(noisySample, 4);
        Serial.print(",");
        Serial.println(filteredSample, 4);

        /* 串口指令处理 */
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
