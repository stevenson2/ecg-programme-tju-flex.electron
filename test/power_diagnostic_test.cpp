/**
 * @file power_diagnostic_test.cpp
 * @brief ESP32-S3-SUPERMINI 功耗诊断测试
 *
 * 逐项启停子系统，配合电流表定位热源。
 * 使用方法：
 *   1. 将电流表串入 USB 5V 供电回路 (或 3.3V 回路)
 *   2. 将本文件复制到 test/ 目录
 *   3. 验证各阶段电流和温度变化
 *
 * 测试序列（每阶段 30 秒）：
 *   阶段 0: 空闲状态 (WiFi/BLE 关闭, CPU 40MHz)
 *   阶段 1: CPU 80MHz
 *   阶段 2: CPU 160MHz
 *   阶段 3: CPU 240MHz
 *   阶段 4: BLE 广播开启 (+0dBm)
 *   阶段 5: BLE 广播 (+3dBm)
 *   阶段 6: BLE 广播 (+9dBm)
 *   阶段 7: BLE 已连接并 Notify
 *   阶段 8: LED 全亮
 *   阶段 9: 恢复空闲
 *
 * 串口输出 format:
 *   [DIAG] phase=0 time=5 temp=45.2 current=XXXmA cpu=40 ble=off led=off
 *   [DIAG] phase=0 time=10 temp=45.5 current=XXXmA cpu=40 ble=off led=off
 */

#include <Arduino.h>
#include <esp_bt.h>
#include <esp_bt_main.h>
#include <esp_gap_ble_api.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEScan.h>
#include <BLEUtils.h>
#include <BLEAdvertising.h>
#include "thermal/thermal.h"

/* ======================== 配置 ======================== */
#define PHASE_DURATION_MS   30000   /* 每阶段持续 30 秒 */
#define REPORT_INTERVAL_MS  5000    /* 每 5 秒报告一次 */
#define TEMP_WAIT_MS        2000    /* 切换后等待温度稳定 */

/* 全局变量 */
static uint8_t  s_phase = 0;
static bool     s_testRunning = true;
static unsigned long s_phaseStart = 0;

/* BLE 引用声明 (来自 main 代码) */
extern bool isBLEConnected(void);
extern void initBLE(void);
extern void sendBLEMessage(const char* msg);

/* LED 控制 */
#define LED_PIN   GPIO_NUM_48

static const char* phaseNames[] = {
    "空闲 40MHz(no BT/WiFi)",
    "CPU 80MHz",
    "CPU 160MHz",
    "CPU 240MHz",
    "BLE 广播 +0dBm",
    "BLE 广播 +3dBm",
    "BLE 广播 +9dBm",
    "BLE Notify 数据",
    "LED 全亮",
    "恢复空闲"
};

void printDiagnosticReport(uint8_t phase, unsigned long elapsed)
{
    Serial.print("[DIAG] phase=");
    Serial.print(phase);
    Serial.print(" phase_name=");
    Serial.print(phaseNames[phase]);
    Serial.print(" time_s=");
    Serial.print(elapsed / 1000);
    Serial.print(" temp=");
    Serial.print(temperatureRead(), 1);
    Serial.print(" cpu=");
    Serial.print(getCpuFrequencyMhz());
    Serial.print(" ble=");
    if (isBLEConnected()) {
        Serial.print("connected");
    } else {
        /* 粗略判断 BLE 是否初始化 */
        Serial.print("advertising");
    }
    Serial.print(" led=");
    Serial.print(digitalRead(LED_PIN) == LOW ? "on" : "off");
    Serial.println();
}

void runPhase(uint8_t phase)
{
    Serial.println();
    Serial.println("==================================================");
    Serial.print("[DIAG] === 阶段 ");
    Serial.print(phase);
    Serial.print(": ");
    Serial.print(phaseNames[phase]);
    Serial.println(" ===");
    Serial.println(" 请记录此时电流表读数");
    Serial.println("==================================================");

    s_phaseStart = millis();

    while (millis() - s_phaseStart < PHASE_DURATION_MS + TEMP_WAIT_MS) {
        unsigned long elapsed = millis() - s_phaseStart;

        /* 等待 TEMP_WAIT_MS 让温度稳定 */
        if (elapsed < TEMP_WAIT_MS) {
            delay(100);
            continue;
        }

        /* 每 5 秒输出一次诊断报告 */
        if ((elapsed - TEMP_WAIT_MS) % REPORT_INTERVAL_MS < 100) {
            printDiagnosticReport(phase, elapsed);
        }

        delay(100);
    }
}

void setup()
{
    Serial.begin(460800);
    delay(2000);  /* 等待串口就绪 */
    Serial.println();
    Serial.println("============================================");
    Serial.println(" ESP32-S3-SUPERMINI 功耗诊断测试 v1.0");
    Serial.println(" 请连接电流表，观察各阶段电流和温度变化");
    Serial.println("============================================");
    Serial.println();
    Serial.println("按 0-9 跳转到对应阶段，按 'q' 退出");
    Serial.println();

    /* 初始化温度 */
    thermalInit();

    /* 初始化 LED */
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, HIGH);  /* 初始熄灭 */

    /* 阶段 0: 空闲 (WiFi/BLE 关闭) */
    // 正常空闲状态
}

void loop()
{
    if (!s_testRunning) {
        delay(1000);
        return;
    }

    /* 检查串口命令 */
    if (Serial.available() > 0) {
        char cmd = Serial.read();
        if (cmd >= '0' && cmd <= '9') {
            s_phase = cmd - '0';
            Serial.print("[DIAG] 手动跳转到阶段 ");
            Serial.println(s_phase);
        } else if (cmd == 'q' || cmd == 'Q') {
            s_testRunning = false;
            Serial.println("[DIAG] 测试已停止");
            return;
        }
    }

    /* ==== 执行各阶段 ==== */

    switch (s_phase) {
        case 0:  /* 空闲 40MHz (baseline) */
            setCpuFrequencyMhz(40);
            digitalWrite(LED_PIN, HIGH);  /* LED 灭 */
            // BLE/WiFi 未初始化, 最小功耗
            runPhase(0);
            s_phase = 1;
            break;

        case 1:  /* CPU 80MHz */
            setCpuFrequencyMhz(80);
            runPhase(1);
            s_phase = 2;
            break;

        case 2:  /* CPU 160MHz */
            setCpuFrequencyMhz(160);
            runPhase(2);
            s_phase = 3;
            break;

        case 3:  /* CPU 240MHz */
            setCpuFrequencyMhz(240);
            runPhase(3);
            s_phase = 4;
            break;

        case 4:  /* BLE 广播 +0dBm */
            // 初始化 BLE 但设为最低功率
            // 注意: 实际实现需引用 ble.cpp 的函数
            esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_ADV, ESP_PWR_LVL_N0);
            esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_CONN_HDL0, ESP_PWR_LVL_N0);
            initBLE();
            runPhase(4);
            s_phase = 5;
            break;

        case 5:  /* BLE 广播 +3dBm */
            esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_ADV, ESP_PWR_LVL_P3);
            esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_CONN_HDL0, ESP_PWR_LVL_P3);
            runPhase(5);
            s_phase = 6;
            break;

        case 6:  /* BLE 广播 +9dBm */
            esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_ADV, ESP_PWR_LVL_P9);
            esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_CONN_HDL0, ESP_PWR_LVL_P9);
            runPhase(6);
            s_phase = 7;
            break;

        case 7:  /* BLE Notify 数据 */
            // 发送实时 ECG 数据
            {
                unsigned long notifyStart = millis();
                while (millis() - notifyStart < PHASE_DURATION_MS + TEMP_WAIT_MS) {
                    unsigned long elapsed = millis() - notifyStart;
                    if (elapsed < TEMP_WAIT_MS) { delay(10); continue; }
                    if (elapsed % REPORT_INTERVAL_MS < 10) {
                        printDiagnosticReport(7, elapsed);
                    }
                    if (isBLEConnected()) {
                        sendBLEMessage("DIAG:0.000,0.000,0.000,0,0.00\r\n");
                    }
                    delay(4);  /* 250Hz 模拟 ECG 数据 */
                }
            }
            s_phase = 8;
            break;

        case 8:  /* LED 全亮 */
            digitalWrite(LED_PIN, LOW);  /* 共阳极: LOW=亮 */
            runPhase(8);
            digitalWrite(LED_PIN, HIGH); /* 熄灭 */
            s_phase = 9;
            break;

        case 9:  /* 恢复空闲 */
            setCpuFrequencyMhz(40);
            esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_ADV, ESP_PWR_LVL_N0);
            digitalWrite(LED_PIN, HIGH);
            runPhase(9);
            Serial.println();
            Serial.println("============================================");
            Serial.println(" [DIAG] 诊断测试完成！");
            Serial.println(" 请查看上述各阶段电流和温度数据");
            Serial.println("============================================");
            s_testRunning = false;
            break;

        default:
            s_testRunning = false;
            break;
    }
}