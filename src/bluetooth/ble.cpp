#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <esp_bt.h>
#include <esp_bt_main.h>
#include <esp_gap_ble_api.h>
#include "bluetooth/ble.h"

/**
 * @file ble.cpp
 * @brief 心电数据 BLE NUS UART 透传模块实现
 *
 * Nordic UART Service (NUS) 标准定义：
 * - Service UUID: 6E400001-B5A3-F393-E0A9-E50E24DCCA9E
 * - TX Char (Notify): 6E400002-B5A3-F393-E0A9-E50E24DCCA9E
 * - RX Char (Write):  6E400003-B5A3-F393-E0A9-E50E24DCCA9E
 */

/* ======================== NUS UUID 定义 ======================== */
#define NUS_SERVICE_UUID    "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define NUS_TX_UUID         "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define NUS_RX_UUID         "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

#define DEVICE_NAME         "ESP32-ECG"

/* 广播间隔参数（单位：0.625ms） */
#define ADV_INTERVAL_MIN    80    /* 50ms  */
#define ADV_INTERVAL_MAX    160   /* 100ms */

/* ======================== 全局对象 ======================== */

static BLEServer            *pServer     = NULL;
static BLECharacteristic    *pTxChar     = NULL;
static bool                  connected   = false;

/* ======================== 连接回调 ======================== */

class ServerCallbacks : public BLEServerCallbacks
{
    void onConnect(BLEServer* srv) override
    {
        connected = true;
        Serial.println("[BLE] 手机已连接");
    }

    void onDisconnect(BLEServer* srv) override
    {
        connected = false;
        Serial.println("[BLE] 手机已断开，重启广播");
        srv->getAdvertising()->start();
    }
};

/* ======================== RX 回调（接收手机指令） ======================== */

class RxCallbacks : public BLECharacteristicCallbacks
{
    void onWrite(BLECharacteristic* pChar) override
    {
        std::string rx = pChar->getValue();
        if (rx.length() > 0)
        {
            Serial.print("[BLE] 收到指令: ");
            Serial.println(rx.c_str());
        }
    }
};

/* ======================== 公共接口 ======================== */

void initBLE(void)
{
    /* 初始化 BLE 设备 */
    BLEDevice::init(DEVICE_NAME);

    /* 设置 BLE 发射功率（ESP32-S3 默认为 0dBm，这里设为最高 +8dBm）*/
    esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_ADV, ESP_PWR_LVL_P9);  /* +9dBm */
    esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_CONN_HDL0, ESP_PWR_LVL_P9);

    /* 创建 BLE 服务端 */
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());

    /* 创建 NUS 服务 */
    BLEService *pSvc = pServer->createService(NUS_SERVICE_UUID);

    /* TX 特征值：发送数据给手机（Notify） */
    pTxChar = pSvc->createCharacteristic(
        NUS_TX_UUID,
        BLECharacteristic::PROPERTY_NOTIFY
    );
    pTxChar->addDescriptor(new BLE2902());

    /* RX 特征值：接收手机指令（Write No Response） */
    BLECharacteristic *pRxChar = pSvc->createCharacteristic(
        NUS_RX_UUID,
        BLECharacteristic::PROPERTY_WRITE_NR
    );
    pRxChar->setCallbacks(new RxCallbacks());

    /* 启动服务 */
    pSvc->start();

    /* ★★★ 关键：使用 pServer->getAdvertising() 而非 BLEDevice::getAdvertising() ★★★
     * 确保广播数据包含完整的服务 UUID，提高手机端扫描成功率 */
    BLEAdvertising *pAdv = pServer->getAdvertising();

    /* 明确的广播类型：可连接、无定向广播 */
    pAdv->setAdvertisementType(ADV_TYPE_IND);

    /* 设置广播间隔（50ms ~ 100ms），快速被发现 */
    pAdv->setMinInterval(ADV_INTERVAL_MIN);
    pAdv->setMaxInterval(ADV_INTERVAL_MAX);

    /* 在广播包中包含服务 UUID（手机扫描时可直接识别） */
    pAdv->addServiceUUID(NUS_SERVICE_UUID);

    /* 开启扫描响应（包含设备名称等信息） */
    pAdv->setScanResponse(true);

    /* 设置首选连接参数 */
    pAdv->setMinPreferred(0x06);  /* 最小连接间隔 7.5ms */
    pAdv->setMaxPreferred(0x12);  /* 最大连接间隔 22.5ms */

    /* 启动广播 */
    pAdv->start();

    Serial.println("[BLE] 设备名称: " DEVICE_NAME);
    Serial.println("[BLE] Service UUID: " NUS_SERVICE_UUID);
    Serial.println("[BLE] 广播已启动，等待手机连接...");
}

void sendBLEMessage(const char* message)
{
    if (!connected) return;
    if (!message || strlen(message) == 0) return;

    pTxChar->setValue((uint8_t*)message, strlen(message));
    pTxChar->notify();
}

bool isBLEConnected(void)
{
    return connected;
}
