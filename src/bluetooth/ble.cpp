#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
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
    BLEDevice::init(DEVICE_NAME);

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

    /* RX 特征值：接收手机指令（Write） */
    BLECharacteristic *pRxChar = pSvc->createCharacteristic(
        NUS_RX_UUID,
        BLECharacteristic::PROPERTY_WRITE_NR
    );
    pRxChar->setCallbacks(new RxCallbacks());

    /* 启动服务 */
    pSvc->start();

    /* 配置广播 */
    BLEAdvertising *pAdv = BLEDevice::getAdvertising();
    pAdv->addServiceUUID(NUS_SERVICE_UUID);
    pAdv->setScanResponse(true);
    pAdv->setMinPreferred(0x06);
    pAdv->setMinPreferred(0x12);
    BLEDevice::startAdvertising();

    Serial.println("[BLE] 设备名称: " DEVICE_NAME);
    Serial.println("[BLE] 用手机连接后即可接收心电数据");
}

void sendBLEMessage(const char* message)
{
    if (!connected) return;
    if (!message || strlen(message) == 0) return;

    pTxChar->setValue((uint8_t*)message, strlen(message));
    pTxChar->notify();
}
