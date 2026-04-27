#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include "bluetooth/ble.h"

/**
 * @file ble.cpp
 * @brief 心电数据 BLE 蓝牙传输模块实现
 */

/* ======================== BLE 服务与特征值定义 ======================== */

/** @brief 心电监测服务UUID */
#define SERVICE_UUID        "0000FFF0-0000-1000-8000-00805F9B34FB"

/** @brief 心电数据特征UUID */
#define CHARACTERISTIC_UUID "0000FFF1-0000-1000-8000-00805F9B34FB"

/** @brief 设备广播名称，包含 ECG_MONITOR 字样 */
#define DEVICE_NAME         "ESP32-ECG-MONITOR"

/* ======================== 全局对象 ======================== */

static BLEServer            *pServer          = NULL;
static BLEService           *pService         = NULL;
static BLECharacteristic    *pCharacteristic  = NULL;
static bool                  deviceConnected  = false;

/* ======================== 连接回调类 ======================== */

/**
 * @brief BLE 服务器回调类，监听客户端连接/断开事件
 */
class MyServerCallbacks : public BLEServerCallbacks
{
    void onConnect(BLEServer* pServer) override
    {
        deviceConnected = true;
        Serial.println("[BLE] 客户端已连接");
    }

    void onDisconnect(BLEServer* pServer) override
    {
        deviceConnected = false;
        Serial.println("[BLE] 客户端已断开，重新开始广播");

        /* 断开后重新广播，允许其他设备连接 */
        pServer->getAdvertising()->start();
    }
};

/* ======================== 公共接口实现 ======================== */

void initBLE(void)
{
    /* 初始化 BLE 设备，设置广播名称 */
    BLEDevice::init(DEVICE_NAME);

    /* 创建 BLE 服务器 */
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());

    /* 创建 BLE 服务 */
    pService = pServer->createService(SERVICE_UUID);

    /* 创建心电数据特征值：支持通知(Notify)和读取(Read) */
    pCharacteristic = pService->createCharacteristic(
        CHARACTERISTIC_UUID,
        BLECharacteristic::PROPERTY_NOTIFY |
        BLECharacteristic::PROPERTY_READ
    );

    /* 添加 BLE 2902 描述符（通知功能必需） */
    pCharacteristic->addDescriptor(new BLE2902());

    /* 设置初始值为 0 */
    float initVal = 0.0f;
    pCharacteristic->setValue((uint8_t*)&initVal, sizeof(float));

    /* 启动服务 */
    pService->start();

    /* 配置并开始广播 */
    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->setMinPreferred(0x06);   /* 与 iOS 连接兼容性设置 */
    pAdvertising->setMinPreferred(0x12);
    BLEDevice::startAdvertising();

    Serial.println("[BLE] 初始化完成，设备名称: " DEVICE_NAME);
}

void sendECGData(float value)
{
    /* 只有在有客户端连接时才发送数据 */
    if (!deviceConnected) {
        return;
    }

    /* 将 float 值按 Little Endian 字节序写入特征值 */
    pCharacteristic->setValue((uint8_t*)&value, sizeof(float));

    /* 触发通知，推送数据到客户端 */
    pCharacteristic->notify();
}
