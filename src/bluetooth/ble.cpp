#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <esp_bt.h>
#include <esp_bt_main.h>
#include <esp_gap_ble_api.h>
#include <freertos/queue.h>
#include <cstring>
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

/* BLE 命令队列: 深度 4, 每条最多 31 字节 (+ null) */
static QueueHandle_t         s_cmdQueue  = NULL;

/* RX 行缓冲: 累加字节直到 '\n' 或 '\0' */
static char  s_rxLineBuf[64] = {0};
static int   s_rxLineLen     = 0;
/* 最后接收字节时间 (防御: 兼容无 '\n' 结束符的命令, 2026-08-10) */
static unsigned long s_lastRxMs = 0;

/* ======================== 连接回调 ======================== */

class ServerCallbacks : public BLEServerCallbacks
{
    void onConnect(BLEServer* srv, esp_ble_gatts_cb_param_t* param) override
    {
        connected = true;
        Serial.println("[BLE] 手机已连接");

        /* 2026-08-14 (重连累积阶梯感根治): 移除外设侧 esp_ble_gap_update_conn_params。
         * 原 2026-08-12 每次 onConnect 主动请求 15-22.5ms 间隔, 但 Android 对
         * "外设发起的 L2CAP 连接参数更新请求"处理不佳: 首次连接生效, 重连时与
         * App 端 requestConnectionPriority(high) 冲突/竞态, 使连接间隔随重连次数
         * 逐步退化 (波形越来越阶梯状), 退出 App 重建 Android BLE 栈才恢复。
         * 连接间隔改由 App 端 requestConnectionPriority 唯一控制 (central 发起,
         * Android 可靠执行); 广播首选连接参数 (setMinPreferred 7.5ms /
         * setMaxPreferred 22.5ms) 保留作为协商提示。 */
        (void)srv;
        (void)param;
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
        if (rx.length() == 0) return;

        if (!s_cmdQueue) return;   /* 队列未就绪 (initBLE 尚未完成) — 丢弃 */

        /* 逐字节累加到行缓冲; 遇到 '\n' / '\r' / '\0' 视为命令结束,
         * 将完整行 POST 到 FreeRTOS 队列供 main.cpp 主循环消费.
         * 无 SPIFFS I/O、无 BLE send、无阻塞调用 — 仅在 Core 0 BLE 上下文。 */
        for (size_t i = 0; i < rx.length(); i++) {
            char c = rx[i];
            if (c == '\n' || c == '\r' || c == '\0') {
                if (s_rxLineLen > 0) {
                    s_rxLineBuf[s_rxLineLen] = '\0';
                    xQueueSend(s_cmdQueue, s_rxLineBuf, 0);  /* 非阻塞, 队列满则丢 */
                    s_rxLineLen = 0;
                }
            } else if (s_rxLineLen < (int)(sizeof(s_rxLineBuf) - 1)) {
                s_rxLineBuf[s_rxLineLen++] = c;
                s_lastRxMs = millis();   /* 记录最后接收时刻 (超时提交用) */
            }
            /* 缓冲区满 → 静默丢弃该字节 (防止溢出) */
        }
    }
};

/* ======================== GAP 回调 (连接参数诊断, 2026-08-12) ========================
 * 打印连接参数协商结果 — 验证手机是否接受外设的连接参数更新请求
 * (Android 常忽略, 若日志显示 status=0 且 min/max≈0x0C-0x12 则生效) */

static void gapCallback(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t* param)
{
    if (event == ESP_GAP_BLE_UPDATE_CONN_PARAMS_EVT) {
        auto* p = &param->update_conn_params;
        Serial.printf("[BLE] conn params evt: status=%d min=%d max=%d latency=%d conn_int=%d timeout=%d\n",
                      p->status, p->min_int, p->max_int, p->latency, p->conn_int, p->timeout);
    }
}

/* ======================== 公共接口 ======================== */

void initBLE(void)
{
    /* 创建 BLE 命令队列 (深度 4, 每条 ≤31+null 字节)
     * 必须在广播前创建, 确保 RxCallbacks::onWrite 可立即使用 */
    s_cmdQueue = xQueueCreate(4, 32);

    /* 初始化 BLE 设备 */
    BLEDevice::init(DEVICE_NAME);

    /* 2026-08-14 修复 (阶梯感): MTU 协商上限 185。默认 MTU 23 下每帧 ~50B
     * 被 L2CAP 拆成 ~3 个 ATT 包 → 250Hz notify 包量 ×3 → 链路拥塞丢帧,
     * App 按固定 250Hz 时间轴绘制 → 波形阶梯/粗糙。客户端 requestMtu 后
     * 单帧 1 包, 包量降 ~3×, 有效吞吐恢复。 */
    BLEDevice::setMTU(185);

    /* 注册 GAP 回调 (连接参数协商诊断, 2026-08-12) */
    esp_ble_gap_register_callback(gapCallback);

    /* 设置 BLE 发射功率（原始 +9dBm，确保连接稳定性）*/
    esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_ADV, ESP_PWR_LVL_P9);   /* +9dBm */
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

bool bleCommandQueueTake(char* out, size_t len)
{
    if (!s_cmdQueue || !out || len == 0) return false;

    /* 2026-08-10 防御: BLE 命令超时提交 — 兼容不带 '\n' 结束符的客户端
     * (App 旧版 sendCommand 未追加换行, 命令永久卡在行缓冲, 定时录制
     * 等 BLE 命令从未送达)。距最后接收 >100ms 且行缓冲非空 → 视为
     * 命令结束并提交。带 '\n' 的客户端不受影响 (收到结束符即提交)。 */
    if (s_rxLineLen > 0 && (millis() - s_lastRxMs) > 100) {
        s_rxLineBuf[s_rxLineLen] = '\0';
        xQueueSend(s_cmdQueue, s_rxLineBuf, 0);
        s_rxLineLen = 0;
    }

    char buf[32];
    if (xQueueReceive(s_cmdQueue, buf, 0) != pdTRUE) return false;

    strncpy(out, buf, len);
    out[len - 1] = '\0';
    return true;
}
