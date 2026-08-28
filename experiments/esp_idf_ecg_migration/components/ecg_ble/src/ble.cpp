/**
 * @file ble.cpp
 * @brief ECG BLE NUS 模块（ESP-IDF NimBLE 移植版）
 */
#include "bluetooth/ble.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "esp_log.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

#define TAG "ecg_ble"
#define DEVICE_NAME "ESP32-ECG"

/* NUS UUIDs（与 Arduino 版相同） */
static const ble_uuid128_t nus_svc_uuid =
    BLE_UUID128_INIT(0x9e,0xca,0xdc,0x24,0x0e,0xe5,0xa9,0xe0,
                     0x93,0xf3,0xa3,0xb5,0x01,0x00,0x40,0x6e);
static const ble_uuid128_t nus_tx_uuid =
    BLE_UUID128_INIT(0x9e,0xca,0xdc,0x24,0x0e,0xe5,0xa9,0xe0,
                     0x93,0xf3,0xa3,0xb5,0x02,0x00,0x40,0x6e);
static const ble_uuid128_t nus_rx_uuid =
    BLE_UUID128_INIT(0x9e,0xca,0xdc,0x24,0x0e,0xe5,0xa9,0xe0,
                     0x93,0xf3,0xa3,0xb5,0x03,0x00,0x40,0x6e);

static uint16_t g_tx_handle;
static uint16_t g_rx_handle;
static uint8_t g_own_addr_type = BLE_OWN_ADDR_PUBLIC;
static uint16_t g_conn_handle = BLE_HS_CONN_HANDLE_NONE;
static bool g_connected = false;
static QueueHandle_t g_cmd_queue = nullptr;
static char g_rx_line[64];
static int g_rx_len = 0;

static int nus_rx_access(uint16_t conn_handle, uint16_t attr_handle,
                         struct ble_gatt_access_ctxt *ctxt, void *arg) {
    if (ctxt->op == BLE_GATT_ACCESS_OP_WRITE_CHR) {
        struct os_mbuf *om = ctxt->om;
        while (om) {
            uint8_t *data = om->om_data;
            int len = om->om_len;
            for (int i = 0; i < len; i++) {
                char c = (char)data[i];
                if (c == '\n' || c == '\r' || c == '\0') {
                    if (g_rx_len > 0 && g_cmd_queue) {
                        g_rx_line[g_rx_len] = '\0';
                        xQueueSend(g_cmd_queue, g_rx_line, 0);
                        g_rx_len = 0;
                    }
                } else if (g_rx_len < (int)sizeof(g_rx_line) - 1) {
                    g_rx_line[g_rx_len++] = c;
                }
            }
            om = SLIST_NEXT(om, om_next);
        }
    }
    return 0;
}

static const struct ble_gatt_svc_def gatt_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &nus_svc_uuid.u,
        .characteristics = (struct ble_gatt_chr_def[]) {
            { .uuid = &nus_tx_uuid.u,
              .access_cb = nus_rx_access,
              .flags = BLE_GATT_CHR_F_NOTIFY,
              .val_handle = &g_tx_handle },
            { .uuid = &nus_rx_uuid.u,
              .access_cb = nus_rx_access,
              .flags = BLE_GATT_CHR_F_WRITE | BLE_GATT_CHR_F_WRITE_NO_RSP,
              .val_handle = &g_rx_handle },
            { 0 }
        },
    },
    { 0 }
};

static int gap_event(struct ble_gap_event *event, void *arg);

static void start_advertising(void) {
    struct ble_hs_adv_fields fields = {0};
    struct ble_gap_adv_params adv_params = {0};
    const char *name = ble_svc_gap_device_name();
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.name = (uint8_t *)name;
    fields.name_len = strlen(name);
    fields.name_is_complete = 1;
    int rc = ble_gap_adv_set_fields(&fields);
    if (rc) { ESP_LOGE(TAG, "adv set fields rc=%d", rc); return; }
    adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
    adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;
    rc = ble_gap_adv_start(g_own_addr_type, NULL, BLE_HS_FOREVER,
                           &adv_params, gap_event, NULL);
    (void)rc;
    ESP_LOGI(TAG, "advertising started");
}

static int gap_event(struct ble_gap_event *event, void *arg) {
    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        if (event->connect.status == 0) {
            g_connected = true;
            g_conn_handle = event->connect.conn_handle;
            ESP_LOGI(TAG, "connected");
        } else {
            ESP_LOGI(TAG, "connect failed");
            start_advertising();
        }
        return 0;
    case BLE_GAP_EVENT_DISCONNECT:
        g_connected = false;
        g_conn_handle = BLE_HS_CONN_HANDLE_NONE;
        start_advertising();
        return 0;
    case BLE_GAP_EVENT_ADV_COMPLETE:
        start_advertising();
        return 0;
    case BLE_GAP_EVENT_SUBSCRIBE:
        ESP_LOGI(TAG, "subscribe");
        return 0;
    default:
        return 0;
    }
}

static void on_sync(void) {
    uint8_t addr_type;
    ble_hs_util_ensure_addr(0);
    ble_hs_id_infer_auto(0, &addr_type);
    g_own_addr_type = addr_type;
    start_advertising();
}

static void on_reset(int reason) {
    ESP_LOGE(TAG, "reset reason=%d", reason);
}

static void gatt_register_cb(struct ble_gatt_register_ctxt *ctxt, void *arg) {
    char buf[BLE_UUID_STR_LEN];
    switch (ctxt->op) {
    case BLE_GATT_REGISTER_OP_SVC:
        ESP_LOGI(TAG, "registered service %s",
                 ble_uuid_to_str(ctxt->svc.svc_def->uuid, buf));
        break;
    case BLE_GATT_REGISTER_OP_CHR:
        ESP_LOGI(TAG, "registered characteristic %s",
                 ble_uuid_to_str(ctxt->chr.chr_def->uuid, buf));
        break;
    case BLE_GATT_REGISTER_OP_DSC:
        ESP_LOGI(TAG, "registered descriptor %s",
                 ble_uuid_to_str(ctxt->dsc.dsc_def->uuid, buf));
        break;
    default:
        break;
    }
}

static void host_task(void *param) {
    nimble_port_run();
    vTaskDelete(NULL);
}

extern "C" void initBLE(void) {
    if (g_cmd_queue == nullptr) {
        g_cmd_queue = xQueueCreate(4, 32);
    }
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }
    ret = nimble_port_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "nimble_port_init failed");
        return;
    }
    ble_svc_gap_init();
    ble_svc_gatt_init();
    ble_svc_gap_device_name_set(DEVICE_NAME);
    int grc = ble_gatts_count_cfg(gatt_svcs);
    if (grc != 0) {
        ESP_LOGE(TAG, "ble_gatts_count_cfg failed rc=%d", grc);
        return;
    }
    grc = ble_gatts_add_svcs(gatt_svcs);
    if (grc != 0) {
        ESP_LOGE(TAG, "ble_gatts_add_svcs failed rc=%d", grc);
        return;
    }

    ble_hs_cfg.reset_cb = on_reset;
    ble_hs_cfg.sync_cb = on_sync;
    ble_hs_cfg.gatts_register_cb = gatt_register_cb;

    nimble_port_freertos_init(host_task);
    ESP_LOGI(TAG, "BLE init OK, device=%s", DEVICE_NAME);
}

extern "C" void sendBLEMessage(const char *message) {
    if (!message || !g_connected || !message[0]) return;
    struct os_mbuf *om = ble_hs_mbuf_from_flat(message, strlen(message));
    if (!om) return;
    ble_gatts_notify_custom(g_conn_handle, g_tx_handle, om);
}

extern "C" bool isBLEConnected(void) {
    return g_connected;
}

extern "C" bool bleCommandQueueTake(char *out, size_t len) {
    char buf[32];
    if (!g_cmd_queue || !out || len == 0) return false;
    if (xQueueReceive(g_cmd_queue, buf, 0) != pdTRUE) return false;
    strncpy(out, buf, len);
    out[len - 1] = '\0';
    return true;
}
