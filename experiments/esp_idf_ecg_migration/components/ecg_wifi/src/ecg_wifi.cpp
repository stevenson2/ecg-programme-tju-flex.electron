/**
 * @file ecg_wifi.cpp
 * @brief WiFi AP + HTTP 传输模块（ESP-IDF 简化移植）
 *
 * 提供：
 *   GET /api/records       -> 文本形式的 records.idx 列表
 *   GET /api/records/{id}/data -> 下载 /spiffs/ecgdata/ecg_rec_<id>.ecgr
 *   DELETE /api/records/{id}    -> 删除记录并刷新 recorder 计数
 */
#include "wifi/ecg_wifi.h"
#include "storage/ecg_recorder.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_http_server.h"
#include "nvs_flash.h"
#include "esp_log.h"

#define ECGR_BASE_PATH "/spiffs/ecgdata"

static const char *TAG = "ecg_wifi";
static bool g_wifi_on = false;
static httpd_handle_t g_server = NULL;
static int s_diagTxPower = 78;
static int s_diagChannel = 6;
static bool s_diagSeqSlow = false;

static int ecgWifiListHandler(httpd_req_t *req) {
    char buf[4096] = {0};
    int n = ecgRecorderList(buf, sizeof(buf));
    if (n < 0) {
        httpd_resp_send_404(req);
        return ESP_OK;
    }
    httpd_resp_set_type(req, "text/plain");
    httpd_resp_send(req, buf, n);
    return ESP_OK;
}

static bool get_record_id_path(httpd_req_t *req, char *path, size_t path_len, uint32_t *id) {
    /* 解析 /api/records/<id>/data 或 /api/records/<id> */
    const char *uri = req->uri;
    const char *p = strstr(uri, "/api/records/");
    if (!p) return false;
    p += strlen("/api/records/");
    char id_str[32] = {0};
    int i = 0;
    while (p[i] && p[i] >= '0' && p[i] <= '9' && i < 31) {
        id_str[i] = p[i];
        i++;
    }
    if (i == 0) return false;
    *id = (uint32_t)strtoul(id_str, NULL, 10);
    snprintf(path, path_len, ECGR_BASE_PATH "/ecg_rec_%u.ecgr", (unsigned)*id);
    return true;
}

static int ecgWifiDataHandler(httpd_req_t *req) {
    char path[160];
    uint32_t id = 0;
    if (!get_record_id_path(req, path, sizeof(path), &id)) {
        httpd_resp_send_404(req);
        return ESP_OK;
    }
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size <= 0) {
        httpd_resp_send_404(req);
        return ESP_OK;
    }
    httpd_resp_set_type(req, "application/octet-stream");
    FILE *f = fopen(path, "rb");
    if (!f) {
        httpd_resp_send_404(req);
        return ESP_OK;
    }
    uint8_t chunk[ECG_WIFI_CHUNK_BYTES];
    size_t rd;
    while ((rd = fread(chunk, 1, sizeof(chunk), f)) > 0) {
        if (httpd_resp_send_chunk(req, (const char *)chunk, rd) != ESP_OK) {
            break;
        }
    }
    fclose(f);
    httpd_resp_send_chunk(req, NULL, 0);
    return ESP_OK;
}

static int ecgWifiDeleteHandler(httpd_req_t *req) {
    char path[160];
    uint32_t id = 0;
    if (!get_record_id_path(req, path, sizeof(path), &id)) {
        httpd_resp_send_404(req);
        return ESP_OK;
    }
    remove(path);
    ecgRecorderRefreshCount();
    httpd_resp_sendstr(req, "OK");
    return ESP_OK;
}

static void register_uri_handlers(httpd_handle_t server) {
    httpd_uri_t list = {.uri="/api/records", .method=HTTP_GET, .handler=ecgWifiListHandler, .user_ctx=NULL};
    httpd_register_uri_handler(server, &list);
    httpd_uri_t data = {.uri="/api/records/*/data", .method=HTTP_GET, .handler=ecgWifiDataHandler, .user_ctx=NULL};
    httpd_register_uri_handler(server, &data);
    httpd_uri_t del = {.uri="/api/records/*", .method=HTTP_DELETE, .handler=ecgWifiDeleteHandler, .user_ctx=NULL};
    httpd_register_uri_handler(server, &del);
}

bool ecgWifiInit(void) {
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "nvs_flash_init failed: %s", esp_err_to_name(err));
        return false;
    }
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    return true;
}

bool ecgWifiStart(void) {
    if (g_wifi_on) return false;

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    esp_netif_create_default_wifi_ap();

    uint8_t mac[6] = {0};
    esp_wifi_get_mac(WIFI_IF_AP, mac);
    char ssid[32];
    snprintf(ssid, sizeof(ssid), "ESP32-ECG-%02X%02X", mac[4], mac[5]);

    wifi_config_t wc = {};
    strncpy((char *)wc.ap.ssid, ssid, sizeof(wc.ap.ssid) - 1);
    strncpy((char *)wc.ap.password, ECG_WIFI_AP_PASSWORD, sizeof(wc.ap.password) - 1);
    wc.ap.ssid_len = strlen(ssid);
    wc.ap.channel = (uint8_t)s_diagChannel;
    wc.ap.authmode = WIFI_AUTH_WPA2_PSK;
    wc.ap.max_connection = 4;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());

    httpd_config_t hc = HTTPD_DEFAULT_CONFIG();
    hc.lru_purge_enable = true;
    esp_err_t herr = httpd_start(&g_server, &hc);
    if (herr != ESP_OK || !g_server) {
        ESP_LOGE(TAG, "httpd_start failed: %s", esp_err_to_name(herr));
        return false;
    }
    register_uri_handlers(g_server);
    g_wifi_on = true;
    ESP_LOGI(TAG, "AP started, SSID=%s", ssid);
    return true;
}

void ecgWifiStop(void) {
    if (g_server) {
        httpd_stop(g_server);
        g_server = NULL;
    }
    esp_wifi_stop();
    esp_wifi_deinit();
    g_wifi_on = false;
}

bool ecgWifiIsOn(void) { return g_wifi_on; }
void ecgWifiProcess(void) { /* httpd runs in its own task in IDF */ }
void ecgWifiDiagSetTxPower(int v) { s_diagTxPower = v; }
void ecgWifiDiagSetChannel(int v) { s_diagChannel = v; }
void ecgWifiDiagSetSeqSlow(bool v) { s_diagSeqSlow = v; }
int ecgWifiDiagGetTxPower(void) { return s_diagTxPower; }
int ecgWifiDiagGetChannel(void) { return s_diagChannel; }
bool ecgWifiDiagGetSeqSlow(void) { return s_diagSeqSlow; }
bool ecgWifiDiagStaConnect(const char* ssid, const char* pass) { (void)ssid; (void)pass; return false; }
void ecgWifiDiagStaDisconnect(void) {}
int ecgWifiDiagStaStatus(void) { return 0; }
void ecgWifiDiagStaIp(char* buf, size_t len) { if (len > 0) buf[0] = '\0'; }
int ecgWifiDiagGetMode(void) { return g_wifi_on ? 2 : 0; }
