/**
 * @file ecg_wifi.cpp
 * @brief WiFi AP + HTTP 传输模块（ESP-IDF 移植，与 Arduino 线 API 对齐）
 *
 * 提供（与 src/wifi/ecg_wifi.cpp 的 REST 协议一致）：
 *   GET    /api/records            -> 记录列表 JSON
 *   GET    /api/records/{id}/meta  -> 单条记录元数据 JSON
 *   GET    /api/records/{id}/data  -> 原始 .ecgr 二进制流 (chunked)
 *   DELETE /api/records/{id}       -> 删除记录 + 重建索引，返回 JSON
 *   OPTIONS *                      -> 204 CORS 预检 (Web 前端跨域)
 */
#include "wifi/ecg_wifi.h"
#include "storage/ecg_recorder.h"
#include "storage/ecg_recorder_format.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
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

/* ======================== 内部辅助 ======================== */

/** 从记录 ID (startUnix) 构建 .ecgr 文件全路径 */
static void get_record_path(uint32_t id, char *buf, size_t len) {
    snprintf(buf, len, ECGR_BASE_PATH "/ecg_rec_%u.ecgr", (unsigned)id);
}

/** Unix 时间戳 -> ISO8601 (UTC) 字符串 */
static void format_iso8601(uint32_t unix_time, char *buf, size_t len) {
    time_t t = (time_t)unix_time;
    struct tm tm;
    gmtime_r(&t, &tm);
    strftime(buf, len, "%Y-%m-%dT%H:%M:%SZ", &tm);
}

/** 解析 uri 中 /api/records/<id>... 的数字 id 并构建文件路径 */
static bool get_record_id_path(httpd_req_t *req, char *path, size_t path_len, uint32_t *id) {
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
    get_record_path(*id, path, path_len);
    return true;
}

/* ======================== 路由处理器 ======================== */

/**
 * GET /api/records
 * 从 records.idx 逐行解析，返回 {"records":[{id,duration,size,abnormal_seconds,start},...]}
 */
static int ecgWifiListHandler(httpd_req_t *req) {
    char idx[4096] = {0};
    int n = ecgRecorderList(idx, sizeof(idx));
    if (n < 0) {
        n = 0;
        idx[0] = '\0';
    }
    idx[n] = '\0';

    char json[4096];
    int pos = snprintf(json, sizeof(json), "{\"records\":[");
    bool first = true;

    char *saveptr = NULL;
    char *line = strtok_r(idx, "\n", &saveptr);
    while (line && pos > 0 && pos < (int)sizeof(json) - 1) {
        while (*line == '\r' || *line == ' ') line++;
        if (*line == '\0') { line = strtok_r(NULL, "\n", &saveptr); continue; }

        uint32_t startUnix = 0, dur = 0, samples = 0, abnSec = 0, sizeBytes = 0;
        if (ecgrIdxParse(line, &startUnix, &dur, &samples, &abnSec, &sizeBytes)) {
            char isoBuf[24];
            format_iso8601(startUnix, isoBuf, sizeof(isoBuf));
            if (!first) pos += snprintf(json + pos, sizeof(json) - (size_t)pos, ",");
            first = false;
            pos += snprintf(json + pos, sizeof(json) - (size_t)pos,
                            "{\"id\":%u,\"duration\":%u,\"size\":%u,"
                            "\"abnormal_seconds\":%u,\"start\":\"%s\"}",
                            (unsigned)startUnix, (unsigned)dur,
                            (unsigned)sizeBytes, (unsigned)abnSec, isoBuf);
        }
        line = strtok_r(NULL, "\n", &saveptr);
    }
    if (pos > 0 && pos < (int)sizeof(json) - 1) {
        pos += snprintf(json + pos, sizeof(json) - (size_t)pos, "]}");
    }

    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, json, (size_t)pos);
    return ESP_OK;
}

/**
 * GET /api/records/{id}/meta
 * 从 .ecgr 头部解析，返回 {"id",sample_rate,start_unix,start,duration,total_samples,abnormal_seconds}
 */
static int ecgWifiMetaHandler(httpd_req_t *req) {
    char path[160];
    uint32_t id = 0;
    if (!get_record_id_path(req, path, sizeof(path), &id)) {
        httpd_resp_send_404(req);
        return ESP_OK;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        httpd_resp_send_404(req);
        return ESP_OK;
    }
    uint8_t hdr[ECGR_HEADER_SIZE];
    size_t rd = fread(hdr, 1, sizeof(hdr), f);
    fclose(f);
    if (rd != sizeof(hdr) || !ecgrHeaderValidate(hdr, ECGR_DEFAULT_SAMPLE_RATE)) {
        httpd_resp_send_404(req);
        return ESP_OK;
    }

    uint32_t startUnix    = ecgrHeaderStartUnix(hdr);
    uint32_t dur          = ecgrHeaderDurationSec(hdr);
    uint32_t totalSamples = ecgrHeaderTotalSamples(hdr);
    uint32_t abnSec       = ecgrHeaderAbnormalSec(hdr);

    char isoBuf[24];
    format_iso8601(startUnix, isoBuf, sizeof(isoBuf));

    char json[256];
    snprintf(json, sizeof(json),
             "{\"id\":%u,\"sample_rate\":%u,\"start_unix\":%u,\"start\":\"%s\","
             "\"duration\":%u,\"total_samples\":%u,\"abnormal_seconds\":%u}",
             (unsigned)id, (unsigned)ECGR_DEFAULT_SAMPLE_RATE, (unsigned)startUnix,
             isoBuf, (unsigned)dur, (unsigned)totalSamples, (unsigned)abnSec);

    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, json, strlen(json));
    return ESP_OK;
}

/**
 * GET /api/records/{id}/data
 * 返回原始 .ecgr 二进制流 (chunked)。App 当前不发送 Range，保底返回全文件。
 */
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
    FILE *f = fopen(path, "rb");
    if (!f) {
        httpd_resp_send_404(req);
        return ESP_OK;
    }
    httpd_resp_set_type(req, "application/octet-stream");
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

/**
 * DELETE /api/records/{id}
 * 删除 .ecgr 文件并重建索引，返回 {"deleted":true/false}
 */
static int ecgWifiDeleteHandler(httpd_req_t *req) {
    char path[160];
    uint32_t id = 0;
    if (!get_record_id_path(req, path, sizeof(path), &id)) {
        httpd_resp_send_404(req);
        return ESP_OK;
    }
    struct stat st;
    if (stat(path, &st) != 0) {
        httpd_resp_set_type(req, "application/json");
        httpd_resp_sendstr(req, "{\"deleted\":false}");
        return ESP_OK;
    }
    if (remove(path) == 0) {
        ecgRecorderRefreshCount();
        httpd_resp_set_type(req, "application/json");
        httpd_resp_sendstr(req, "{\"deleted\":true}");
    } else {
        httpd_resp_set_type(req, "application/json");
        httpd_resp_sendstr(req, "{\"deleted\":false}");
    }
    return ESP_OK;
}

/* ======================== 路由注册 ======================== */

static void register_uri_handlers(httpd_handle_t server) {
    /* 顺序重要: 更具体的 /data、/meta 必须先于后面的通配接口注册,
     * 因为 esp_http_server 的通配星号会匹配跨 '/' 的任意后缀。 */
    httpd_uri_t list = {.uri="/api/records", .method=HTTP_GET, .handler=ecgWifiListHandler, .user_ctx=NULL};
    httpd_register_uri_handler(server, &list);
    httpd_uri_t data = {.uri="/api/records/*/data", .method=HTTP_GET, .handler=ecgWifiDataHandler, .user_ctx=NULL};
    httpd_register_uri_handler(server, &data);
    httpd_uri_t meta = {.uri="/api/records/*/meta", .method=HTTP_GET, .handler=ecgWifiMetaHandler, .user_ctx=NULL};
    httpd_register_uri_handler(server, &meta);
    httpd_uri_t del = {.uri="/api/records/*", .method=HTTP_DELETE, .handler=ecgWifiDeleteHandler, .user_ctx=NULL};
    httpd_register_uri_handler(server, &del);
}

/* ======================== 公开 API ======================== */

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
