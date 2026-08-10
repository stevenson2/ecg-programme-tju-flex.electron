/**
 * @file ecg_wifi.cpp
 * @brief WiFi AP 模式下 HTTP 传输模块 — 录制记录管理实现
 *
 * 启动 ESP32-S3 SoftAP + 同步 WebServer，提供 RESTful API
 * 供 Flutter App 查询/下载/删除 ECG 录制记录。
 *
 * == 路由表 ==
 *   GET    /api/records            → 记录列表 JSON
 *   GET    /api/records/{id}/meta  → 单条记录元数据 JSON
 *   GET    /api/records/{id}/data  → 原始 .ecgr 二进制流 (支持 Range)
 *   DELETE /api/records/{id}       → 删除记录 + 重建索引
 *   其他                             → 404
 *
 * == BLE 共存 ==
 * ESP32-S3 无线电支持 WiFi+BLE 时分复用。AP 运行期间 BLE NUS
 * 保持激活, BLE 事件回调在 WiFi 任务间隙调度, 无需额外处理。
 *
 * == 内存 ==
 *   - WebServer 对象 + 路由表: ~4-6KB 堆
 *   - AP 协议栈: ~30KB 堆 (WiFi.begin 分配)
 *   - 流缓冲区: ECG_WIFI_CHUNK_BYTES (默认 1024 字节, 栈)
 *   - 不缓冲整个录制文件 (最高可达数 MB)
 */

#include "wifi/ecg_wifi.h"
#include "storage/ecg_recorder_format.h"
#include "storage/ecg_recorder.h"

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <uri/UriBraces.h>
#include <SPIFFS.h>
#include <time.h>

/* ======================== 静态变量 ======================== */

/** WebServer 实例指针 (init 时创建, 生命周期与模块相同) */
static WebServer* g_server = NULL;

/** AP 是否已启动 */
static bool g_wifiOn = false;

/* ======================== 诊断配置 (DIAG 命令可调, 2026-08-10) ========================
 * WiFi beacon 不可见专项: 8 轮最小二分全部通过, 正式固件全模块失败。
 * 本组变量让"烧录一次, 运行时切换多变量"成为可能, 默认值 = 正式固件原行为。
 * 证据来源: WiFiManager PR#1865 (2.0.17 世代 S3 实测)、ESP-IDF#13508 (RF 相关,
 * TX power/信道/位置)、ESPHome#6456 + PlatformIO 社区 (N16R8 降 TX power 修复)。 */
static int  s_diagTxPower = 78;    /* 0=跳过 setTxPower, 34=8.5dBm, 60=15dBm, 78=19.5dBm(原行为) */
static int  s_diagChannel = 6;     /* AP 信道 (原行为=6) */
static bool s_diagSeqSlow = false; /* true=PR#1865 式: WIFI_OFF→500ms→WIFI_AP→500ms→softAP→setSleep(false) */

/* ======================== 内部辅助: 路径构建 ======================== */

/** ECG 数据基础路径 (与 recorder 模块一致: ecg_recorder.cpp ECGR_BASE_PATH) */
#define ECGR_BASE_PATH  "/ecgdata"

/**
 * @brief 从记录 ID (startUnixTime) 构建 .ecgr 文件全路径
 * @param id   记录 ID (= startUnixTime)
 * @param buf  输出缓冲区
 * @param len  缓冲区大小
 * @return 写入字节数, 缓冲区不足返回 -1
 */
static int getRecordPath(uint32_t id, char* buf, size_t len)
{
    return snprintf(buf, len, ECGR_BASE_PATH "/ecg_rec_%u.ecgr", (unsigned)id);
}

/* ======================== 内部辅助: ISO8601 时间格式化 ======================== */

/**
 * @brief 将 Unix 时间戳格式化为 ISO8601 字符串 (UTC)
 *
 * 使用 gmtime_r 将时间戳转为 UTC 分解时间, 再以 strftime 格式化。
 * ESP32 Arduino 核心工具链 (newlib) 支持 gmtime_r。
 *
 * == 时间戳来源说明 ==
 * 当前固件未同步 NTP, 录制时间戳来自 millis()/1000 (上电秒数)。
 * 因此 ISO8601 格式为 1970-01-01T00:XX:XXZ, 代表上电后经过的秒数。
 * 若后续集成 NTP, gmtime_r 自动输出真实 UTC 时间。
 *
 * @param unixTime  Unix 时间戳 (uint32_t, 兼容 2038 年前)
 * @param buf      输出缓冲区
 * @param bufLen   缓冲区大小 (建议 ≥ 24)
 */
static void formatISO8601(uint32_t unixTime, char* buf, size_t bufLen)
{
    time_t t = (time_t)unixTime;
    struct tm tm;
    gmtime_r(&t, &tm);
    strftime(buf, bufLen, "%Y-%m-%dT%H:%M:%SZ", &tm);
}

/* ======================== 内部辅助: Range 头解析 ======================== */

/**
 * @brief 解析 HTTP Range 头部并计算字节范围
 *
 * 支持的格式 (RFC 7233):
 *   - bytes=start-end   (两端指定, end 为 inclusive)
 *   - bytes=start-      (从 start 到文件末尾)
 *   - bytes=-suffix     (文件末尾 suffix 字节)
 *
 * @param rangeHeader  Range 头值 (不含 "Range: " 前缀, 例如 "bytes=0-1023")
 * @param fileSize     文件总字节数
 * @param outStart     输出: 起始偏移量
 * @param outEnd       输出: 结束偏移量 (inclusive)
 * @return true 解析成功且在文件范围内, false 无效请求
 */
static bool parseRangeHeader(const String& rangeHeader, uint32_t fileSize,
                              uint32_t* outStart, uint32_t* outEnd)
{
    if (!outStart || !outEnd) return false;
    if (fileSize == 0) return false;

    // 必须以前缀 "bytes=" 开头
    const char* prefix = "bytes=";
    if (!rangeHeader.startsWith(prefix)) return false;

    String rangeSpec = rangeHeader.substring(strlen(prefix));
    rangeSpec.trim();
    if (rangeSpec.length() == 0) return false;

    int dashPos = rangeSpec.indexOf('-');
    if (dashPos < 0) return false;

    String startStr = rangeSpec.substring(0, dashPos);
    String endStr   = rangeSpec.substring(dashPos + 1);

    uint32_t start = 0;
    uint32_t end   = fileSize - 1;

    if (startStr.length() > 0) {
        // "bytes=start-end" 或 "bytes=start-"
        start = (uint32_t)startStr.toInt();
        if (endStr.length() > 0) {
            // "bytes=start-end"
            end = (uint32_t)endStr.toInt();
        }
        // else: "bytes=start-" → end 保持 fileSize-1
    } else {
        // "bytes=-suffix" → 最后 suffix 字节
        if (endStr.length() == 0) return false;
        uint32_t suffix = (uint32_t)endStr.toInt();
        if (suffix > fileSize) suffix = fileSize;
        start = fileSize - suffix;
        end   = fileSize - 1;
    }

    // 验证范围
    if (start > end) return false;
    if (start >= fileSize) return false;
    if (end >= fileSize) end = fileSize - 1;

    *outStart = start;
    *outEnd   = end;
    return true;
}

/* ======================== 内部辅助: 索引重建 ======================== */

/**
 * @brief 重建 records.idx 索引文件
 *
 * 扫描 /ecgdata/*.ecgr, 读取每文件头部, 重写索引行。
 * 逻辑与 ecg_recorder.cpp::rebuildIndex() 等价, 但在此模块独立实现
 * 以避免耦合 recorder 的 static 函数。
 */
static void rebuildIdx(void)
{
    SPIFFS.remove(ECGR_BASE_PATH "/records.idx");

    fs::File idx = SPIFFS.open(ECGR_BASE_PATH "/records.idx", "w");
    if (!idx) {
        Serial.println("[WiFi] WARN: cannot create records.idx for rebuild");
        return;
    }

    fs::File dir = SPIFFS.open(ECGR_BASE_PATH);
    if (!dir || !dir.isDirectory()) {
        idx.close();
        return;
    }

    uint32_t count = 0;
    fs::File f = dir.openNextFile();
    while (f) {
        const char* name = f.name();
        if (name && strstr(name, ".ecgr")) {
            if (f.size() >= ECGR_HEADER_SIZE) {
                uint8_t hdr[ECGR_HEADER_SIZE];
                if (f.read(hdr, ECGR_HEADER_SIZE) == ECGR_HEADER_SIZE
                    && ecgrHeaderValidate(hdr, ECGR_DEFAULT_SAMPLE_RATE))
                {
                    uint32_t samples = ecgrHeaderTotalSamples(hdr);
                    uint32_t dur    = ecgrHeaderDurationSec(hdr);
                    uint32_t abn    = ecgrHeaderAbnormalSec(hdr);
                    uint32_t st     = ecgrHeaderStartUnix(hdr);
                    uint32_t sz     = (uint32_t)f.size();

                    char line[128];
                    int lineLen = ecgrIdxLine(line, sizeof(line),
                                              st, dur, samples, abn, sz);
                    if (lineLen > 0) {
                        idx.write((const uint8_t*)line, (size_t)lineLen);
                        count++;
                    }
                }
            }
        }
        f.close();
        f = dir.openNextFile();
    }
    dir.close();

    idx.flush();
    idx.close();

    Serial.printf("[WiFi] idx rebuilt: %u records\n", (unsigned)count);
}

/* ======================== 路由处理器 (前向声明 + 实现) ======================== */

static void handleRecordsList();
static void handleRecordsMeta();
static void handleRecordsData();
static void handleRecordsDelete();
static void handleNotFound();

/**
 * @brief GET /api/records
 *
 * 返回所有录制记录的 JSON 数组 (从 records.idx 读取)。
 * 空列表返回 {"records":[]}。
 */
static void handleRecordsList(void)
{
    // 构建 JSON 数组, 逐行解析 records.idx
    String json = "{\"records\":[";
    bool first = true;

    fs::File idx = SPIFFS.open(ECGR_BASE_PATH "/records.idx", "r");
    if (idx) {
        // 读取整个 idx 文件到字符串缓冲区 (idx 文件很小, 通常 <1KB)
        size_t idxSize = idx.size();
        if (idxSize > 0 && idxSize < 4096) {
            char* buf = (char*)malloc(idxSize + 1);
            if (buf) {
                size_t n = idx.read((uint8_t*)buf, idxSize);
                if (n == idxSize) {
                    buf[idxSize] = '\0';

                    // 逐行解析
                    char* saveptr = NULL;
                    char* line = strtok_r(buf, "\n", &saveptr);
                    while (line) {
                        // 跳过空行
                        while (*line == '\r' || *line == ' ') line++;
                        if (*line == '\0') {
                            line = strtok_r(NULL, "\n", &saveptr);
                            continue;
                        }

                        uint32_t startUnix = 0, dur = 0, samples = 0;
                        uint32_t abnSec = 0, sizeBytes = 0;
                        if (ecgrIdxParse(line, &startUnix, &dur, &samples,
                                        &abnSec, &sizeBytes))
                        {
                            char isoBuf[24];
                            formatISO8601(startUnix, isoBuf, sizeof(isoBuf));

                            // 追加 JSON 对象
                            if (!first) json += ',';
                            first = false;

                            char entry[256];
                            snprintf(entry, sizeof(entry),
                                     "{\"id\":%u,\"duration\":%u,"
                                     "\"size\":%u,\"abnormal_seconds\":%u,"
                                     "\"start\":\"%s\"}",
                                     (unsigned)startUnix, (unsigned)dur,
                                     (unsigned)sizeBytes, (unsigned)abnSec,
                                     isoBuf);
                            json += entry;
                        }

                        line = strtok_r(NULL, "\n", &saveptr);
                    }
                }
                free(buf);
            }
        }
        idx.close();
    }

    json += "]}";

    g_server->send(200, "application/json", json);
}

/**
 * @brief GET /api/records/{id}/meta
 *
 * 返回单条录制记录的元数据 JSON。
 * 文件不存在时返回 404。
 */
static void handleRecordsMeta(void)
{
    String idStr = g_server->pathArg(0);
    uint32_t id = (uint32_t)idStr.toInt();

    char path[64];
    getRecordPath(id, path, sizeof(path));

    fs::File f = SPIFFS.open(path, "r");
    if (!f || f.size() < ECGR_HEADER_SIZE) {
        if (f) f.close();
        g_server->send(404, "application/json", "{\"error\":\"not found\"}");
        return;
    }

    uint8_t hdr[ECGR_HEADER_SIZE];
    if (f.read(hdr, ECGR_HEADER_SIZE) != ECGR_HEADER_SIZE
        || !ecgrHeaderValidate(hdr, ECGR_DEFAULT_SAMPLE_RATE))
    {
        f.close();
        g_server->send(404, "application/json", "{\"error\":\"not found\"}");
        return;
    }

    uint32_t startUnix    = ecgrHeaderStartUnix(hdr);
    uint32_t dur          = ecgrHeaderDurationSec(hdr);
    uint32_t totalSamples = ecgrHeaderTotalSamples(hdr);
    uint32_t abnSec       = ecgrHeaderAbnormalSec(hdr);

    f.close();

    char isoBuf[24];
    formatISO8601(startUnix, isoBuf, sizeof(isoBuf));

    char json[256];
    snprintf(json, sizeof(json),
             "{\"id\":%u,\"sample_rate\":%u,\"start_unix\":%u,"
             "\"start\":\"%s\",\"duration\":%u,\"total_samples\":%u,"
             "\"abnormal_seconds\":%u}",
             (unsigned)id,
             (unsigned)ECGR_DEFAULT_SAMPLE_RATE,
             (unsigned)startUnix,
             isoBuf,
             (unsigned)dur,
             (unsigned)totalSamples,
             (unsigned)abnSec);

    g_server->send(200, "application/json", json);
}

/**
 * @brief GET /api/records/{id}/data
 *
 * 返回原始 .ecgr 二进制数据。
 * 支持 Range 头 → 206 Partial Content; 无 Range → 200 全量。
 * 数据以 ECG_WIFI_CHUNK_BYTES 分块流式发送, 不缓冲整个文件。
 */
static void handleRecordsData(void)
{
    String idStr = g_server->pathArg(0);
    uint32_t id = (uint32_t)idStr.toInt();

    char path[64];
    getRecordPath(id, path, sizeof(path));

    fs::File f = SPIFFS.open(path, "r");
    if (!f || f.size() < ECGR_HEADER_SIZE) {
        if (f) f.close();
        g_server->send(404, "application/json", "{\"error\":\"not found\"}");
        return;
    }

    uint32_t fileSize = (uint32_t)f.size();

    // 验证头部以确认文件有效
    uint8_t hdr[ECGR_HEADER_SIZE];
    if (f.read(hdr, ECGR_HEADER_SIZE) != ECGR_HEADER_SIZE
        || !ecgrHeaderValidate(hdr, ECGR_DEFAULT_SAMPLE_RATE))
    {
        f.close();
        g_server->send(404, "application/json", "{\"error\":\"not found\"}");
        return;
    }

    // 检查 Range 头
    String rangeHeader = g_server->header("Range");
    uint32_t rangeStart = 0;
    uint32_t rangeEnd   = fileSize - 1;
    bool isRange = false;

    if (rangeHeader.length() > 0) {
        if (!parseRangeHeader(rangeHeader, fileSize, &rangeStart, &rangeEnd)) {
            // 无效 Range → 416 Range Not Satisfiable
            f.close();
            g_server->sendHeader("Content-Range",
                                 String("bytes */") + String(fileSize));
            g_server->send(416, "text/plain", "Range Not Satisfiable");
            return;
        }
        isRange = true;
    }

    uint32_t contentLen = rangeEnd - rangeStart + 1;

    // 设置响应头
    g_server->sendHeader("Content-Type", "application/octet-stream");
    g_server->sendHeader("Accept-Ranges", "bytes");

    if (isRange) {
        char rangeHeaderVal[96];
        snprintf(rangeHeaderVal, sizeof(rangeHeaderVal),
                 "bytes %u-%u/%u",
                 (unsigned)rangeStart, (unsigned)rangeEnd, (unsigned)fileSize);
        g_server->sendHeader("Content-Range", rangeHeaderVal);
        g_server->sendHeader("Content-Length", String(contentLen));
        g_server->send(206, "application/octet-stream", "");
    } else {
        g_server->sendHeader("Content-Length", String(contentLen));
        g_server->send(200, "application/octet-stream", "");
    }

    // 流式发送数据: seek 到起始位置, 分块读取 + sendContent
    f.seek(rangeStart, SeekSet);
    uint32_t remaining = contentLen;
    uint8_t chunk[ECG_WIFI_CHUNK_BYTES];

    while (remaining > 0) {
        size_t toRead = (remaining < ECG_WIFI_CHUNK_BYTES)
                        ? (size_t)remaining : ECG_WIFI_CHUNK_BYTES;
        size_t n = f.read(chunk, toRead);
        if (n == 0) break;
        g_server->sendContent((const char*)chunk, n);
        remaining -= (uint32_t)n;
    }

    f.close();
}

/**
 * @brief DELETE /api/records/{id}
 *
 * 删除 .ecgr 文件并重建 records.idx 索引。
 * 成功: 200 {"deleted":true}
 * 文件不存在: 404 {"deleted":false}
 */
static void handleRecordsDelete(void)
{
    String idStr = g_server->pathArg(0);
    uint32_t id = (uint32_t)idStr.toInt();

    char path[64];
    getRecordPath(id, path, sizeof(path));

    // 检查文件是否存在 (先验证以避免删除后 idx 重建显示误导日志)
    if (!SPIFFS.exists(path)) {
        g_server->send(404, "application/json", "{\"deleted\":false}");
        return;
    }

    // 删除 .ecgr 文件
    bool removed = SPIFFS.remove(path);
    if (removed) {
        // 重建索引以同步 records.idx
        rebuildIdx();
        g_server->send(200, "application/json", "{\"deleted\":true}");
    } else {
        // 文件存在但删除失败 (例如文件被打开锁定)
        g_server->send(404, "application/json", "{\"deleted\":false}");
    }
}

/**
 * @brief 404 处理器: 所有未匹配路由
 */
static void handleNotFound(void)
{
    g_server->send(404, "application/json", "{\"error\":\"not found\"}");
}

/* ======================== 公开 API ======================== */

bool ecgWifiInit(void)
{
    if (g_server) return true; // 已初始化

    Serial.println("[WiFi] init: creating WebServer + registering routes...");

    g_server = new WebServer(ECG_WIFI_PORT);

    // 注册路由 (顺序重要: 更具体的模式先注册, 因为 WebServer
    // 按 handler 链表顺序匹配, 首个命中即返回)

    // GET /api/records — 记录列表
    g_server->on(Uri("/api/records"), HTTP_GET, handleRecordsList);

    // GET /api/records/{id}/meta — 记录元数据
    g_server->on(UriBraces("/api/records/{}/meta"), HTTP_GET, handleRecordsMeta);

    // GET /api/records/{id}/data — 原始数据流
    g_server->on(UriBraces("/api/records/{}/data"), HTTP_GET, handleRecordsData);

    // DELETE /api/records/{id} — 删除记录
    g_server->on(UriBraces("/api/records/{}"), HTTP_DELETE, handleRecordsDelete);

    // 404 兜底
    g_server->onNotFound(handleNotFound);

    Serial.println("[WiFi] routes registered: GET /api/records* + DELETE /api/records/{}");
    Serial.println("[WiFi] init OK (AP not started)");
    return true;
}

bool ecgWifiStart(void)
{
    if (g_wifiOn) return false;
    if (!g_server) {
        // 防御: 若未 init 则自动 init
        if (!ecgWifiInit()) return false;
    }

    // 构建 SSID: ESP32-ECG-<MAC 最后 2 字节大写十六进制>
    String mac = WiFi.macAddress();
    String ssid = "ESP32-ECG-";
    // MAC 格式: "XX:XX:XX:XX:XX:XX"
    // 取最后 2 字节: 位置 12-13 和 15-16 (跳过分隔符)
    // 提取第 5 和第 6 字节 (从末尾数)
    int len = mac.length();
    if (len >= 17) {
        ssid += mac.charAt(len - 5);
        ssid += mac.charAt(len - 4);
        ssid += mac.charAt(len - 2);
        ssid += mac.charAt(len - 1);
    } else {
        ssid += "0000"; // 安全降级
    }

    // 启动 SoftAP
    // 2026-08-10 排查: 手机/电脑均搜不到 beacon → 显式 AP 模式 + 显式配置
    // (softAP 前先 WiFi.mode(WIFI_AP), 避免 macAddress() 残留 STA 模式状态;
    //  显式信道 6 / 广播不隐藏 / 最大连接 4)
    // 注: 已移除 STA 扫描诊断 (scanDelete 后切 AP 疑残留状态破坏 beacon)
    WiFi.mode(WIFI_AP);

    /* 诊断序列 (DIAG SEQ 1, PR#1865 式): 先 WIFI_OFF 再 WIFI_AP, 带 500ms 延时。
     * 证据: WiFiManager PR#1865 (2026-05, arduino 2.0.17 世代 S3 实测) 确认
     * 快速模式切换会遗留不稳定状态致 S3/C3 softAP 不可见/不可连;
     * esp-idf#17055 (N16R8, IDF5.5) 同样指向初始化顺序敏感。 */
    if (s_diagSeqSlow) {
        WiFi.mode(WIFI_OFF);
        delay(500);
        WiFi.mode(WIFI_AP);
        delay(500);
    }

    WiFi.softAPConfig(IPAddress(192, 168, 4, 1), IPAddress(192, 168, 4, 1),
                      IPAddress(255, 255, 255, 0));
    bool apOk = WiFi.softAP(ssid.c_str(), ECG_WIFI_AP_PASSWORD, s_diagChannel, 0, 4);
    if (!apOk) {
        Serial.println("[WiFi] ERROR: softAP failed");
        return false;
    }
    /* 最大发射功率须在 AP 启动后设置 (此前在 softAP 前调用触发
     * 'Neither AP or STA has been started' 警告导致功率设置失败)
     * 2026-08-10 诊断: 默认 19.5dBm (原行为)。DIAG TXP 0 跳过 / 34=8.5dBm / 60=15dBm。
     * 证据: N16R8 上多例"降 TX power 修复" (ESPHome#6456 8.5dB, PlatformIO 社区
     * esp_wifi_set_max_tx_power(40)=10dBm, arduino-esp32#6551 WIFI_POWER_8_5dBm);
     * esp-idf#14008 提及 "S3 tx power 异常"。 */
    if (s_diagTxPower > 0) {
        WiFi.setTxPower((wifi_power_t)s_diagTxPower);
    }
    if (s_diagSeqSlow) {
        /* 服务端模式关闭节能 (AP 模式本身不休眠, 双保险) */
        WiFi.setSleep(false);
        delay(500);
    }
    Serial.printf("[WiFi] AP diag: mode=%d status=%d channel=%d mac=%s heap=%lu txp=%d seq=%d\n",
                  (int)WiFi.getMode(), (int)WiFi.status(),
                  WiFi.channel(), WiFi.softAPmacAddress().c_str(),
                  (unsigned long)ESP.getFreeHeap(), s_diagTxPower,
                  s_diagSeqSlow ? 1 : 0);

    // 启动 HTTP 服务器
    g_server->begin();

    g_wifiOn = true;

    Serial.printf("[WiFi] AP started: SSID=%s, IP=%s, port=%d\n",
                  ssid.c_str(),
                  WiFi.softAPIP().toString().c_str(),
                  ECG_WIFI_PORT);
    Serial.println("[WiFi] BLE NUS remains active (ESP32-S3 WiFi+BLE coexistence)");

    return true;
}

void ecgWifiStop(void)
{
    if (!g_wifiOn) return;

    Serial.println("[WiFi] stopping AP...");

    // 先停止 HTTP 服务器, 再关闭 AP
    g_server->stop();
    WiFi.softAPdisconnect(true);

    g_wifiOn = false;
    Serial.println("[WiFi] AP stopped");
}

bool ecgWifiIsOn(void)
{
    return g_wifiOn;
}

void ecgWifiProcess(void)
{
    if (g_wifiOn && g_server) {
        // handleClient() 在空闲时极廉价 (微秒级), 每迭代调用一次即可
        g_server->handleClient();
    }
}

/* ======================== 诊断配置实现 (DIAG 命令, 2026-08-10) ======================== */

void ecgWifiDiagSetTxPower(int v) { s_diagTxPower = v; }
void ecgWifiDiagSetChannel(int v) { s_diagChannel = v; }
void ecgWifiDiagSetSeqSlow(bool v) { s_diagSeqSlow = v; }
int  ecgWifiDiagGetTxPower(void) { return s_diagTxPower; }
int  ecgWifiDiagGetChannel(void) { return s_diagChannel; }
bool ecgWifiDiagGetSeqSlow(void) { return s_diagSeqSlow; }

/* ======================== STA 测试实现 (候选D, 2026-08-10) ======================== */

bool ecgWifiDiagStaConnect(const char* ssid, const char* pass)
{
    if (!ssid || !pass || strlen(ssid) == 0) return false;
    /* AP_STA 共存: 不停止 AP, 追加 STA 连接 (验证 WiFi TX/RX 全链路)。
     * WiFi.begin 非阻塞, 状态由 ecgWifiDiagStaStatus / ecgWifiDiagStaIp 查询。 */
    WiFi.mode(WIFI_AP_STA);
    WiFi.begin(ssid, pass);
    return true;
}

void ecgWifiDiagStaDisconnect(void)
{
    WiFi.disconnect();
    WiFi.mode(WIFI_AP);
}

int ecgWifiDiagStaStatus(void)
{
    return (int)WiFi.status();
}

void ecgWifiDiagStaIp(char* buf, size_t len)
{
    if (!buf || len == 0) return;
    IPAddress ip = WiFi.localIP();
    snprintf(buf, len, "%d.%d.%d.%d", ip[0], ip[1], ip[2], ip[3]);
}

int ecgWifiDiagGetMode(void)
{
    return (int)WiFi.getMode();
}
