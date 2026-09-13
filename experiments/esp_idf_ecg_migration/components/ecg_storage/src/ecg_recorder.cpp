/**
 * @file ecg_recorder.cpp
 * @brief ECG 录制模块（ESP-IDF POSIX/SPIFFS 移植版）
 *
 * 与 Arduino 版保持同一公开 API 与 ECGR 文件格式，录制期间使用 PSRAM 缓冲，
 * STOP 时一次性写入文件，避免录制中 SPIFFS 阻塞。
 */
#include "storage/ecg_recorder.h"
#include "storage/ecg_recorder_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include "esp_timer.h"
#include "esp_spiffs.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"

#define ECGR_BASE_PATH "/spiffs/ecgdata"

static bool g_isRecording = false;
static uint32_t g_startUnix = 0;
static uint32_t g_totalSamples = 0;
static uint32_t g_durationSec = 0;
static uint32_t g_abnormalSec = 0;

static uint8_t *g_psramBuf = NULL;
static size_t g_psramCap = 0;
static uint8_t *g_bmpBuf = NULL;
static size_t g_bmpCap = 0;

static bool g_autoRecord = false;
static uint8_t g_consecutiveNormal = 0;
static uint8_t g_rearmCooldown = 0;   /* M1: 停止后自动录制的再武装冷却 (秒) */
static bool g_currentSecondAbnormal = false;
static char g_currentPath[128];
static uint32_t g_recordCount = 0;

static bool g_storageMounted = false;

/* ---- Round-G: 录制终结异步化 ----
 * STOP 的写文件+保留策略+索引重建曾阻塞采样循环 0.8-2.2s (§112)。
 * 现在 STOP 只把会话所有权 (缓冲+元数据) 打包入队立即返回;
 * worker (核 1) 带互斥锁执行全部 SPIFFS 文件操作。
 * g_ioMutex 同时覆盖 Start 的 stat 循环与 REC_LIST 读 —— SPIFFS 无内核级
 * 锁, 所有录制路径的文件操作必须串行。PushSample 只写 RAM, 不取锁。 */
typedef struct {
    char path[128];
    uint8_t *psram;
    size_t psramLen;
    uint8_t *bmp;
    size_t bmpLen;
    uint32_t startUnix;
    uint32_t totalSamples;
    uint32_t durationSec;
    uint32_t abnormalSec;
} EcgrFinalize;

static QueueHandle_t g_finQueue = NULL;
static TaskHandle_t g_workerTask = NULL;
static SemaphoreHandle_t g_ioMutex = NULL;
static volatile uint32_t g_finPending = 0;

static void ecgrWorkerInit(void);   /* Round-G: 定义于文件后部 */

static void *ecgrAlloc(size_t size) {
    void *p = heap_caps_malloc_prefer(size, 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT, MALLOC_CAP_8BIT);
    if (!p) p = malloc(size);
    return p;
}

static void *ecgrRealloc(void *ptr, size_t size) {
    void *p = heap_caps_realloc(ptr, size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!p) p = realloc(ptr, size);
    return p;
}

static bool nameIsEcgr(const char *name) {
    return name && strstr(name, ".ecgr") != NULL;
}

static void makeFullPath(char *out, size_t outLen, const char *name) {
    if (name[0] == '/') {
        snprintf(out, outLen, "%s", name);
    } else {
        snprintf(out, outLen, ECGR_BASE_PATH "/%s", name);
    }
}

static void removeRecordFile(const char *name) {
    if (!name || name[0] == '\0') return;
    char full[320];
    makeFullPath(full, sizeof(full), name);
    remove(full);
}

static uint32_t readHeaderStart(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return UINT32_MAX;
    uint8_t hdr[ECGR_HEADER_SIZE];
    size_t rd = fread(hdr, 1, ECGR_HEADER_SIZE, f);
    fclose(f);
    if (rd != ECGR_HEADER_SIZE) return UINT32_MAX;
    return ecgrHeaderStartUnix(hdr);
}

static void deleteOldestRecord(uint32_t protectUnix) {
    DIR *d = opendir(ECGR_BASE_PATH);
    if (!d) return;
    uint32_t oldestUnix = UINT32_MAX;
    char oldestPath[320] = "";
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (!nameIsEcgr(e->d_name)) continue;
        char full[320];
        makeFullPath(full, sizeof(full), e->d_name);
        uint32_t st = readHeaderStart(full);
        if (st == UINT32_MAX) continue;
        if (st != protectUnix && st < oldestUnix) {
            oldestUnix = st;
            snprintf(oldestPath, sizeof(oldestPath), "%s", full);
        }
    }
    closedir(d);
    if (oldestPath[0]) {
        printf("[ECGR] delete oldest: %s\n", oldestPath);
        remove(oldestPath);
    }
}

static void scanAndCleanInvalid(void) {
    DIR *d = opendir(ECGR_BASE_PATH);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (!nameIsEcgr(e->d_name)) continue;
        char full[320];
        makeFullPath(full, sizeof(full), e->d_name);
        FILE *f = fopen(full, "rb");
        if (!f) {
            remove(full);
            continue;
        }
        uint8_t hdr[ECGR_HEADER_SIZE];
        size_t rd = fread(hdr, 1, ECGR_HEADER_SIZE, f);
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fclose(f);
        if (rd != ECGR_HEADER_SIZE || sz < ECGR_HEADER_SIZE ||
            !ecgrHeaderValidate(hdr, ECG_REC_SAMPLE_RATE)) {
            remove(full);
            continue;
        }
        uint32_t samples = ecgrHeaderTotalSamples(hdr);
        uint32_t dur = ecgrHeaderDurationSec(hdr);
        bool hasBmp = (hdr[ECGR_OFF_FLAGS] & ECGR_FLAG_HAS_ABNORMAL_BITMAP) != 0;
        if (samples == 0 || (uint32_t)sz != ecgrFileSize(samples, dur, hasBmp)) {
            remove(full);
        }
    }
    closedir(d);
}

static void rebuildIndex(void) {
    char idxPath[160];
    snprintf(idxPath, sizeof(idxPath), ECGR_BASE_PATH "/records.idx");
    remove(idxPath);

    DIR *d = opendir(ECGR_BASE_PATH);
    if (!d) return;
    FILE *idx = fopen(idxPath, "w");
    if (!idx) {
        closedir(d);
        return;
    }
    uint32_t count = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (!nameIsEcgr(e->d_name)) continue;
        char full[320];
        makeFullPath(full, sizeof(full), e->d_name);
        FILE *f = fopen(full, "rb");
        if (!f) continue;
        uint8_t hdr[ECGR_HEADER_SIZE];
        if (fread(hdr, 1, ECGR_HEADER_SIZE, f) == ECGR_HEADER_SIZE) {
            uint32_t samples = ecgrHeaderTotalSamples(hdr);
            uint32_t dur = ecgrHeaderDurationSec(hdr);
            uint32_t abn = ecgrHeaderAbnormalSec(hdr);
            uint32_t st = ecgrHeaderStartUnix(hdr);
            fseek(f, 0, SEEK_END);
            uint32_t sz = (uint32_t)ftell(f);
            char line[128];
            int len = ecgrIdxLine(line, sizeof(line), st, dur, samples, abn, sz);
            if (len > 0) fwrite(line, 1, (size_t)len, idx);
            count++;
        }
        fclose(f);
    }
    closedir(d);
    fclose(idx);
    g_recordCount = count;
    printf("[ECGR] index rebuilt: %u records\n", (unsigned)count);
}

static void appendIdxLine(uint32_t startUnix, uint32_t dur, uint32_t samples,
                          uint32_t abnSec, uint32_t sizeBytes) {
    char idxPath[160];
    snprintf(idxPath, sizeof(idxPath), ECGR_BASE_PATH "/records.idx");
    FILE *idx = fopen(idxPath, "a");
    if (!idx) return;
    char line[128];
    int len = ecgrIdxLine(line, sizeof(line), startUnix, dur, samples, abnSec, sizeBytes);
    if (len > 0) fwrite(line, 1, (size_t)len, idx);
    fclose(idx);
}

static void enforceRetention(uint32_t protectUnix) {
    while (g_recordCount > ECG_REC_KEEP_MAX) {
        uint32_t before = g_recordCount;
        deleteOldestRecord(protectUnix);
        /* 文件删除后计数需要重扫 */
        g_recordCount = 0;
        DIR *d = opendir(ECGR_BASE_PATH);
        if (d) {
            struct dirent *e;
            while ((e = readdir(d)) != NULL) if (nameIsEcgr(e->d_name)) g_recordCount++;
            closedir(d);
        }
        if (g_recordCount >= before) break; /* 防死循环 */
    }
}

static void pushSampleToBuffer(int16_t sample) {
    size_t need = (size_t)g_totalSamples * 2 + 2;
    if (g_psramBuf == NULL || need > g_psramCap) {
        size_t newCap = (g_psramCap == 0) ? (64 * 1024) : g_psramCap;
        while (newCap < need) newCap *= 2;
        uint8_t *p = (uint8_t *)ecgrRealloc(g_psramBuf, newCap);
        if (!p) {
            printf("[ECGR] ERROR: buffer alloc failed, sample dropped\n");
            return;
        }
        g_psramBuf = p;
        g_psramCap = newCap;
    }
    _ecgr_write16le(&g_psramBuf[g_totalSamples * 2], (uint16_t)sample);
    g_totalSamples++;
}

bool ecgRecorderInit(void) {
    ecgrWorkerInit();   /* Round-G: 先起终结 worker (幂等) */
    if (!g_storageMounted) {
        esp_vfs_spiffs_conf_t conf = {
            .base_path = "/spiffs",
            .partition_label = "storage",
            .max_files = 10,
            .format_if_mount_failed = true,
        };
        esp_err_t err = esp_vfs_spiffs_register(&conf);
        if (err != ESP_OK) {
            printf("[ECGR] ERROR: spiffs mount failed (%s)\n", esp_err_to_name(err));
            return false;
        }
        g_storageMounted = true;
    }
    mkdir(ECGR_BASE_PATH, 0755);
    scanAndCleanInvalid();
    rebuildIndex();
    enforceRetention(0);   /* Round-G: 保留策略仅在开机维护 (§113) */
    if (g_psramBuf) { free(g_psramBuf); g_psramBuf = NULL; g_psramCap = 0; }
    if (g_bmpBuf) { free(g_bmpBuf); g_bmpBuf = NULL; g_bmpCap = 0; }
    g_isRecording = false;
    g_totalSamples = 0;
    g_durationSec = 0;
    g_abnormalSec = 0;
    g_startUnix = 0;
    g_currentPath[0] = '\0';
    g_consecutiveNormal = 0;
    g_currentSecondAbnormal = false;
    printf("[ECGR] init OK\n");
    return true;
}

bool ecgRecorderStart(void) {
    if (g_isRecording) return false;
    uint32_t startBase = (uint32_t)(esp_timer_get_time() / 1000000ULL);
    /* Round-G: stat 循环触 SPIFFS, 与 worker 终结互斥; 非阻塞拿锁 ——
     * 撞上终结中则本秒放弃 (自动路径下秒重试), 绝不阻塞采样循环 */
    if (xSemaphoreTake(g_ioMutex, 0) != pdTRUE) return false;
    g_startUnix = startBase;
    snprintf(g_currentPath, sizeof(g_currentPath),
             ECGR_BASE_PATH "/ecg_rec_%u.ecgr", (unsigned)g_startUnix);
    for (int i = 0; i < 1000; i++) {
        struct stat st;
        if (stat(g_currentPath, &st) != 0) break;
        g_startUnix++;
        snprintf(g_currentPath, sizeof(g_currentPath),
                 ECGR_BASE_PATH "/ecg_rec_%u.ecgr", (unsigned)g_startUnix);
    }
    xSemaphoreGive(g_ioMutex);
    g_psramBuf = (uint8_t *)ecgrAlloc(64 * 1024);
    g_bmpBuf = (uint8_t *)ecgrAlloc(1024);
    if (!g_psramBuf || !g_bmpBuf) {
        if (g_psramBuf) { free(g_psramBuf); g_psramBuf = NULL; }
        if (g_bmpBuf) { free(g_bmpBuf); g_bmpBuf = NULL; }
        g_psramCap = 0; g_bmpCap = 0;
        g_currentPath[0] = '\0';
        return false;
    }
    g_psramCap = 64 * 1024;
    g_bmpCap = 1024;
    g_isRecording = true;
    g_totalSamples = 0;
    g_durationSec = 0;
    g_abnormalSec = 0;
    g_consecutiveNormal = 0;
    g_currentSecondAbnormal = false;
    printf("[ECGR] recording started: %s\n", g_currentPath);
    return true;
}

void ecgRecorderPushSample(int16_t sample) {
    if (g_isRecording) pushSampleToBuffer(sample);
}

void ecgRecorderSetSecondAbnormal(bool abnormal) {
    g_currentSecondAbnormal = abnormal;
    if (!g_isRecording) {
        /* M1 修复: 再武装改为时间冷却 (停止后 5 秒), 逐秒递减不受 abnormal 冻结。
         * 原条件 "连续正常>=5s 或计数==0" 在持续 abnormal (如 M1 报警擎住 >=30s)
         * 下死锁: 计数冻结在 <5, 录制永远不再自动启动 (2026-09-12 平线验收发现)。 */
        if (g_rearmCooldown > 0) g_rearmCooldown--;
        if (g_autoRecord && abnormal && g_rearmCooldown == 0) {
            printf("[ECGR] auto-record: abnormal edge, starting...\n");
            ecgRecorderStart();
            g_consecutiveNormal = 0;
        }
        if (!abnormal && g_autoRecord) g_consecutiveNormal++;
        return;
    }
    g_durationSec++;
    {
        size_t need = (size_t)g_durationSec;
        if (g_bmpBuf == NULL || need > g_bmpCap) {
            size_t newCap = (g_bmpCap == 0) ? 1024 : g_bmpCap;
            while (newCap < need) newCap *= 2;
            uint8_t *p = (uint8_t *)ecgrRealloc(g_bmpBuf, newCap);
            if (p) { g_bmpBuf = p; g_bmpCap = newCap; }
        }
        if (g_bmpBuf && need <= g_bmpCap) {
            g_bmpBuf[need - 1] = g_currentSecondAbnormal ? 1 : 0;
        }
    }
    if (g_currentSecondAbnormal) g_abnormalSec++;
    if (g_autoRecord) {
        if (!abnormal) {
            g_consecutiveNormal++;
            if (g_consecutiveNormal >= ECG_REC_AUTO_STOP_ABNORMAL_SECS) {
                printf("[ECGR] auto-record: consecutive normal, stopping...\n");
                ecgRecorderStop();
                g_consecutiveNormal = 0;
            }
        } else {
            g_consecutiveNormal = 0;
        }
    }
}

/* Round-G: 终结一个会话 (写文件+索引+保留策略)。worker 任务或队列满时的
 * 主线程回退路径共用。调用方必须持有 g_ioMutex (回退路径自行取锁)。 */
static void finalizeItem(EcgrFinalize *it) {
    uint64_t t0 = esp_timer_get_time();
    bool ok = false;
    {
        FILE *f = fopen(it->path, "wb");
        if (f) {
            uint8_t hdr[ECGR_HEADER_SIZE];
            uint32_t durFromSamples =
                (it->totalSamples + ECG_REC_SAMPLE_RATE - 1) / ECG_REC_SAMPLE_RATE;
            uint32_t finalDur = (it->durationSec > durFromSamples)
                                    ? it->durationSec : durFromSamples;
            ecgrHeaderInit(hdr, ECG_REC_SAMPLE_RATE, it->startUnix,
                           it->totalSamples, finalDur, it->abnormalSec, 0);
            fwrite(hdr, 1, ECGR_HEADER_SIZE, f);
            if (it->totalSamples > 0 && it->psram)
                fwrite(it->psram, 1, (size_t)it->totalSamples * 2, f);
            if (it->abnormalSec > 0 && it->bmp) {
                if (it->bmpLen < finalDur) {
                    fwrite(it->bmp, 1, it->bmpLen, f);
                    /* 尾部零填充到 finalDur */
                    for (size_t i = it->bmpLen; i < finalDur; i++) fputc(0, f);
                } else {
                    fwrite(it->bmp, 1, finalDur, f);
                }
            }
            fclose(f);
            ok = true;
        }
    }
    if (ok) {
        uint32_t fileSize = ecgrFileSize(it->totalSamples, it->durationSec,
                                         it->abnormalSec > 0);
        appendIdxLine(it->startUnix, it->durationSec, it->totalSamples,
                      it->abnormalSec, fileSize);
        /* Round-G: 保留删除/索引重建挪到仅 Init 时执行 (§113)。
         * 实测 SPIFFS 删除+重建在 flash 擦写期间全局禁 cache, 两个核一起
         * 冻结 ~1.1s —— 任务卸载救不了, 只能减少擦写次数。
         * appendIdxLine 是增量追加, records.idx 对 REC_LIST/下载保持最新;
         * 保留策略推迟到下次 Init (开机), SPIFFS 4MB 对 2.5KB/记录可攒
         * ~1600 条, 单次上电会话内不会满。 */
        printf("[ECGR] finalized %s (%u samples, %u sec, %u abn, %u B) in %ums\n",
               it->path, (unsigned)it->totalSamples, (unsigned)it->durationSec,
               (unsigned)it->abnormalSec, (unsigned)fileSize,
               (unsigned)((esp_timer_get_time() - t0) / 1000));
    } else {
        printf("[ECGR] ERROR: finalize write failed: %s\n", it->path);
    }
    free(it->psram);
    free(it->bmp);
}

static void ecgrWorkerTask(void *arg) {
    (void)arg;
    EcgrFinalize *it = NULL;
    while (true) {
        if (xQueueReceive(g_finQueue, &it, portMAX_DELAY) == pdTRUE && it) {
            if (xSemaphoreTake(g_ioMutex, pdMS_TO_TICKS(10000)) == pdTRUE) {
                finalizeItem(it);
                xSemaphoreGive(g_ioMutex);
            } else {
                printf("[ECGR] ERROR: io mutex timeout, recording lost\n");
                free(it->psram);
                free(it->bmp);
            }
            free(it);
            g_finPending = g_finPending - 1;
        }
    }
}

static void ecgrWorkerInit(void) {
    if (g_workerTask) return;
    g_ioMutex = xSemaphoreCreateMutex();
    g_finQueue = xQueueCreate(4, sizeof(EcgrFinalize *));
    if (!g_ioMutex || !g_finQueue) return;
    xTaskCreatePinnedToCore(ecgrWorkerTask, "ecgr_worker", 4096, NULL, 3,
                            &g_workerTask, 1);
}

bool ecgRecorderStop(void) {
    if (!g_isRecording) return false;
    g_isRecording = false;
    uint32_t durFromSamples = (g_totalSamples + ECG_REC_SAMPLE_RATE - 1) / ECG_REC_SAMPLE_RATE;
    uint32_t finalDur = (g_durationSec > durFromSamples) ? g_durationSec : durFromSamples;
    (void)finalDur;
    EcgrFinalize *it = (EcgrFinalize *)malloc(sizeof(EcgrFinalize));
    if (!it) return false;
    it->path[0] = '\0';
    snprintf(it->path, sizeof(it->path), "%s", g_currentPath);
    it->psram = g_psramBuf; it->psramLen = g_psramCap;
    it->bmp = g_bmpBuf; it->bmpLen = g_bmpCap;
    it->startUnix = g_startUnix;
    it->totalSamples = g_totalSamples;
    it->durationSec = g_durationSec;
    it->abnormalSec = g_abnormalSec;
    g_psramBuf = NULL; g_psramCap = 0;
    g_bmpBuf = NULL; g_bmpCap = 0;
    g_totalSamples = 0; g_durationSec = 0; g_abnormalSec = 0;
    g_startUnix = 0; g_currentPath[0] = '\0';
    g_rearmCooldown = ECG_REC_AUTO_STOP_ABNORMAL_SECS;   /* M1: 时间冷却式再武装 */
    if (xQueueSend(g_finQueue, &it, 0) != pdTRUE) {
        /* 队列满 (4 个未终结会话): 回退为同步终结, 保数据不保节拍 */
        if (xSemaphoreTake(g_ioMutex, pdMS_TO_TICKS(10000)) == pdTRUE) {
            finalizeItem(it);
            xSemaphoreGive(g_ioMutex);
        }
        free(it);
    } else {
        g_finPending = g_finPending + 1;
    }
    return true;
}

bool ecgRecorderIsRecording(void) { return g_isRecording; }

int ecgRecorderList(char *outBuf, int bufLen) {
    if (!outBuf || bufLen <= 0) return -1;
    char idxPath[160];
    snprintf(idxPath, sizeof(idxPath), ECGR_BASE_PATH "/records.idx");
    /* Round-G: records.idx 由 worker 维护, 读时与终结互斥 */
    if (g_ioMutex && xSemaphoreTake(g_ioMutex, pdMS_TO_TICKS(5000)) != pdTRUE) {
        return -1;
    }
    FILE *f = fopen(idxPath, "rb");
    if (!f) { outBuf[0] = '\0'; if (g_ioMutex) xSemaphoreGive(g_ioMutex); return 0; }
    size_t n = fread(outBuf, 1, (size_t)(bufLen - 1), f);
    fclose(f);
    outBuf[n] = '\0';
    if (g_ioMutex) xSemaphoreGive(g_ioMutex);
    return (int)n;
}

uint32_t ecgRecorderRecordCount(void) { return g_recordCount; }

void ecgRecorderRefreshCount(void) {
    g_recordCount = 0;
    DIR *d = opendir(ECGR_BASE_PATH);
    if (d) {
        struct dirent *e;
        while ((e = readdir(d)) != NULL) if (nameIsEcgr(e->d_name)) g_recordCount++;
        closedir(d);
    }
}

uint32_t ecgRecorderCurrentRecordStart(void) { return g_isRecording ? g_startUnix : 0; }
uint32_t ecgRecorderCurrentDurationSec(void) { return g_isRecording ? g_durationSec : 0; }
void ecgRecorderSetAutoRecord(bool enable) { g_autoRecord = enable; if (!enable) g_consecutiveNormal = 0; }
bool ecgRecorderAutoRecordEnabled(void) { return g_autoRecord; }

void ecgRecorderReset(void) {
    if (g_isRecording) ecgRecorderStop();
    g_totalSamples = 0; g_durationSec = 0; g_abnormalSec = 0;
    g_startUnix = 0; g_currentPath[0] = '\0';
    g_consecutiveNormal = 0; g_currentSecondAbnormal = false;
    if (g_psramBuf) { free(g_psramBuf); g_psramBuf = NULL; g_psramCap = 0; }
    if (g_bmpBuf) { free(g_bmpBuf); g_bmpBuf = NULL; g_bmpCap = 0; }
}
