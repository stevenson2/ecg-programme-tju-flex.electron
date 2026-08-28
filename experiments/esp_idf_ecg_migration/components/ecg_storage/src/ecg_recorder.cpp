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
static bool g_currentSecondAbnormal = false;
static char g_currentPath[128];
static uint32_t g_recordCount = 0;

static bool g_storageMounted = false;

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
    char full[160];
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
    char oldestPath[160] = "";
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (!nameIsEcgr(e->d_name)) continue;
        char full[160];
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
        char full[160];
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
        char full[160];
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
        if (g_autoRecord && abnormal) {
            if (g_consecutiveNormal >= ECG_REC_AUTO_STOP_ABNORMAL_SECS || g_consecutiveNormal == 0) {
                printf("[ECGR] auto-record: abnormal edge, starting...\n");
                ecgRecorderStart();
                g_consecutiveNormal = 0;
            }
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

bool ecgRecorderStop(void) {
    if (!g_isRecording) return false;
    g_isRecording = false;
    uint32_t durFromSamples = (g_totalSamples + ECG_REC_SAMPLE_RATE - 1) / ECG_REC_SAMPLE_RATE;
    uint32_t finalDur = (g_durationSec > durFromSamples) ? g_durationSec : durFromSamples;
    bool ok = false;
    {
        uint8_t hdr[ECGR_HEADER_SIZE];
        ecgrHeaderInit(hdr, ECG_REC_SAMPLE_RATE, g_startUnix, g_totalSamples, finalDur, g_abnormalSec, 0);
        FILE *f = fopen(g_currentPath, "wb");
        if (f) {
            fwrite(hdr, 1, ECGR_HEADER_SIZE, f);
            if (g_totalSamples > 0 && g_psramBuf) fwrite(g_psramBuf, 1, (size_t)g_totalSamples * 2, f);
            if (g_abnormalSec > 0 && g_bmpBuf) {
                if (g_bmpCap < finalDur) {
                    uint8_t *p = (uint8_t *)ecgrRealloc(g_bmpBuf, finalDur);
                    if (p) { g_bmpBuf = p; g_bmpCap = finalDur; }
                }
                if (g_bmpBuf && g_bmpCap >= finalDur) {
                    if (finalDur > g_durationSec) memset(g_bmpBuf + g_durationSec, 0, finalDur - g_durationSec);
                    fwrite(g_bmpBuf, 1, finalDur, f);
                }
            }
            fclose(f);
            ok = true;
        }
    }
    if (!ok) {
        printf("[ECGR] stop failed: cannot write %s\n", g_currentPath);
        free(g_psramBuf); free(g_bmpBuf); g_psramBuf = NULL; g_bmpBuf = NULL;
        g_psramCap = 0; g_bmpCap = 0; g_totalSamples = 0; g_durationSec = 0;
        g_abnormalSec = 0; g_startUnix = 0; g_currentPath[0] = '\0';
        return false;
    }
    uint32_t fileSize = ecgrFileSize(g_totalSamples, finalDur, g_abnormalSec > 0);
    appendIdxLine(g_startUnix, finalDur, g_totalSamples, g_abnormalSec, fileSize);
    g_recordCount++;
    enforceRetention(g_startUnix);
    rebuildIndex();
    printf("[ECGR] stopped: %u samples, %u sec, %u abn, size=%u\n",
           (unsigned)g_totalSamples, (unsigned)finalDur, (unsigned)g_abnormalSec, (unsigned)fileSize);
    free(g_psramBuf); free(g_bmpBuf); g_psramBuf = NULL; g_bmpBuf = NULL;
    g_psramCap = 0; g_bmpCap = 0;
    g_totalSamples = 0; g_durationSec = 0; g_abnormalSec = 0;
    g_startUnix = 0; g_currentPath[0] = '\0';
    return true;
}

bool ecgRecorderIsRecording(void) { return g_isRecording; }

int ecgRecorderList(char *outBuf, int bufLen) {
    if (!outBuf || bufLen <= 0) return -1;
    char idxPath[160];
    snprintf(idxPath, sizeof(idxPath), ECGR_BASE_PATH "/records.idx");
    FILE *f = fopen(idxPath, "rb");
    if (!f) { outBuf[0] = '\0'; return 0; }
    size_t n = fread(outBuf, 1, (size_t)(bufLen - 1), f);
    fclose(f);
    outBuf[n] = '\0';
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
