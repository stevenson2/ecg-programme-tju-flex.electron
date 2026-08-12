/**
 * @file ecg_recorder.cpp
 * @brief ECG 录制模块实现 — SPIFFS 存储, 崩溃安全头部管理
 *
 * 录制格式: .ecgr (ecg_recorder_format.h), 250Hz int16 样本 + 异常位图。
 * 崩溃安全: REC_START 即写空头部, REC_STOP 回写最终值; 挂载时扫描删除损坏文件。
 */

#include "storage/ecg_recorder.h"
#include "storage/ecg_recorder_format.h"

#include <Arduino.h>
#include <SPIFFS.h>

/* ======================== 静态变量 ======================== */

/** 当前打开的录制文件 (fs::File 句柄) */
static fs::File g_recFile;

/** 是否正在录制 */
static bool g_isRecording = false;

/** 录制头部缓冲区 (32 字节) — 录制过程中实时更新, STOP 时回写 */
static uint8_t g_headerBuf[ECGR_HEADER_SIZE];

/** 录制起始 Unix 时间戳 (millis()/1000, 若未同步 NTP 则为上电秒数) */
static uint32_t g_startUnix = 0;

/** 已录制样本总数 */
static uint32_t g_totalSamples = 0;

/** 已录制完整秒数 (由 1Hz tick 递增) */
static uint32_t g_durationSec = 0;

/** 异常秒累计 */
static uint32_t g_abnormalSec = 0;

/** 写缓冲区 */
static uint8_t g_batchBuf[ECG_REC_BATCH_BYTES];
static uint16_t g_batchIdx = 0;

/** 自动录制开关 */
static bool g_autoRecord = false;

/** 自动录制: 连续正常秒计数器 */
static uint8_t g_consecutiveNormal = 0;

/** 当前秒异常标记 (由 ecgRecorderSetSecondAbnormal 设置) */
static bool g_currentSecondAbnormal = false;

/** 当前录制文件路径 (用于 STOP 时重开 + idx 条目) */
static char g_currentPath[64];

/** 记录总数 (挂载扫描时统计) */
static uint32_t g_recordCount = 0;

/* ======================== 内部辅助函数 ======================== */

/** SPIFFS 存储前缀 */
#define ECGR_BASE_PATH  "/ecgdata"

/**
 * @brief 将写缓冲区刷入文件
 *
 * 若 g_batchIdx > 0, 将缓冲区内容写入 g_recFile 并清零索引。
 * 写入失败时打印警告 (不中止录制 — 存储满时优雅降级)。
 */
static void flushBatch(void) {
    if (g_batchIdx == 0) return;
    if (!g_recFile) return;

    size_t written = g_recFile.write(g_batchBuf, g_batchIdx);
    if (written != g_batchIdx) {
        // 存储空间不足或 I/O 错误 — 丢弃这批数据, 继续录制
        Serial.printf("[ECGR] WARN: batch flush short write %u/%u\n",
                      (unsigned)written, (unsigned)g_batchIdx);
    }
    g_recFile.flush();
    g_batchIdx = 0;
}

/**
 * @brief 将单个 int16 样本以小端序写入批次缓冲区
 *
 * ESP32 为小端架构, memcpy 直接等效 LE 字节序;
 * 此处显式拆分保证跨平台一致性 (格式头可独立测试)。
 */
static void pushSampleToBatch(int16_t sample) {
    _ecgr_write16le(&g_batchBuf[g_batchIdx], (uint16_t)sample);
    g_batchIdx += 2;
    if (g_batchIdx >= ECG_REC_BATCH_BYTES) {
        flushBatch();
    }
}

/**
 * @brief 删除记录文件 (修复 f.name() 返回纯文件名导致的 remove 失败, 2026-08-12)
 *
 * SPIFFS 文件实际存储路径含虚拟目录前缀 ("/ecgdata/xxx"), 而 f.name() 返回
 * 纯文件名 ("ecg_rec_N.ecgr")。此前 deleteOldestRecord/scanAndCleanInvalid 直接
 * SPIFFS.remove(name) 在根目录找不到文件 → 删除静默失败 → 保留策略失效
 * (实测: "deleting oldest record" 连续 5 次删同一文件, 记录堆积, TH §40)。
 */
static void removeRecordFile(const char* name)
{
    if (!name || name[0] == '\0') return;
    char fullPath[80];
    if (name[0] == '/') {
        snprintf(fullPath, sizeof(fullPath), "%s", name);
    } else {
        snprintf(fullPath, sizeof(fullPath), ECGR_BASE_PATH "/%s", name);
    }
    SPIFFS.remove(fullPath);
}

/**
 * @brief 删除最旧记录 (按 startUnixTime 排序)
 *
 * 扫描 /ecgdata/*.ecgr, 解析头部取 startUnixTime,
 * 删除 startUnixTime 最小的文件, 同时更新 g_recordCount。
 */
static void deleteOldestRecord(void) {
    fs::File dir = SPIFFS.open(ECGR_BASE_PATH);
    if (!dir || !dir.isDirectory()) return;

    uint32_t oldestUnix = UINT32_MAX;
    char     oldestPath[64] = "";

    fs::File f = dir.openNextFile();
    while (f) {
        const char* name = f.name();
        if (name && strstr(name, ".ecgr")) {
            // 尝试读取头部获取 startUnix
            if (f.size() >= ECGR_HEADER_SIZE) {
                uint8_t hdr[ECGR_HEADER_SIZE];
                if (f.read(hdr, ECGR_HEADER_SIZE) == ECGR_HEADER_SIZE) {
                    uint32_t st = ecgrHeaderStartUnix(hdr);
                    if (st < oldestUnix) {
                        oldestUnix = st;
                        strncpy(oldestPath, name, sizeof(oldestPath) - 1);
                        oldestPath[sizeof(oldestPath) - 1] = '\0';
                    }
                }
            }
        }
        f.close();
        f = dir.openNextFile();
    }
    dir.close();

    if (oldestPath[0] != '\0') {
        Serial.printf("[ECGR] deleting oldest record: %s\n", oldestPath);
        removeRecordFile(oldestPath);
        if (g_recordCount > 0) g_recordCount--;
    }
}

/**
 * @brief 执行保留策略: 最多 10 条 + 空闲空间检查
 *
 * 1. 若记录数 > ECG_REC_KEEP_MAX, 删除最旧直到 ≤ 上限
 * 2. 若 SPIFFS 空闲空间 < ECG_REC_FREE_MIN_BYTES, 删除最旧直到 ≥ 阈值
 */
static void enforceRetention(void) {
    // 上限策略
    while (g_recordCount > ECG_REC_KEEP_MAX) {
        deleteOldestRecord();
    }

    // 空闲空间策略
    size_t totalBytes = SPIFFS.totalBytes();
    size_t usedBytes  = SPIFFS.usedBytes();
    while (g_recordCount > 0
           && (totalBytes - usedBytes) < ECG_REC_FREE_MIN_BYTES) {
        deleteOldestRecord();
        usedBytes = SPIFFS.usedBytes();
    }
}

/**
 * @brief 向 records.idx 追加一行条目
 *
 * 格式: <startUnix>,<dur>,<samples>,<abnSec>,<sizeBytes>\n
 */
static void appendIdxLine(uint32_t startUnix, uint32_t dur,
                          uint32_t samples, uint32_t abnSec,
                          uint32_t sizeBytes) {
    char line[128];
    int len = ecgrIdxLine(line, sizeof(line),
                          startUnix, dur, samples, abnSec, sizeBytes);
    if (len <= 0 || len >= (int)sizeof(line)) return;

    // 追加模式打开索引文件
    fs::File idx = SPIFFS.open(ECGR_BASE_PATH "/records.idx", "a");
    if (!idx) {
        Serial.println("[ECGR] WARN: cannot open records.idx for append");
        return;
    }
    idx.write((const uint8_t*)line, (size_t)len);
    idx.flush();
    idx.close();
}

/**
 * @brief 重建 records.idx (挂载时调用)
 *
 * 扫描 /ecgdata/*.ecgr, 读取每文件头部, 写入索引行。
 * 先删除旧 idx 文件再重建。
 */
static void rebuildIndex(void) {
    SPIFFS.remove(ECGR_BASE_PATH "/records.idx");

    fs::File idx = SPIFFS.open(ECGR_BASE_PATH "/records.idx", "w");
    if (!idx) {
        Serial.println("[ECGR] WARN: cannot create records.idx");
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
                if (f.read(hdr, ECGR_HEADER_SIZE) == ECGR_HEADER_SIZE) {
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

    g_recordCount = count;
    Serial.printf("[ECGR] index rebuilt: %u records\n", (unsigned)count);
}

/**
 * @brief 验证并清理损坏的 .ecgr 文件 (挂载时调用)
 *
 * 检查逻辑:
 *   - 魔数 "ECGR" 无效 → 删除
 *   - 版本号不匹配 → 删除
 *   - totalSamples == 0 → 删除 (崩溃中断的录制)
 *   - 文件大小与 header 计算不一致 → 删除
 *   - 缺少异常位图数据 (flags bit0 置位但文件不够大) → 删除
 */
static void scanAndCleanInvalid(void) {
    fs::File dir = SPIFFS.open(ECGR_BASE_PATH);
    if (!dir || !dir.isDirectory()) return;

    fs::File f = dir.openNextFile();
    while (f) {
        const char* name = f.name();
        if (name && strstr(name, ".ecgr")) {
            size_t fileSize = f.size();
            bool   invalid  = false;

            if (fileSize < ECGR_HEADER_SIZE) {
                invalid = true;
            } else {
                uint8_t hdr[ECGR_HEADER_SIZE];
                if (f.read(hdr, ECGR_HEADER_SIZE) != ECGR_HEADER_SIZE) {
                    invalid = true;
                } else {
                    // 魔数 + 版本校验
                    if (!ecgrHeaderValidate(hdr, ECG_REC_SAMPLE_RATE)) {
                        invalid = true;
                    } else {
                        uint32_t samples = ecgrHeaderTotalSamples(hdr);
                        uint32_t dur     = ecgrHeaderDurationSec(hdr);
                        uint8_t  flags   = hdr[ECGR_OFF_FLAGS];
                        bool     hasBmp  = (flags & ECGR_FLAG_HAS_ABNORMAL_BITMAP) != 0;

                        // totalSamples == 0 表示崩溃中断的录制
                        if (samples == 0) {
                            invalid = true;
                        } else {
                            // 文件大小一致性校验
                            uint32_t expected = ecgrFileSize(samples, dur, hasBmp);
                            if ((uint32_t)fileSize != expected) {
                                invalid = true;
                            }
                        }
                    }
                }
            }

            f.close();

            if (invalid) {
                Serial.printf("[ECGR] removing invalid record: %s (size=%u)\n",
                              name, (unsigned)fileSize);
                removeRecordFile(name);
            }
        } else {
            f.close();
        }
        f = dir.openNextFile();
    }
    dir.close();
}

/**
 * @brief 确保 /ecgdata 目录存在 (SPIFFS 扁平存储, 目录仅虚拟路径)
 */
static void ensureBaseDir(void) {
    // SPIFFS 无需显式创建目录, open 带路径即可
    // 此处仅验证挂载成功
}

/* ======================== 公开 API ======================== */

bool ecgRecorderInit(void) {
    Serial.println("[ECGR] init: mounting SPIFFS ecgdata partition...");

    // 挂载 ecgdata 分区 (formatOnFail=true: 首次使用自动格式化)
    if (!SPIFFS.begin(true, "/spiffs", 8, "ecgdata")) {
        Serial.println("[ECGR] ERROR: SPIFFS mount failed");
        return false;
    }

    size_t totalBytes = SPIFFS.totalBytes();
    size_t usedBytes  = SPIFFS.usedBytes();
    Serial.printf("[ECGR] SPIFFS mounted: total=%u KB, used=%u KB\n",
                  (unsigned)(totalBytes / 1024),
                  (unsigned)(usedBytes / 1024));

    ensureBaseDir();

    // 挂载扫描: 清理损坏文件
    scanAndCleanInvalid();

    // 重建索引文件
    rebuildIndex();

    // 重置内部状态
    g_isRecording    = false;
    g_totalSamples   = 0;
    g_durationSec    = 0;
    g_abnormalSec    = 0;
    g_batchIdx       = 0;
    g_startUnix      = 0;
    g_consecutiveNormal = 0;
    g_currentSecondAbnormal = false;
    g_currentPath[0] = '\0';

    Serial.println("[ECGR] init OK");
    return true;
}

bool ecgRecorderStart(void) {
    if (g_isRecording) {
        return false;
    }

    // 生成时间戳 (millis()/1000: 上电后秒数; 若 NTP 同步则替换为真实 epoch)
    g_startUnix = (uint32_t)(millis() / 1000);

    // 构建文件路径
    snprintf(g_currentPath, sizeof(g_currentPath),
             ECGR_BASE_PATH "/ecg_rec_%u.ecgr", (unsigned)g_startUnix);

    // 创建文件并写入初始头部
    g_recFile = SPIFFS.open(g_currentPath, "w");
    if (!g_recFile) {
        Serial.printf("[ECGR] ERROR: cannot create %s\n", g_currentPath);
        g_currentPath[0] = '\0';
        return false;
    }

    // 初始化头部: totalSamples=0, duration=0, abnormalSec=0
    ecgrHeaderInit(g_headerBuf, ECG_REC_SAMPLE_RATE, g_startUnix,
                   0, 0, 0, 0);
    size_t written = g_recFile.write(g_headerBuf, ECGR_HEADER_SIZE);
    if (written != ECGR_HEADER_SIZE) {
        Serial.println("[ECGR] ERROR: header write failed");
        g_recFile.close();
        SPIFFS.remove(g_currentPath);
        g_currentPath[0] = '\0';
        return false;
    }
    g_recFile.flush();

    // 重新以追加模式打开 (SPIFFS "w" 模式后续写入为覆盖语义? 稳妥用 "a")
    g_recFile.close();
    g_recFile = SPIFFS.open(g_currentPath, "a");
    if (!g_recFile) {
        Serial.println("[ECGR] ERROR: reopen for append failed");
        SPIFFS.remove(g_currentPath);
        g_currentPath[0] = '\0';
        return false;
    }

    g_isRecording  = true;
    g_totalSamples = 0;
    g_durationSec  = 0;
    g_abnormalSec  = 0;
    g_batchIdx     = 0;
    g_consecutiveNormal = 0;
    g_currentSecondAbnormal = false;

    Serial.printf("[ECGR] recording started: %s\n", g_currentPath);
    return true;
}

void ecgRecorderPushSample(int16_t sample) {
    if (!g_isRecording) return;
    pushSampleToBatch(sample);
    g_totalSamples++;
}

void ecgRecorderSetSecondAbnormal(bool abnormal) {
    g_currentSecondAbnormal = abnormal;

    if (!g_isRecording) {
        // 自动录制: 异常上升沿触发开始
        if (g_autoRecord && abnormal) {
            if (g_consecutiveNormal >= ECG_REC_AUTO_STOP_ABNORMAL_SECS
                || g_consecutiveNormal == 0) {
                // 异常上升沿: 开始录制
                Serial.println("[ECGR] auto-record: abnormal edge, starting...");
                ecgRecorderStart();
                g_consecutiveNormal = 0;
                // 重置此秒标记 (录制刚启动, 当前秒计为第 0 秒)
                g_currentSecondAbnormal = abnormal;
            }
        }
        if (!abnormal && g_autoRecord) {
            g_consecutiveNormal++;
        }
        return;
    }

    // ===== 录制中: 秒边界处理 =====

    // 递增秒计数
    g_durationSec++;

    // 写入异常位图字节 (1 = 异常秒, 0 = 正常秒)
    if (g_recFile) {
        uint8_t bmpByte = g_currentSecondAbnormal ? 1 : 0;
        g_recFile.write(&bmpByte, 1);
    }
    if (g_currentSecondAbnormal) {
        g_abnormalSec++;
    }

    // 更新头部中 flags 的异常位图标志 (稍后 STOP 时写入头部)
    if (g_abnormalSec > 0) {
        g_headerBuf[ECGR_OFF_FLAGS] |= ECGR_FLAG_HAS_ABNORMAL_BITMAP;
    }

    // 自动录制: 连续正常秒达到阈值 → 自动停止
    if (g_autoRecord) {
        if (!abnormal) {
            g_consecutiveNormal++;
            if (g_consecutiveNormal >= ECG_REC_AUTO_STOP_ABNORMAL_SECS) {
                Serial.println("[ECGR] auto-record: consecutive normal, stopping...");
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

    Serial.println("[ECGR] stopping recording...");

    // 1. 刷写剩余缓冲数据
    flushBatch();
    g_recFile.flush();

    // 2. 关闭追加模式句柄
    g_recFile.close();

    // 3. 计算最终头部的 durationSec (向上取整)
    uint32_t durFromSamples = (g_totalSamples + ECG_REC_SAMPLE_RATE - 1)
                              / ECG_REC_SAMPLE_RATE;
    uint32_t finalDur = (g_durationSec > durFromSamples)
                        ? g_durationSec : durFromSamples;

    // 4. 重新以读写模式打开, seek(0) 回写最终头部
    g_recFile = SPIFFS.open(g_currentPath, "r+");
    if (g_recFile) {
        ecgrHeaderUpdate(g_headerBuf, g_totalSamples, finalDur, g_abnormalSec);
        g_recFile.seek(0, SeekSet);
        size_t written = g_recFile.write(g_headerBuf, ECGR_HEADER_SIZE);
        if (written != ECGR_HEADER_SIZE) {
            Serial.println("[ECGR] WARN: final header write truncated");
        }
        g_recFile.flush();
        g_recFile.close();
    } else {
        // "r+" 不支持时降级: 重新创建临时文件, 写入头部 + 追加原数据
        // 简化处理: 仅打印告警, header 可能不完整
        Serial.println("[ECGR] WARN: cannot reopen for header rewrite, "
                       "file may have empty header");
    }

    // 5. 计算最终文件大小
    uint32_t fileSize = ecgrFileSize(g_totalSamples, finalDur,
                                     (g_abnormalSec > 0));

    // 6. 追加索引行
    appendIdxLine(g_startUnix, finalDur, g_totalSamples,
                  g_abnormalSec, fileSize);
    g_recordCount++;

    // 7. 保留策略
    enforceRetention();

    Serial.printf("[ECGR] recording stopped: %u samples, %u sec, "
                  "%u abnormal sec, size=%u B\n",
                  (unsigned)g_totalSamples, (unsigned)finalDur,
                  (unsigned)g_abnormalSec, (unsigned)fileSize);

    // 8. 重置状态
    g_isRecording  = false;
    g_totalSamples = 0;
    g_durationSec  = 0;
    g_abnormalSec  = 0;
    g_batchIdx     = 0;
    g_startUnix    = 0;
    g_currentPath[0] = '\0';
    g_consecutiveNormal = 0;

    return true;
}

bool ecgRecorderIsRecording(void) {
    return g_isRecording;
}

int ecgRecorderList(char* outBuf, int bufLen) {
    if (!outBuf || bufLen <= 0) return -1;

    fs::File idx = SPIFFS.open(ECGR_BASE_PATH "/records.idx", "r");
    if (!idx) {
        // 无索引文件 → 返回空字符串
        if (bufLen > 0) outBuf[0] = '\0';
        return 0;
    }

    size_t fileSize = idx.size();
    if (fileSize >= (size_t)(bufLen - 1)) {
        // 缓冲区不足
        idx.close();
        return -1;
    }

    int totalRead = 0;
    if (fileSize > 0) {
        totalRead = (int)idx.read((uint8_t*)outBuf, fileSize);
        if (totalRead < (int)fileSize) {
            idx.close();
            return -1;
        }
        outBuf[totalRead] = '\0';
    } else {
        outBuf[0] = '\0';
    }
    idx.close();
    return totalRead;
}

uint32_t ecgRecorderRecordCount(void) {
    return g_recordCount;
}

uint32_t ecgRecorderCurrentRecordStart(void) {
    return g_isRecording ? g_startUnix : 0;
}

uint32_t ecgRecorderCurrentDurationSec(void) {
    return g_isRecording ? g_durationSec : 0;
}

void ecgRecorderSetAutoRecord(bool enable) {
    g_autoRecord = enable;
    if (!enable) {
        g_consecutiveNormal = 0;
    }
}

bool ecgRecorderAutoRecordEnabled(void) {
    return g_autoRecord;
}

void ecgRecorderReset(void) {
    if (g_isRecording) {
        ecgRecorderStop();
    }
    g_totalSamples = 0;
    g_durationSec  = 0;
    g_abnormalSec  = 0;
    g_batchIdx     = 0;
    g_startUnix    = 0;
    g_currentPath[0] = '\0';
    g_consecutiveNormal = 0;
    g_currentSecondAbnormal = false;
}
