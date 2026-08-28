/**
 * @file ecg_recorder_format.h
 * @brief ECG 记录文件格式定义（纯 C++，无 Arduino/SPIFFS 依赖）
 *
 * 文件扩展名: .ecgr，存储于 SPIFFS /ecgdata/ 目录
 * 32 字节头部 + int16 样本流 + 可选异常位图
 *
 * 该头文件可供固件、PC 端解码器、云端 mock 构建共同引用，
 * 不依赖任何平台特定头文件，仅需 stdint.h / string.h。
 */

#ifndef ECG_RECORDER_FORMAT_H
#define ECG_RECORDER_FORMAT_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ======================== 魔数与版本 ======================== */
#define ECGR_MAGIC      "ECGR"          /* 文件标识 */
#define ECGR_MAGIC_LEN  4
#define ECGR_VERSION    1               /* 格式版本号 */
#define ECGR_HEADER_SIZE 32            /* 头部字节数 */

/* 标志位 */
#define ECGR_FLAG_HAS_ABNORMAL_BITMAP  0x01  /* bit0: 包含异常位图 */

/* 默认采样率 */
#define ECGR_DEFAULT_SAMPLE_RATE  250

/* 头部字段偏移量 (字节) */
#define ECGR_OFF_MAGIC          0
#define ECGR_OFF_VERSION        4
#define ECGR_OFF_FLAGS          5
#define ECGR_OFF_SAMPLE_RATE    6
#define ECGR_OFF_START_UNIX     10
#define ECGR_OFF_DURATION_SEC   14
#define ECGR_OFF_TOTAL_SAMPLES  18
#define ECGR_OFF_ABNORMAL_SEC   22
#define ECGR_OFF_RESERVED       26

/* ======================== 内部辅助: 小端读写 ======================== */

/** 将 uint32 以小端序写入缓冲区 */
static inline void _ecgr_write32le(uint8_t* dst, uint32_t val) {
    dst[0] = (uint8_t)(val);
    dst[1] = (uint8_t)(val >> 8);
    dst[2] = (uint8_t)(val >> 16);
    dst[3] = (uint8_t)(val >> 24);
}

/** 从缓冲区以小端序读取 uint32 */
static inline uint32_t _ecgr_read32le(const uint8_t* src) {
    return ((uint32_t)src[0])
         | ((uint32_t)src[1] << 8)
         | ((uint32_t)src[2] << 16)
         | ((uint32_t)src[3] << 24);
}

/** 将 uint16 以小端序写入缓冲区 */
static inline void _ecgr_write16le(uint8_t* dst, uint16_t val) {
    dst[0] = (uint8_t)(val);
    dst[1] = (uint8_t)(val >> 8);
}

/* ======================== 公共 API ======================== */

/* ======================== 头部初始化与读写 ======================== */

/**
 * @brief 初始化 32 字节文件头部
 *
 * 按 ECGR 格式将所有字段写入 hdr (必须至少 32 字节)。
 * 标志位的异常位图比特根据 abnormalSec 是否 > 0 或 flags 参数自动设置。
 */
static inline void ecgrHeaderInit(uint8_t* hdr,
                                   uint32_t sampleRate,
                                   uint32_t startUnix,
                                   uint32_t totalSamples,
                                   uint32_t durationSec,
                                   uint32_t abnormalSec,
                                   uint8_t flags) {
    /* 魔数 "ECGR" */
    hdr[0] = 'E'; hdr[1] = 'C'; hdr[2] = 'G'; hdr[3] = 'R';
    /* 版本 */
    hdr[ECGR_OFF_VERSION] = ECGR_VERSION;
    /* 标志 */
    uint8_t f = flags;
    if (abnormalSec > 0) f |= ECGR_FLAG_HAS_ABNORMAL_BITMAP;
    hdr[ECGR_OFF_FLAGS] = f;
    /* 各计数字段 */
    _ecgr_write32le(&hdr[ECGR_OFF_SAMPLE_RATE], sampleRate);
    _ecgr_write32le(&hdr[ECGR_OFF_START_UNIX], startUnix);
    _ecgr_write32le(&hdr[ECGR_OFF_DURATION_SEC], durationSec);
    _ecgr_write32le(&hdr[ECGR_OFF_TOTAL_SAMPLES], totalSamples);
    _ecgr_write32le(&hdr[ECGR_OFF_ABNORMAL_SEC], abnormalSec);
    /* 保留字段清零 */
    for (int i = ECGR_OFF_RESERVED; i < ECGR_HEADER_SIZE; i++) {
        hdr[i] = 0;
    }
}

/**
 * @brief 验证头部是否为有效 ECGR 格式 (魔数 + 版本 + 采样率)
 */
static inline bool ecgrHeaderValidate(const uint8_t* hdr, uint32_t sampleRate) {
    if (hdr[0] != 'E' || hdr[1] != 'C' || hdr[2] != 'G' || hdr[3] != 'R')
        return false;
    if (hdr[ECGR_OFF_VERSION] != ECGR_VERSION)
        return false;
    if (_ecgr_read32le(&hdr[ECGR_OFF_SAMPLE_RATE]) != sampleRate)
        return false;
    return true;
}

/** @brief 从头部读取总样本数 */
static inline uint32_t ecgrHeaderTotalSamples(const uint8_t* hdr) {
    return _ecgr_read32le(&hdr[ECGR_OFF_TOTAL_SAMPLES]);
}

/** @brief 从头部读取录制时长 (秒) */
static inline uint32_t ecgrHeaderDurationSec(const uint8_t* hdr) {
    return _ecgr_read32le(&hdr[ECGR_OFF_DURATION_SEC]);
}

/** @brief 从头部读取异常秒数 */
static inline uint32_t ecgrHeaderAbnormalSec(const uint8_t* hdr) {
    return _ecgr_read32le(&hdr[ECGR_OFF_ABNORMAL_SEC]);
}

/** @brief 从头部读取起始 Unix 时间戳 */
static inline uint32_t ecgrHeaderStartUnix(const uint8_t* hdr) {
    return _ecgr_read32le(&hdr[ECGR_OFF_START_UNIX]);
}

/**
 * @brief 更新头部中的计数字段 (停止录制时调用)
 *
 * 修改 totalSamples / durationSec / abnormalSec 三个字段；
 * 同时自动更新 flags 中的异常位图标志位。
 */
static inline void ecgrHeaderUpdate(uint8_t* hdr,
                                     uint32_t totalSamples,
                                     uint32_t durationSec,
                                     uint32_t abnormalSec) {
    _ecgr_write32le(&hdr[ECGR_OFF_TOTAL_SAMPLES], totalSamples);
    _ecgr_write32le(&hdr[ECGR_OFF_DURATION_SEC], durationSec);
    _ecgr_write32le(&hdr[ECGR_OFF_ABNORMAL_SEC], abnormalSec);
    /* 更新位图标志 */
    if (abnormalSec > 0) {
        hdr[ECGR_OFF_FLAGS] |= ECGR_FLAG_HAS_ABNORMAL_BITMAP;
    } else {
        hdr[ECGR_OFF_FLAGS] &= (uint8_t)(~ECGR_FLAG_HAS_ABNORMAL_BITMAP);
    }
}

/* ======================== 文件大小计算 ======================== */

/**
 * @brief 根据参数计算文件总字节数
 *
 * 文件大小 = 32 (header) + totalSamples*2 (int16 样本) + bitmap 字节
 * hasBitmap 为 true 时，bitmap 长度为 durationSec 字节。
 */
static inline uint32_t ecgrFileSize(uint32_t totalSamples,
                                     uint32_t durationSec,
                                     bool hasBitmap) {
    uint32_t sz = ECGR_HEADER_SIZE + totalSamples * 2;
    if (hasBitmap) sz += durationSec;
    return sz;
}

/**
 * @brief 从文件大小反推样本数
 *
 * totalSamples = (fileSize - 32 - bitmapBytes) / 2
 */
static inline uint32_t ecgrSamplesFromFileSize(uint32_t fileSize,
                                                uint32_t durationSec,
                                                bool hasBitmap) {
    uint32_t payload = fileSize - ECGR_HEADER_SIZE;
    if (hasBitmap) {
        if (payload < durationSec) return 0;
        payload -= durationSec;
    }
    return payload / 2;
}

/* ======================== 索引行构建与解析 ======================== */

/**
 * @brief 构建索引文件中的一行
 *
 * 格式: "<startUnix>,<dur>,<samples>,<abnSec>,<sizeBytes>\n"
 * 返回写入字节数 (不含末尾 \0); 缓冲区不足返回 -1。
 */
static inline int ecgrIdxLine(char* buf, int bufLen,
                               uint32_t startUnix,
                               uint32_t dur,
                               uint32_t samples,
                               uint32_t abnSec,
                               uint32_t sizeBytes) {
    /* 最坏情况: 5 个 uint32 最大值 = 5×10 + 4 逗号 + 1 换行 = 55, 加安全边距 */
    if (bufLen < 64) return -1; // 安全下限

    int len = snprintf(buf, (size_t)bufLen,
                       "%u,%u,%u,%u,%u\n",
                       (unsigned)startUnix, (unsigned)dur,
                       (unsigned)samples, (unsigned)abnSec,
                       (unsigned)sizeBytes);
    if (len < 0 || len >= bufLen) return -1;
    return len;
}

/**
 * @brief 解析索引文件中的一行
 *
 * 从 "<startUnix>,<dur>,<samples>,<abnSec>,<sizeBytes>" 解析各字段。
 * 允许末尾有 \n 或 \r\n。
 */
static inline bool ecgrIdxParse(const char* line,
                                 uint32_t* outStartUnix,
                                 uint32_t* outDur,
                                 uint32_t* outSamples,
                                 uint32_t* outAbnSec,
                                 uint32_t* outSizeBytes) {
    if (!line || line[0] == '\0') return false;

    unsigned long v[5];
    int n = sscanf(line, "%lu,%lu,%lu,%lu,%lu",
                   &v[0], &v[1], &v[2], &v[3], &v[4]);
    if (n != 5) return false;

    /* 检查第 5 个值后是否只有空白/换行 (拒绝多余字段) */
    const char* p = line;
    for (int i = 0; i < 5; i++) {
        while (*p >= '0' && *p <= '9') p++;
        if (i < 4) {
            if (*p != ',') return false;
            p++; // 跳过逗号
        }
    }
    /* 跳过末尾空白及换行 */
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (*p != '\0') return false; // 有额外字段

    if (outStartUnix) *outStartUnix = (uint32_t)v[0];
    if (outDur)       *outDur       = (uint32_t)v[1];
    if (outSamples)   *outSamples   = (uint32_t)v[2];
    if (outAbnSec)    *outAbnSec    = (uint32_t)v[3];
    if (outSizeBytes) *outSizeBytes = (uint32_t)v[4];
    return true;
}

#ifdef __cplusplus
}
#endif

#endif /* ECG_RECORDER_FORMAT_H */
