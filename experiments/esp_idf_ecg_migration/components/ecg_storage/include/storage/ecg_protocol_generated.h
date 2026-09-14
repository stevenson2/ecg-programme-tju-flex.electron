/**
 * @file ecg_protocol_generated.h
 * @brief 由 protocol/ecg_proto.json 自动生成，禁止手改。
 *        重新生成: python scripts/gen_protocol_constants.py
 */
#ifndef ECG_PROTOCOL_GENERATED_H
#define ECG_PROTOCOL_GENERATED_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ECG_PROTO_VERSION          2
#define ECG_PROTO_FW_VER           "2.0.0"
#define ECG_PROTO_COLUMNS_V1       "1, 2, 3, 4, 5, 6, 7, 8, 9"
#define ECG_PROTO_COLUMNS_V2       "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
#define ECG_BLE_DEVICE_NAME_PREFIX "ESP32-ECG"
#define ECG_MODEL_NAME             "v3-A"

/* asrc 位定义（BLE 第 10 列 / ECGR v2 位图共用） */
#define ECG_ASRC_AI    0x01u  /* AI 异常 */
#define ECG_ASRC_RS    0x02u  /* 停搏 / 过缓过速 */
#define ECG_ASRC_FLAT  0x04u  /* 时间停搏（疑似电极脱落） */
#define ECG_ASRC_VF    0x08u  /* VF / VT 疑似 */
#define ECG_ASRC_LEADOFF 0x10u  /* 电极脱落 */

#define ECG_ASRC_KNOWN_MASK       0x1Fu

/* ECGR */
#define ECG_ECGR_MAGIC_0 0x45
#define ECG_ECGR_MAGIC_1 0x43
#define ECG_ECGR_MAGIC_2 0x47
#define ECG_ECGR_MAGIC_3 0x52
#define ECG_ECGR_HEADER_SIZE 32
#define ECG_ECGR_VERSION_1 1
#define ECG_ECGR_VERSION_2 2
#define ECG_ECGR_VERSION_CURRENT ECG_ECGR_VERSION_2
#define ECG_ECGR_SCALE_TO_VOLTS 8000.0
#define ECG_ECGR_FLAG_HAS_ABNORMAL_BITMAP 0x01u
#define ECG_ECGR_RESERVED0_ABNORMAL_IS_ASRC 0x01u

#ifdef __cplusplus
}
#endif

#endif /* ECG_PROTOCOL_GENERATED_H */
