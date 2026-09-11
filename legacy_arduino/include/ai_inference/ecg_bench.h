/**
 * @file ecg_bench.h
 * @brief ESP32-S3 串口 bench 模式（ECG_AI_BENCH 构建专用）
 *
 * PC 通过 USB-CDC 二进制协议直驱双模型 + AND 门控（M4 规范上板验收）。
 * 命令集见 ecg_bench.cpp 头注释。bench 模式独占 Serial；
 * 生产构建（无 ECG_AI_BENCH）不编译本模块逻辑。
 */
#ifndef ECG_BENCH_H
#define ECG_BENCH_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void ecg_bench_init(void);
bool ecg_bench_active(void);
void ecg_bench_on_byte(uint8_t b);

#ifdef __cplusplus
}
#endif

#endif /* ECG_BENCH_H */
