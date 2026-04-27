#ifndef BLE_H
#define BLE_H

/**
 * @file ble.h
 * @brief 心电数据 BLE 蓝牙传输模块
 *
 * 基于 ESP32 内置 BLE 协议栈实现。
 * 提供 ECG_MONITOR 服务和特征值，用于实时传输滤波后的心电数据。
 * 数据格式：IEEE 754 单精度浮点数（Little Endian）
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 初始化 BLE 设备和服务
 *
 * 设备将以 \"ESP32-ECG-MONITOR\" 名称广播。
 * 特征值支持通知(Notify)和读取(Read)属性。
 */
void initBLE(void);

/**
 * @brief 通过 BLE 发送一帧心电数据
 *
 * @param value 滤波后的心电信号值（单位：mV）
 */
void sendECGData(float value);

#ifdef __cplusplus
}
#endif

#endif /* BLE_H */
