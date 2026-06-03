#ifndef BLE_H
#define BLE_H

/**
 * @file ble.h
 * @brief 心电数据 BLE 蓝牙传输模块（NUS UART 透传）
 *
 * 实现 Nordic UART Service (NUS) 标准，兼容
 * Serial Bluetooth Terminal 等手机 App。
 *
 * 发送格式：CSV 文本行（与串口格式一致）
 *   "<clean>,<noisy>,<filtered>\r\n"
 *
 * 接收：手机端可发送命令（如 'r' 重置滤波器）
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 初始化 BLE 设备和 NUS 服务
 *
 * 设备以 "ESP32-ECG" 名称广播。
 * 使用标准 Nordic UART Service UUID。
 */
void initBLE(void);

/**
 * @brief 通过 BLE 发送 CSV 文本行
 *
 * @param message 以 null 结尾的字符串，最大长度 20 字节
 */
void sendBLEMessage(const char* message);

/**
 * @brief 检查 BLE 是否已连接
 * @return true 手机已连接, false 未连接
 */
bool isBLEConnected(void);

#ifdef __cplusplus
}
#endif

#endif /* BLE_H */
