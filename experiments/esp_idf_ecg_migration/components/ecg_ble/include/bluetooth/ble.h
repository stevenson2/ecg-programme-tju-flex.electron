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
#include <stddef.h>   /* size_t */

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
 * @brief 通过 BLE 发送文本。按当前 ATT_MTU-3 在 ';' 或换行处切块。
 *
 * 未连接、TX 未订阅、或协议栈拒绝时返回 false，且不把失败当成已送达。
 * 订阅前的调用不计入 radio 失败计数。
 *
 * @param message 以 null 结尾的字符串
 * @return true 全部切块都已交给协议栈
 */
bool sendBLEMessage(const char* message);

/** @brief notify 被协议栈拒绝的累计次数（订阅前的跳过不计入）。 */
uint32_t bleNotifyFailCount(void);

/**
 * @brief 检查 BLE 是否已连接
 * @return true 手机已连接, false 未连接
 */
bool isBLEConnected(void);

/**
 * @brief 从 BLE 命令队列中非阻塞取出一个命令行
 *
 * 用于 main.cpp 主循环轮询 BLE RX 回调投递的命令。
 *
 * @param out  输出缓冲区 (存放 null 结尾的命令串)
 * @param len  缓冲区大小 (字节)
 * @return     true 成功取出一条命令
 */
bool bleCommandQueueTake(char* out, size_t len);

/**
 * @brief 实时波形流开关 (STREAM 1/0)。关流时连接参数切到慢间隔 (省射频),
 * 开流且 TX 已订阅时切回快间隔。
 */
void bleSetStreamActive(bool active);

#ifdef __cplusplus
}
#endif

#endif /* BLE_H */
