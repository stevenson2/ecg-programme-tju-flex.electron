/**
 * @file ecg_wifi.h
 * @brief WiFi AP 模式下 HTTP 传输模块 — 录制记录管理 API
 *
 * 在 ESP32-S3 上启动 SoftAP，通过同步 WebServer (core 2.0.17) 提供
 * RESTful API 供 Flutter App 下载/删除 ECG 录制记录。
 *
 * == 共存说明 ==
 * ESP32-S3 无线电支持 WiFi + BLE 共存 (时分复用)。
 * BLE NUS 在 AP 运行期间保持激活, 不需要任何额外操作。
 * 初始化顺序: BLE init → WiFi init → WiFi start (按需)。
 *
 * == STA 模式 ==
 * 尚未实现 (仅占位注释)。需要时添加 ecgWifiConnectSTA() 函数。
 */
 
#ifndef ECG_WIFI_H
#define ECG_WIFI_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ======================== 可调参数 (#ifndef 默认值) ======================== */

/** @brief AP 密码 (8 位数字, 方便输入) */
#ifndef ECG_WIFI_AP_PASSWORD
#define ECG_WIFI_AP_PASSWORD  "12345678"
#endif

/** @brief WebServer 响应数据块大小 (字节) */
#ifndef ECG_WIFI_CHUNK_BYTES
#define ECG_WIFI_CHUNK_BYTES  1024
#endif

/** @brief HTTP 服务端口 */
#ifndef ECG_WIFI_PORT
#define ECG_WIFI_PORT  80
#endif

/* ======================== 公共 API ======================== */

/**
 * @brief 初始化 WiFi 传输模块
 *
 * 创建 WebServer 对象并注册所有 HTTP 路由。
 * 注意: 不启动 AP, 仅分配对象和注册路由。
 * 在 BLE init 后, 进入主循环前调用。
 *
 * @return true 初始化成功
 */
bool ecgWifiInit(void);

/**
 * @brief 启动 WiFi AP 模式并开始 HTTP 服务
 *
 * SSID: ESP32-ECG-XXXX, 其中 XXXX 为 WiFi MAC 地址最后 2 字节的 4 位大写十六进制。
 * 密码: ECG_WIFI_AP_PASSWORD。
 * 若 AP 已启动/服务器已运行, 返回 false。
 *
 * @return true 启动成功
 */
bool ecgWifiStart(void);

/**
 * @brief 停止 WiFi AP 和 HTTP 服务
 * 
 * 顺序: server.stop() → WiFi.softAPdisconnect(true)。
 * 已停止状态下调用为安全 no-op。
 */
void ecgWifiStop(void);

/**
 * @brief 查询 AP 是否正在运行
 * @return true AP 已启动
 */
bool ecgWifiIsOn(void);

/**
 * @brief 处理一次 HTTP 请求 (主循环中每迭代调用)
 *
 * 内部调用 server.handleClient()。
 * 空闲时极廉价 (微秒级), 建议每迭代调用一次。
 */
void ecgWifiProcess(void);

#ifdef __cplusplus
}
#endif

#endif /* ECG_WIFI_H */
