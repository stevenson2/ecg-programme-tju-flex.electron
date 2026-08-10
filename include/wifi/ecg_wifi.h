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
#include <stddef.h>   /* size_t (ecgWifiDiagStaIp) */

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

/* ======================== 诊断配置 API (WiFi beacon 专项, 2026-08-10) ========================
 * 由 main.cpp 的 DIAG 命令 (串口+BLE 双通道) 调用。
 * 所有设置仅影响下一次 ecgWifiStart(), 默认值与正式固件行为完全一致。
 */

/** @brief 设置下次 WIFI_ON 的 AP 发射功率策略 (0=跳过 setTxPower, 34=8.5dBm, 78=19.5dBm) */
void ecgWifiDiagSetTxPower(int v);

/** @brief 设置下次 WIFI_ON 的 AP 信道 (合法值 1/6/11) */
void ecgWifiDiagSetChannel(int v);

/** @brief 设置 AP 启动序列 (false=当前快速序列, true=PR#1865 式慢速 OFF→AP 切换 + setSleep(false)) */
void ecgWifiDiagSetSeqSlow(bool v);

int  ecgWifiDiagGetTxPower(void);
int  ecgWifiDiagGetChannel(void);
bool ecgWifiDiagGetSeqSlow(void);

/* ======================== STA 测试 API (候选D, 2026-08-10) ========================
 * DIAG STA <ssid> <pass> 使用: 以 AP_STA 共存模式连接真实路由器,
 * 验证 WiFi TX/RX 全链路 (对照 AP beacon 不可见问题)。不停止 AP。 */

/** @brief 发起 STA 连接 (AP 保持运行, AP_STA 共存), 非阻塞 */
bool ecgWifiDiagStaConnect(const char* ssid, const char* pass);

/** @brief 断开 STA, 回到纯 AP 模式 */
void ecgWifiDiagStaDisconnect(void);

/** @brief 当前 STA 状态 (arduino wl_status_t 数值: 0=IDLE, 3=CONNECTED 等) */
int ecgWifiDiagStaStatus(void);

/** @brief STA 获取的 IP 字符串 (未连接为 "0.0.0.0") */
void ecgWifiDiagStaIp(char* buf, size_t len);

/** @brief 当前 WiFi 模式 (arduino wifi_mode_t 数值: 1=STA, 2=AP, 3=AP_STA) */
int ecgWifiDiagGetMode(void);

#ifdef __cplusplus
}
#endif

#endif /* ECG_WIFI_H */
