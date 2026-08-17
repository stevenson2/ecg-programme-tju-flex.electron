# ESP32-ECG Web Console（前端网页）

一个**零构建、零依赖**的静态前端网页：深色医疗科技风，适配桌面与移动端。
直接双击打开即可使用离线演示；在 Chrome / Edge 下可进一步连接真实硬件。

- 入口文件：`web/index.html`
- 样式：`web/css/styles.css`
- 逻辑：`web/js/*.js`（普通脚本，无模块加载，`file://` 直接打开可用）

## 四个页面

| 页面 | 功能 |
|------|------|
| **概览** | 项目能力、数据流架构、核心指标（数字均来自 `PROJECT_SUMMARY.md` / `docs/FINAL_RESULTS.md`） |
| **实时监测** | 三通道波形滚动显示 + 心率 / SQI / 运动 / AI 异常报警卡片 |
| **录制管理** | 连接 ESP32 SoftAP REST API，浏览 / 下载 / 删除在板 `.ecgr` 录制 |
| **本地回放** | 拖入 `.ecgr` 文件，概览 + 缩放 + 播放 + 异常秒高亮 + CSV 导出 |

## 快速开始

```bash
# 方式一：直接双击 web/index.html（离线演示、本地回放完全可用）

# 方式二：本地静态服务器（推荐，便于开发）
cd web
python -m http.server 8080
# 浏览器打开 http://localhost:8080
```

## 三种数据源

### 1. 演示信号（默认）
浏览器内合成 PQRST 心电：心率漂移、50Hz 工频、基线漂移、周期性运动片段与
演示性 AI 异常片段（约每 75~115 秒出现一次，锁存 5 秒），输出与固件一致的
9 列 CSV 语义。

### 2. 串口（Web Serial，Chrome / Edge 桌面版）
1. 选择「串口 (Web Serial)」→ 波特率 `460800` → 连接并选择 ESP32 端口；
2. 页面解析固件 100Hz 输出：

   ```
   clean,noisy,filtered,bpm,true_bpm,sqi,motion,abnormal_flag,confidence
   ```

   `[心率]`、`[温度]` 等诊断行会自动过滤。

### 3. 蓝牙（Web Bluetooth，Chrome / Edge）
直连 Nordic UART Service（`6E400001-B5A3-F393-E0A9-E50E24DCCA9E`），
接收 125Hz 9 列帧，并可通过 `REC_START` / `REC_STOP` / `REC_STATUS`
按钮远程控制板上录制。

## 录制管理（WiFi AP）

- 电脑连接热点 `ESP32-ECG-XXXX`（密码 `12345678`）；
- 地址默认 `http://192.168.4.1`；
- 端点：`GET /api/records`、`GET /api/records/{id}/meta`、
  `GET /api/records/{id}/data`、`DELETE /api/records/{id}`。

> 浏览器要求后端返回 CORS 响应头。本仓库已同步为 `src/wifi/ecg_wifi.cpp`
> 增加 `Access-Control-Allow-Origin` 及 OPTIONS 预检处理（编译检查通过、
> 未烧录）。旧固件会触发「CORS 拦截」提示，页面会给出直链备选方案。

## .ecgr 格式支持

- 32 字节头部：`ECGR` 魔数、版本、flags、采样率、起始时间、时长、样本数、异常秒；
- `int16` 小端样本流，电压换算 `V = sample / 8000`；
- 可选异常位图（flags bit0，时长秒数字节），回放中红色高亮；
- 头部计数大于实际数据时自动按可用数据截断并提示。

## 浏览器兼容性

| 功能 | Chrome / Edge | Firefox | Safari |
|------|:---:|:---:|:---:|
| 离线演示 / 概览 / .ecgr 回放 | ✅ | ✅ | ✅ |
| Web Serial | ✅ | ❌ | ❌ |
| Web Bluetooth | ✅ | ❌ | ❌（实验性） |
| WiFi 录制管理 | ✅（需固件 CORS） | ✅ | ✅ |

## 说明

- 本页面为科研 / 教学 / 工程演示用途，不构成医疗器械，输出不用于临床诊断；
- 固件侧修改仅做 `pio run` 编译检查，未烧录上传（遵循项目硬件操作规范）。
