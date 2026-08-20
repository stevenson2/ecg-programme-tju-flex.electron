# 云端 REST v1 接口规范（阶段C 下一步待办 #1, 2026-08-08）

> **任务来源**：阶段C 定稿决策 —— 本阶段只做 REST 规范 + App 上传客户端 + 本地
> mock，不接真实云。
> **验收**：本规范文件（Contract C8）；mock 服务器与 App 客户端按本规范实现。
> **结论先行**：REST v1 共 5 端点（上传 / 触发分析 / 报告 / 列表 / 预留删除），
> 鉴权统一 Bearer Token（开发期占位 `dev-token`）；上传用 multipart/form-data
> （part "meta" = JSON 元数据 + part "data" = 原始 .ecgr 二进制，int16 流）；报告
> 为三段式 JSON（summary / events / recommendation）。数据部分与固件 ECGR 格式
> （Contract C5，`include/storage/ecg_recorder_format.h`）完全一致，固件 / App
> 解码器 / 云 mock 三端共用同一布局。

---

## 1. 概述

### 1.1 场景

```
板上 ECG 记录 (SPIFFS, .ecgr) → BLE 下载至手机 App → App 经 REST v1 上传
→ 云端分析（对拍级异常做汇总与事件提取）→ 分析报告回传 App 展示
```

三条链路的主角：固件负责采集与落盘，App 负责传输与展示，云端负责重分析与报告
生成。本规范定义 App 与云端之间的全部交互。

### 1.2 Base URL 约定

| 项 | 值 |
|----|----|
| Base URL | `https://api.example.com/v1` |
| 联调替代 | 本地 mock 用 `http://<host>:<port>/v1`（App 配置可切换） |
| 数据格式 | JSON（UTF-8） |
| 上传格式 | `multipart/form-data`（仅上传端点） |
| 时间戳 | Unix 秒（整数） |
| 时区 | UTC |

### 1.3 鉴权

| 项 | 值 |
|----|----|
| 请求头 | `Authorization: Bearer <token>` |
| token 来源 | 云端下发（登录 / 设备注册流程，后续阶段定义） |
| 开发期占位 | `dev-token`（mock 接受任意非空 token，但请求必须携带该头） |
| 401 语义 | 缺少 Authorization 头 / token 无效 / token 过期，**一律返回 401**，不区分具体原因（避免 token 枚举） |

除预留端点（§2.5）外，所有端点均需鉴权；鉴权失败时错误响应见 §5 错误码表。

---

## 2. 端点表（5）

### 2.1 `POST /v1/records` — 上传记录

| 项 | 值 |
|----|----|
| 方法 / 路径 | `POST /v1/records` |
| 请求体 | `multipart/form-data`，两个 part： |
| part "meta" | JSON 字符串（§3 元数据 Schema），`Content-Type: application/json` |
| part "data" | 原始 .ecgr 文件字节流（§4 数据部分格式），`Content-Type: application/octet-stream` |
| 成功 | `201 Created` → `{"record_id":"...","status":"uploaded"}` |
| 400 | 元数据缺失（缺 part 或缺必填字段）/ JSON 格式错误 / ECGR 头校验失败 |
| 401 | 未鉴权 |
| 413 | 记录超限（单条上限 20MB，对应 ~43min @250Hz int16，见 §5） |

说明：

- part 顺序不强制（服务端按 part name 识别），但约定先 meta 后 data。
- `record_id` 由服务端生成（UUID），客户端以其作为后续 analyze / report 的路径参数。
- 服务端校验 data 前 4 字节 magic == "ECGR"，且头部 sampleRate 与 meta.sample_rate 一致，否则 400。

### 2.2 `POST /v1/records/{id}/analyze` — 触发分析

| 项 | 值 |
|----|----|
| 方法 / 路径 | `POST /v1/records/{id}/analyze` |
| 请求体 | 无（空 body） |
| 同步语义（mock） | `200 OK` → `{"status":"analyzing"}` —— mock 同步完成分析并落库报告 |
| 异步语义（真实后端） | `202 Accepted` → `{"job_id":"..."}` —— 分析入队，客户端轮询 GET report 直到 status=completed |
| 400 | `{id}` 格式非法 |
| 404 | 记录不存在 |
| 说明 | 客户端需同时兼容 200 与 202 两种返回；对已分析完成的记录重复触发返回 409（或直接返回既有报告，二选一，mock 采用后者） |

### 2.3 `GET /v1/records/{id}/report` — 分析报告

| 项 | 值 |
|----|----|
| 方法 / 路径 | `GET /v1/records/{id}/report` |
| 成功 | `200 OK` → 报告 JSON（§6 报告格式） |
| 404 | 记录不存在 |
| 409 | 分析未完成（status 为 pending / analyzing / failed 时报告未就绪） |
| 说明 | 报告生成后重复获取结果幂等；App 用该端点轮询（间隔建议 ≥2s）直至 status=completed |

### 2.4 `GET /v1/users/{uid}/records` — 记录列表

| 项 | 值 |
|----|----|
| 方法 / 路径 | `GET /v1/users/{uid}/records` |
| 查询参数 | `page`：页码，从 1 起，默认 1；`page_size`：每页条数，默认 20，上限 100 |
| 成功 | `200 OK` → `{"total":N,"page":P,"page_size":S,"items":[...]}` |
| 401 | 未鉴权 |
| 说明 | `items` 为记录摘要数组，每项含 `record_id` / `start_unix` / `duration_sec` / `status` / `abnormal_ratio` 等（与 §3 meta 字段对应）；`total` 为该用户记录总数 |

### 2.5 `DELETE /v1/records/{id}` — 删除记录（预留）

| 项 | 值 |
|----|----|
| 方法 / 路径 | `DELETE /v1/records/{id}` |
| 状态 | ⚠️ **预留（可选）**：本阶段 mock 不实现（返回 501 Not Implemented），App 暂不调用 |
| 成功 | `204 No Content` |
| 404 | 记录不存在 |

---

## 3. 元数据 Schema（part "meta" 的 JSON）

| 字段 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| device_id | string | ✅ | 设备标识（BLE 设备名 / 固件序列号） |
| firmware_version | string | ✅ | 固件版本号（如 `"0.1.0"`） |
| sample_rate | int | — | 采样率 Hz，**默认 250**；与 ECGR 头交叉校验 |
| duration_sec | int | — | 录制时长（秒） |
| total_samples | int | — | 总样本数 |
| abnormal_seconds | int | — | 异常秒数（来自 1Hz 异常位图） |
| abnormal_ratio | float | — | 异常占比 = abnormal_seconds / duration_sec（0–1） |
| start_unix | int | — | 录制起始 Unix 时间戳（秒） |
| onboard_ai_summary | object | — | 板上 AI 摘要（见下表） |

`onboard_ai_summary` 对象：

| 字段 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| mean_confidence | float | ✅ | 板上 AI 全时段平均异常置信度（0–1） |
| max_confidence | float | ✅ | 全时段最大异常置信度（0–1） |
| abnormal_flag_count | int | ✅ | 1Hz 异常位图中置 1 的秒数 |
| model | string | ✅ | 板上模型标识，固定 `"exp6-SGD"` |

示例：

```json
{
  "device_id": "ESP32-ECG-001",
  "firmware_version": "0.1.0",
  "sample_rate": 250,
  "duration_sec": 60,
  "total_samples": 15000,
  "abnormal_seconds": 4,
  "abnormal_ratio": 0.067,
  "start_unix": 1754294400,
  "onboard_ai_summary": {
    "mean_confidence": 0.12,
    "max_confidence": 0.93,
    "abnormal_flag_count": 4,
    "model": "exp6-SGD"
  }
}
```

---

## 4. 数据部分格式（part "data"）

- 上传内容为**原始 .ecgr 文件字节流**，不做任何额外编码 / 封装。
- 布局引用 **Contract C5**（`include/storage/ecg_recorder_format.h`）：

| 段 | 内容 |
|----|------|
| 头部（32B） | magic `"ECGR"`（4B）+ version（1B）+ flags（1B）+ sampleRate / startUnix / durationSec / totalSamples / abnormalSec（各 uint32，**全小端**）+ reserved（6B） |
| 样本流 | int16 LE 样本 @250Hz（录制时固件 2:1 抽取 500→250Hz，int16 缩放统一 scale=8000.0） |
| 异常位图 | 每异常秒 1 字节；仅当 abnormalSec>0 时存在（flags bit0 `ECGR_FLAG_HAS_ABNORMAL_BITMAP` 置位） |

- 文件大小 = 32 + totalSamples×2 + bitmap 字节数（bitmap 长度 = durationSec）。
- 服务端校验：前 4 字节 == `"ECGR"`；header.sampleRate 与 meta.sample_rate 一致。
- **三端一致**：固件（SPIFFS 记录器）/ App 解码器 / 云 mock 共用同一格式定义
  （头文件声明固件、PC 解码器、云 mock 可共同引用，不依赖平台）；任一端的布局
  变更必须三端同步。

---

## 5. 错误码表

统一错误响应结构：

```json
{"error": {"code": 400, "message": "..."}}
```

| code | HTTP | 语义 | message 示例 |
|:---:|------|------|------|
| 400 | Bad Request | 元数据缺失 / JSON 格式错误 / 必填字段缺失 / ECGR 头校验失败 / 路径参数非法 | `"meta part missing"` |
| 401 | Unauthorized | 缺少 / 无效 / 过期 token（不区分原因） | `"invalid token"` |
| 404 | Not Found | 记录 / 用户不存在 | `"record not found"` |
| 409 | Conflict | 分析未完成，报告未生成 | `"analysis not completed"` |
| 413 | Payload Too Large | 单条记录超限（上限 20MB） | `"record too large"` |
| 416 | Range Not Satisfiable | 预留：断点续传 / 数据下载类端点（阶段 B WiFi 传输已用 Range 断点，云端下载预留） | — |

客户端对 4xx 的处理约定：400/401 终止流程并提示用户（401 提示重新登录）；
404/409 按状态机分支（404 重新上传，409 继续轮询）；413 提示本地记录过大。

---

## 6. 报告格式（`GET /v1/records/{id}/report`）

```json
{
  "record_id": "...",
  "status": "completed",
  "summary": {...},
  "events": [...],
  "recommendation": "..."
}
```

`status` 取值：`pending`（已上传未分析）/ `analyzing`（分析中）/ `completed`
（报告就绪）/ `failed`（分析失败）。

`summary`：

| 字段 | 类型 | 说明 |
|------|------|------|
| duration_sec | int | 总时长（秒） |
| abnormal_seconds | int | 异常秒数 |
| abnormal_ratio | float | 异常占比（0–1） |
| mean_bpm | float | 平均 BPM 估计 |

`events`（数组，每事件一个对象）：

| 字段 | 类型 | 说明 |
|------|------|------|
| start_sec | int | 事件起始秒 |
| end_sec | int | 事件结束秒 |
| type | string | 事件类型（如 `"abnormal_beat"` / `"tachycardia"` / `"bradycardia"`，枚举以云端分析模型为准） |
| confidence | float | 事件置信度（0–1） |

`recommendation`：字符串建议文案。**mock 用模拟文案并注明 `"simulated"`**
（如 `"simulated: 建议近期复查心电图，重点关注异常时段"`），真实后端的文案
由云端医疗规则生成，不含该标记。

---

## 7. Mock 一致性说明

- 实现文件：`tools/cloud_mock/server.py`（本地 mock 服务器，独立任务实现，本规范
  为其唯一契约）。
- 覆盖端点：§2.1–§2.4（上传 / 触发分析 / 报告 / 列表）；§2.5 预留 DELETE 不实现。
- analyze 行为：**同步返回 200**，并生成**确定性模拟报告**——以 record_id 为
  随机种子，同一 record_id 重复分析结果完全一致（幂等）；summary 取自 meta 的
  abnormal_seconds / abnormal_ratio，mean_bpm 在 60–100 区间由种子生成；
  events 由异常秒位图聚类派生；recommendation 为 `"simulated: ..."` 文案。
- 鉴权：接受任意非空 Bearer token（开发期占位 `dev-token`），缺失则 401。
- 一致性纪律：规范 / mock / App 客户端（上传与报告展示）三端互为唯一契约，
  任何一端变更必须三端同步，并以 mock 冒烟测试交叉验证。
