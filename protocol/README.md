# 协议契约（P0-1 单一真值源）

`protocol/ecg_proto.json` 是 ESP32-ECG 全系统唯一机器可读协议契约：

| 域 | 内容 | 主要消费端 |
|---|---|---|
| ble | NUS UUID、帧列定义（v1 9 列 / v2 10 列）、HELLO 协商、asrc 位定义、命令集 | 固件 main.cc、App csv_parser、web ecg-core |
| ecgr | 32B 头、v1/v2 版本语义、截断策略、样本标定 | 固件 ecg_recorder_format.h、App codec、web ecg-core、pc_tools/ecgr.py |
| wifi_rest | 板载 HTTP 端点 | App record_api、web records |
| cloud_v1 | 云端 /v1 端点与 metadata schema | App upload_service、tools/cloud_mock/server.py |

## 生成物（禁止手改）

`scripts/gen_protocol_constants.py` 从契约生成三份常量（全部入库；改契约后必须重跑）：
