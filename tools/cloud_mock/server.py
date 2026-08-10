#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云端 Mock 服务器 — ECG 记录流水线 REST API v1

实现 Contract C8 全量端点，零外部依赖 (纯 Python 标准库)。
HTTP 服务基于 http.server.ThreadingHTTPServer，multipart 解析基于 email.parser。

端点:
  POST   /v1/records                — 上传记录 (multipart: meta JSON + data .ecgr)
  POST   /v1/records/{id}/analyze   — 模拟同步分析
  GET    /v1/records/{id}/report    — 确定性模拟报告
  GET    /v1/users/{uid}/records    — 分页列表

认证: Bearer token = "dev-token"，其他 token 或缺失 -> 401
端口: 默认 8000，可通过命令行参数指定

参考: AGENTS.md Contract C8, include/storage/ecg_recorder_format.h
"""

import email.parser
import hashlib
import io
import json
import os
import re
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# ======================== 配置 ========================
API_TOKEN = "dev-token"  # Bearer token
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 20

# 需验证的 metadata 必填字段
REQUIRED_META_FIELDS = {"device_id", "firmware_version"}


# ======================== 辅助函数 ========================

def make_json_response(data, status=200):
    """构建 JSON 响应体字节序列。"""
    body = json.dumps(data, ensure_ascii=False, indent=2)
    return body.encode("utf-8"), status, "application/json; charset=utf-8"


def make_error(code, message, status):
    """构建错误 JSON 响应。"""
    return make_json_response({"error": {"code": code, "message": message}}, status)


def hash_deterministic(record_id):
    """将 record_id 映射到确定性浮点数 (0.0, 1.0)。

    用于基于 record_id 生成稳定的模拟报告内容，
    不同 record_id 产生不同报告，同一 record_id 永远一致。
    """
    h = hashlib.md5(record_id.encode()).hexdigest()
    # 取前 8 个 hex 字符转换为 (0,1) 的浮点数
    frac = int(h[:8], 16) / 0x100000000
    return frac


def parse_multipart(content_type, body):
    """解析 multipart/form-data 请求体。

    返回 dict: {"meta": json_str, "data": bytes, ...}
    在标准库路径下使用 email.parser.BytesParser。
    """
    # 提取 boundary
    match = re.search(r'boundary=([^;\s]+)', content_type)
    if not match:
        return None
    boundary = match.group(1).encode()

    # 构造完整 MIME 消息 (headers + 空行 + body)
    mime_data = b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
    msg = email.parser.BytesParser().parsebytes(mime_data)

    parts = {}
    for part in msg.get_payload():
        # get_content_disposition() 只返回主类型 ('form-data'),
        # name 参数须经 get_param 获取 (修复: 原实现误判 'name=' 导致全部 part 被跳过)
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True)
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        parts[name] = payload

    return parts


def load_meta(record_id):
    """从侧边文件加载记录的 metadata JSON。"""
    meta_path = os.path.join(DATA_DIR, f"{record_id}.meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_record(record_id, ecgr_data, meta):
    """将 ECGR 数据和 metadata 侧边文件保存到 data/ 目录。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    # 保存二进制数据
    ecgr_path = os.path.join(DATA_DIR, f"{record_id}.ecgr")
    with open(ecgr_path, "wb") as f:
        f.write(ecgr_data)
    # 保存 metadata JSON 侧边文件
    meta_path = os.path.join(DATA_DIR, f"{record_id}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def list_records():
    """扫描 data/ 目录下所有 .ecgr 文件，返回 (record_id, meta) 列表。

    按 start_unix 降序排列，缺失 metadata 的记录排在末尾。
    """
    records = []
    if not os.path.isdir(DATA_DIR):
        return records
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".ecgr"):
            continue
        record_id = fname[:-5]  # 去掉 .ecgr
        meta = load_meta(record_id)
        start_unix = meta.get("start_unix", 0) if meta else 0
        records.append((record_id, meta, start_unix))
    # 按 start_unix 降序
    records.sort(key=lambda x: x[2], reverse=True)
    return records


def build_report(record_id, meta):
    """基于 record_id + meta 构建确定性模拟报告。"""
    frac = hash_deterministic(record_id)

    duration_sec = meta.get("duration_sec", 60)
    total_samples = meta.get("total_samples", duration_sec * 250)
    sample_rate = meta.get("sample_rate", 250)

    # 模拟分析结果 — 由 seed (frac) 决定具体数值
    mean_confidence = round(0.5 + frac * 0.4, 3)
    max_confidence = round(0.85 + frac * 0.14, 3)

    # 事件数: 基于 abnormal_seconds 和 frac
    abnormal_seconds = meta.get("abnormal_seconds", 0)
    event_count = max(1, min(3, int(abnormal_seconds * 0.5) + int(frac * 2)))

    # 生成模拟事件
    events = []
    for i in range(event_count):
        start_offset = frac * (duration_sec - 10) + i * 5
        evt = {
            "start_sec": round(start_offset, 1),
            "end_sec": round(start_offset + 3.0 + frac * 2, 1),
            "type": "simulated",
            "confidence": round(0.7 + frac * 0.25 + i * 0.03, 3),
        }
        events.append(evt)

    summary = {
        "duration_sec": duration_sec,
        "total_samples": total_samples,
        "sample_rate": sample_rate,
        "abnormal_seconds": abnormal_seconds,
        "abnormal_ratio": round(abnormal_seconds / duration_sec, 4) if duration_sec else 0.0,
        "mean_confidence": mean_confidence,
        "max_confidence": max_confidence,
        "event_count": event_count,
    }

    return {
        "record_id": record_id,
        "status": "completed",
        "summary": summary,
        "events": events,
        "recommendation": (
            f"经 AI 模型分析，该记录中检测到 {event_count} 处疑似异常事件。"
            "建议结合临床表现进行综合判断。"
            "（模拟报告，非真实医疗诊断）"
        ),
    }


# ======================== HTTP 请求处理器 ========================

class ECGCloudHandler(BaseHTTPRequestHandler):
    """REST API v1 请求处理器。"""

    # 抑制标准库的日志输出 (以减小噪声)
    def log_message(self, fmt, *args):
        print(f"[SERVER] {fmt % args}", flush=True)

    def _check_auth(self):
        """验证 Bearer token。返回 None 表示通过；否则返回 (body, status, content_type)。"""
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return make_error("unauthorized", "缺少或无效的认证信息", 401)
        token = auth[7:].strip()
        if token != API_TOKEN:
            return make_error("unauthorized", "无效的认证令牌", 401)
        return None

    def _auth_or_fail(self):
        """验证认证，失败时直接写入 401 响应并返回 True。"""
        err = self._check_auth()
        if err:
            body, status, ct = err
            self.send_response(status)
            self.send_header("Content-Type", ct)
            self.end_headers()
            self.wfile.write(body)
            return True
        return False

    def _send_json(self, data, status=200):
        """便捷发送 JSON 响应。"""
        body, _, ct = make_json_response(data, status)
        self.send_response(status)
        self.send_header("Content-Type", ct)
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code, message, status):
        """便捷发送错误 JSON 响应。"""
        body, _, ct = make_error(code, message, status)
        self.send_response(status)
        self.send_header("Content-Type", ct)
        self.end_headers()
        self.wfile.write(body)

    # -------------------- 路由分发 --------------------

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        # POST /v1/records — 上传记录
        if re.match(r"^/v1/records$", path):
            self._handle_upload()
            return

        # POST /v1/records/{id}/analyze — 模拟分析
        m = re.match(r"^/v1/records/([a-f0-9\-]+)/analyze$", path)
        if m:
            record_id = m.group(1)
            self._handle_analyze(record_id)
            return

        self._send_error("not_found", "未知的 API 路径", 404)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        # GET /v1/records/{id}/report — 获取报告
        m = re.match(r"^/v1/records/([a-f0-9\-]+)/report$", path)
        if m:
            record_id = m.group(1)
            self._handle_get_report(record_id)
            return

        # GET /v1/users/{uid}/records — 分页列表
        m = re.match(r"^/v1/users/([^/]+)/records$", path)
        if m:
            uid = m.group(1)
            self._handle_list_records(uid, parsed.query)
            return

        self._send_error("not_found", "未知的 API 路径", 404)

    # -------------------- 端点实现 --------------------

    def _handle_upload(self):
        if self._auth_or_fail():
            return

        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            self._send_error("bad_request", "无效的请求体", 400)
            return

        body = self.rfile.read(content_length)

        # 解析 multipart
        parts = parse_multipart(content_type, body)
        if not parts:
            self._send_error("bad_request", "无法解析 multipart 请求体", 400)
            return

        meta_raw = parts.get("meta")
        data_raw = parts.get("data")

        if not meta_raw or not data_raw:
            self._send_error("bad_request", "缺少 meta 或 data 部分", 400)
            return

        # 解析 metadata JSON
        try:
            meta = json.loads(meta_raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_error("bad_request", "meta 部分不是有效的 JSON", 400)
            return

        # 验证必填字段
        missing = REQUIRED_META_FIELDS - set(meta.keys())
        if missing:
            self._send_error(
                "bad_request",
                f"缺少必填字段: {', '.join(sorted(missing))}",
                400,
            )
            return

        # 生成 record_id
        record_id = uuid.uuid4().hex

        # 持久化
        save_record(record_id, data_raw, meta)

        resp = {
            "record_id": record_id,
            "status": "uploaded",
        }
        self._send_json(resp, 201)

    def _handle_analyze(self, record_id):
        if self._auth_or_fail():
            return

        meta = load_meta(record_id)
        if not meta:
            self._send_error("not_found", f"记录不存在: {record_id}", 404)
            return

        # 同步：直接标记为已分析 (mock)
        meta["analyzed"] = True
        meta_path = os.path.join(DATA_DIR, f"{record_id}.meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        resp = {
            "record_id": record_id,
            "status": "analyzed",
        }
        self._send_json(resp, 200)

    def _handle_get_report(self, record_id):
        if self._auth_or_fail():
            return

        meta = load_meta(record_id)
        if not meta:
            self._send_error("not_found", f"记录不存在: {record_id}", 404)
            return

        report = build_report(record_id, meta)
        self._send_json(report, 200)

    def _handle_list_records(self, uid, query_string):
        if self._auth_or_fail():
            return

        params = parse_qs(query_string)
        page = int(params.get("page", [DEFAULT_PAGE])[0])
        page_size = int(params.get("page_size", [DEFAULT_PAGE_SIZE])[0])

        records = list_records()

        # 分页
        start = (page - 1) * page_size
        end = start + page_size
        page_records = records[start:end]

        result = {
            "records": [
                {
                    "record_id": rid,
                    "device_id": meta.get("device_id", ""),
                    "firmware_version": meta.get("firmware_version", ""),
                    "duration_sec": meta.get("duration_sec", 0),
                    "start_unix": meta.get("start_unix", 0),
                    "status": "analyzed" if (meta or {}).get("analyzed") else "uploaded",
                }
                for rid, meta, _ in page_records
            ],
            "page": page,
            "page_size": page_size,
            "total": len(records),
        }
        self._send_json(result, 200)


# ======================== 服务入口 ========================

def main():
    port = DEFAULT_PORT
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"错误: 无效的端口号 '{sys.argv[1]}'", file=sys.stderr)
            sys.exit(1)

    server = ThreadingHTTPServer((DEFAULT_HOST, port), ECGCloudHandler)
    print(f"ECG Cloud Mock Server 启动于 http://{DEFAULT_HOST}:{port}")
    print(f"数据目录: {DATA_DIR}")
    print(f"认证 token: {API_TOKEN}")
    print("按 Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.server_close()


if __name__ == "__main__":
    main()
