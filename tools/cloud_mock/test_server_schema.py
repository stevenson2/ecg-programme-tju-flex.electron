#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cloud mock metadata schema 校验测试（P0-4）。

运行: python tools/cloud_mock/test_server_schema.py
覆盖:
  - 合法 metadata 通过
  - 缺少必填字段 / 类型错误 / onboard_ai_summary.model 缺失 / ratio 越界 -> 报错
  - 契约文件存在且 metadata_schema 必填字段与 server.py 常量一致
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import server  # noqa: E402


def check(cond: bool, msg: str) -> int:
    print(("PASS: " if cond else "FAIL: ") + msg)
    return 0 if cond else 1


def main() -> int:
    fails = 0
    valid = {
        "device_id": "ESP32-ECG-001",
        "firmware_version": "2.0.0",
        "sample_rate": 250,
        "duration_sec": 60,
        "total_samples": 15000,
        "abnormal_seconds": 4,
        "abnormal_ratio": 0.067,
        "start_unix": 1754294400,
        "onboard_ai_summary": {"model": "v3-A", "abnormal_seconds": 4,
                               "abnormal_ratio": 0.067, "total_duration": 60},
    }
    fails += check(server.validate_meta(valid) == [], "合法 metadata 通过")

    bad_missing = dict(valid)
    del bad_missing["firmware_version"]
    fails += check(any("firmware_version" in e for e in server.validate_meta(bad_missing)),
                   "缺少 firmware_version 报错")

    bad_type = dict(valid)
    bad_type["sample_rate"] = "250"
    fails += check(any("sample_rate" in e for e in server.validate_meta(bad_type)),
                   "sample_rate 类型错误报错")

    bad_model = dict(valid)
    bad_model["onboard_ai_summary"] = {"abnormal_seconds": 1}
    fails += check(any("model" in e for e in server.validate_meta(bad_model)),
                   "onboard_ai_summary.model 缺失报错")

    bad_ratio = dict(valid)
    bad_ratio["abnormal_ratio"] = 1.5
    fails += check(any("abnormal_ratio" in e for e in server.validate_meta(bad_ratio)),
                   "abnormal_ratio 越界报错")

    contract_path = ROOT / "protocol" / "ecg_proto.json"
    fails += check(contract_path.exists(), "protocol/ecg_proto.json 存在")
    with open(contract_path, encoding="utf-8") as f:
        contract = json.load(f)
    required = set(contract["cloud_v1"]["metadata_schema"]["required"])
    fails += check(required == server.REQUIRED_META_FIELDS,
                   "契约必填字段与 server.REQUIRED_META_FIELDS 一致")

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAIL'} ({'schema ok' if fails == 0 else 'schema broken'})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())