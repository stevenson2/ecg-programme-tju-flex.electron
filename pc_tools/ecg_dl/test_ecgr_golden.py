#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PC 端金样测试（P0-1 四端一致）：读取 protocol/golden/* fixture 逐字段断言。

运行（仓库根或 pc_tools/ecg_dl）:
    python pc_tools/ecg_dl/test_ecgr_golden.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from ecgr import parse_ecgr  # noqa: E402

GOLDEN = ROOT / "protocol" / "golden"
PASS = 0
FAIL = 0


def check(cond: bool, msg: str) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {msg}")


def main() -> int:
    expected = json.loads((GOLDEN / "ecgr_expected.json").read_text(encoding="utf-8"))
    for file, exp in expected.items():
        rec = parse_ecgr((GOLDEN / file).read_bytes())
        check(rec.version == exp["version"], f"{file}: version")
        check(rec.sample_rate == exp["sampleRate"], f"{file}: sampleRate")
        check(rec.duration_sec == exp["durationSec"], f"{file}: durationSec")
        check(rec.total_samples == exp["totalSamples"], f"{file}: totalSamples")
        check(rec.abnormal_seconds == exp["abnormalSeconds"], f"{file}: abnormalSeconds")
        check(rec.has_bitmap == exp["hasBitmap"], f"{file}: hasBitmap")
        check(rec.bitmap_is_asrc == exp["bitmapIsAsrc"], f"{file}: bitmapIsAsrc")
        check(list(rec.bitmap) == exp["bitmap"], f"{file}: bitmap")
        check(rec.truncated == exp["truncated"], f"{file}: truncated")

    # 契约文件（唯一真值源）可读且与解析器常量一致
    contract = json.loads((ROOT / "protocol" / "ecg_proto.json").read_text(encoding="utf-8"))
    check(contract["ecgr"]["versions"]["supported"] == [1, 2], "contract supports v1+v2")
    check(contract["ble"]["frame"]["columns_v2"][-1]["name"] == "asrc",
          "contract v2 last column is asrc")
    check(contract["ble"]["asrc"]["bits"][0]["bit"] == "0x01", "contract asrc AI bit")

    print(f"PASS: {PASS}  FAIL: {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())