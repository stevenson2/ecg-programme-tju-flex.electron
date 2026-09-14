# -*- coding: utf-8 -*-
"""生成 protocol/golden/* 金样 fixture（P0-1 三端共享）。

产物:
  protocol/golden/ble_frames.json   固定 BLE/CSV 帧 -> 期望逐字段
  protocol/golden/ecgr_v1.bin       v1 记录字节
  protocol/golden/ecgr_v2.bin       v2 记录字节（含 asrc 位图）
  protocol/golden/ecgr_expected.json 每个 bin 的期望解码字段

用法: python scripts/gen_protocol_golden.py [--check]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "protocol" / "golden"
SAMPLE_RATE = 250
SCALE = 8000.0
SAMPLES = [0, 1000, -2000, 3000, -4000, 32767, -32768]


def build_ecgr(version: int, duration: int, bitmap: list[int], samples: list[int]) -> bytes:
    flags = 0x01 if bitmap is not None else 0x00
    reserved0 = 0x01 if version == 2 else 0x00
    header = bytearray(32)
    header[0:4] = b"ECGR"
    header[4] = version
    header[5] = flags
    struct.pack_into("<IIIII", header, 6, SAMPLE_RATE, 1700000000, duration,
                     len(samples), sum(1 for b in bitmap if b != 0))
    header[26] = reserved0
    body = b"".join(struct.pack("<h", s) for s in samples)
    bmp = bytes(bitmap)
    return bytes(header) + body + bmp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    frames = {
        "v1_frame": "0.125,-0.010,0.120,75,0,0.923,0,0,0.014",
        "v1_frame_abnormal": "0.125,-0.010,0.120,75,0,0.923,0,1,0.014",
        "v2_frame": "0.125,-0.010,0.120,75,0,0.923,0,0,0.014,0",
        "v2_frame_ai": "0.125,-0.010,0.120,75,0,0.923,0,1,0.014,1",
        "v2_frame_multibit": "0.125,-0.010,0.120,75,0,0.923,0,1,0.014,20",
        "v2_batch": ("0.125,-0.010,0.120,75,0,0.923,0,0,0.014,0;"
                     "0.125,-0.010,0.120,75,0,0.923,0,1,0.014,1;"),
        "hello": "HELLO,2,2.0.0,asrc:0x01,0x02,0x04,0x08,0x10;",
        "diagnostic_line": "[heart] 75 BPM",
    }
    expected = {
        "v1_frame": {"clean": 0.125, "bpm": 75, "abnormal": 0, "asrc": 0, "protoVer": 1},
        "v1_frame_abnormal": {"abnormal": 1, "asrc": 1, "protoVer": 1},
        "v2_frame": {"abnormal": 0, "asrc": 0, "protoVer": 2},
        "v2_frame_ai": {"abnormal": 1, "asrc": 1, "protoVer": 2},
        "v2_frame_multibit": {"abnormal": 1, "asrc": 0x14, "protoVer": 2},
    }

    v1 = build_ecgr(version=1, duration=3, bitmap=[1, 0, 1], samples=SAMPLES)
    v2 = build_ecgr(version=2, duration=3, bitmap=[0x01, 0x14, 0x10], samples=SAMPLES)
    # 截断样本: v2 头部声明 duration=3 但只写到第 2 个位图字节（容忍策略）
    v2_trunc = v2[:-1]

    targets = {
        OUT / "ble_frames.json": json.dumps(
            {"frames": frames, "expected": expected}, ensure_ascii=False, indent=2) + "\n",
        OUT / "ecgr_v1.bin": v1,
        OUT / "ecgr_v2.bin": v2,
        OUT / "ecgr_v2_truncated.bin": v2_trunc,
        OUT / "ecgr_expected.json": json.dumps({
            "ecgr_v1.bin": {
                "version": 1, "sampleRate": SAMPLE_RATE, "durationSec": 3,
                "totalSamples": len(SAMPLES), "abnormalSeconds": 2,
                "hasBitmap": True, "bitmapIsAsrc": False,
                "bitmap": [1, 0, 1], "asrcBySecond": [1, 0, 1],
                "truncated": False,
            },
            "ecgr_v2.bin": {
                "version": 2, "sampleRate": SAMPLE_RATE, "durationSec": 3,
                "totalSamples": len(SAMPLES), "abnormalSeconds": 3,
                "hasBitmap": True, "bitmapIsAsrc": True,
                "bitmap": [1, 20, 16], "asrcBySecond": [1, 20, 16],
                "truncated": False,
            },
            "ecgr_v2_truncated.bin": {
                "version": 2, "sampleRate": SAMPLE_RATE, "durationSec": 3,
                "totalSamples": len(SAMPLES), "abnormalSeconds": 3,
                "hasBitmap": True, "bitmapIsAsrc": True,
                "bitmap": [1, 20, 0], "asrcBySecond": [1, 20, 0],
                "truncated": True,
            },
        }, ensure_ascii=False, indent=2) + "\n",
    }

    rc = 0
    for path, data in targets.items():
        blob = data if isinstance(data, (bytes, bytearray)) else data.encode("utf-8")
        if args.check:
            existing = path.read_bytes() if path.exists() else None
            if existing != blob:
                print(f"STALE: {path.relative_to(ROOT)}")
                rc = 1
            else:
                print(f"OK: {path.relative_to(ROOT)}")
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            print(f"wrote: {path.relative_to(ROOT)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())