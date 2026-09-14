#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ECGR 记录解析（P0-3 四端统一，单一真值源 protocol/ecg_proto.json）。

- 支持 v1（位图 0/1）与 v2（位图 asrc 位掩码）；
- 截断策略统一为容忍：解码到实际可用字节，truncated=True，不拒收；
- 样本标定 volts = int16 / 8000.0。

用法:
    from ecgr import read_ecgr
    rec = read_ecgr("rec_latest.ecgr")
    rec.samples_v / rec.bitmap / rec.version

命令行自检:
    python pc_tools/ecg_dl/ecgr.py <path.ecgr> [--json]
"""
from __future__ import annotations

import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT_PATH = _REPO_ROOT / "protocol" / "ecg_proto.json"


def _load_contract() -> dict:
    with open(_CONTRACT_PATH, encoding="utf-8") as f:
        return json.load(f)


_C = _load_contract()
_ECGR = _C["ecgr"]
MAGIC = _ECGR["magic"].encode("ascii")
MAGIC_BYTES = bytes(_ECGR["magic_bytes"])
HEADER_SIZE = int(_ECGR["header_size"])
VERSION_SUPPORTED = tuple(int(v) for v in _ECGR["versions"]["supported"])
VERSION_CURRENT = int(_ECGR["versions"]["current"])
SCALE_TO_VOLTS = float(_ECGR["sample_scale"]["lsb"])
FLAG_HAS_ABNORMAL_BITMAP = 0x01
RESERVED0_ABNORMAL_IS_ASRC = 0x01


@dataclass
class EcgrRecord:
    version: int
    flags: int
    sample_rate: int
    start_unix: int
    duration_sec: int
    total_samples: int
    abnormal_seconds: int
    has_bitmap: bool
    bitmap: List[int] = field(default_factory=list)
    samples_v: List[float] = field(default_factory=list)
    truncated: bool = False
    src_path: Optional[str] = None

    @property
    def bitmap_is_asrc(self) -> bool:
        return self.version == 2 and (bool(self.flags & FLAG_HAS_ABNORMAL_BITMAP))

    def asrc_seconds(self) -> List[int]:
        """按 v1/v2 语义规范化位图：v1 的 0/1 → 0x01 AI；v2 原样。"""
        if not self.has_bitmap:
            return []
        if self.version == 1:
            return [0x01 if b else 0x00 for b in self.bitmap]
        return list(self.bitmap)


def parse_ecgr(data: bytes, src_path: Optional[str] = None) -> EcgrRecord:
    if len(data) < HEADER_SIZE:
        raise ValueError(
            f"ECGR 过短: {len(data)} < {HEADER_SIZE} 字节头部"
        )
    if data[:4] != MAGIC_BYTES:
        raise ValueError("魔数不匹配: 不是 ECGR 记录")
    version = data[4]
    if version not in VERSION_SUPPORTED:
        raise ValueError(f"不支持的 ECGR 版本: {version} (支持 {VERSION_SUPPORTED})")

    flags = data[5]
    sample_rate, start_unix, duration_sec, header_samples, abnormal_seconds = (
        struct.unpack_from("<IIIII", data, 6)
    )
    has_bitmap = bool(flags & FLAG_HAS_ABNORMAL_BITMAP)

    payload = len(data) - HEADER_SIZE
    # P0-3 统一截断容忍（正向解码）：
    # 样本流按头部 totalSamples 读，文件不足读到字节上限；
    # 位图从样本流之后读，缺失尾部按 0 补齐；任一处不足 truncated=True。
    if header_samples:
        total_samples = min(int(header_samples), payload // 2)
    elif has_bitmap:
        total_samples = max(0, payload - int(duration_sec)) // 2
    else:
        total_samples = payload // 2

    truncated = bool(header_samples) and total_samples < int(header_samples)
    if not header_samples and has_bitmap and payload < int(duration_sec) + total_samples * 2:
        truncated = True

    samples = struct.unpack_from(f"<{total_samples}h", data, HEADER_SIZE) if total_samples else ()
    samples_v = [s / SCALE_TO_VOLTS for s in samples]

    bitmap: List[int] = []
    if has_bitmap:
        bmp_start = HEADER_SIZE + total_samples * 2
        raw_bmp = data[bmp_start:bmp_start + int(duration_sec)]
        bitmap = list(raw_bmp)
        if len(bitmap) < int(duration_sec):
            bitmap += [0] * (int(duration_sec) - len(bitmap))
            truncated = True

    return EcgrRecord(
        version=int(version),
        flags=int(flags),
        sample_rate=int(sample_rate),
        start_unix=int(start_unix),
        duration_sec=int(duration_sec),
        total_samples=int(total_samples),
        abnormal_seconds=int(abnormal_seconds),
        has_bitmap=has_bitmap,
        bitmap=bitmap,
        samples_v=samples_v,
        truncated=truncated,
        src_path=src_path,
    )


def read_ecgr(path) -> EcgrRecord:
    p = Path(path)
    return parse_ecgr(p.read_bytes(), src_path=str(p))


def _main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    rec = read_ecgr(argv[0])
    if "--json" in argv:
        print(json.dumps({
            "path": rec.src_path,
            "version": rec.version,
            "sample_rate": rec.sample_rate,
            "duration_sec": rec.duration_sec,
            "total_samples": rec.total_samples,
            "abnormal_seconds": rec.abnormal_seconds,
            "has_bitmap": rec.has_bitmap,
            "bitmap_is_asrc": rec.bitmap_is_asrc,
            "truncated": rec.truncated,
        }, ensure_ascii=False, indent=2))
        return 0
    print(f"{rec.src_path}: v{rec.version} fs={rec.sample_rate} dur={rec.duration_sec}s "
          f"samples={rec.total_samples} abn={rec.abnormal_seconds} "
          f"bitmap={'yes' if rec.has_bitmap else 'no'} truncated={rec.truncated}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
