#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 protocol/ecg_proto.json 生成三端常量文件（P0-1 单一真值源）。

生成物（全部入库）：
  experiments/esp_idf_ecg_migration/components/ecg_storage/include/storage/ecg_protocol_generated.h  C++
  ecg_app/lib/services/protocol_generated.dart                   Dart
  web/js/protocol_generated.js                                   JS

用法（仓库根）：
  python scripts/gen_protocol_constants.py            # 生成
  python scripts/gen_protocol_constants.py --check    # 只校验生成物是否最新（CI/本地）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "protocol" / "ecg_proto.json"


def load_contract() -> dict:
    with open(CONTRACT, encoding="utf-8") as f:
        return json.load(f)


def c_header(d: dict) -> str:
    asrc = d["ble"]["asrc"]["bits"]
    cols_in = ", ".join(str(c["index"]) for c in d["ble"]["frame"]["columns_v1"])
    cols_v2 = ", ".join(str(c["index"]) for c in d["ble"]["frame"]["columns_v2"])
    bits = "\n".join(
        "#define ECG_ASRC_%-9s %-6s /* %s */"
        % (b["name"], (b["bit"] + "u"), b["ui_label"])
        for b in asrc
    ).replace("   ", " ")
    return f"""/**
 * @file ecg_protocol_generated.h
 * @brief 由 protocol/ecg_proto.json 自动生成，禁止手改。
 *        重新生成: python scripts/gen_protocol_constants.py
 */
#ifndef ECG_PROTOCOL_GENERATED_H
#define ECG_PROTOCOL_GENERATED_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

#define ECG_PROTO_VERSION          {d["ble"]["proto_ver"]}
#define ECG_PROTO_FW_VER           "{d["ble"]["fw_ver"]}"
#define ECG_PROTO_COLUMNS_V1       "{cols_in}"
#define ECG_PROTO_COLUMNS_V2       "{cols_v2}"
#define ECG_BLE_DEVICE_NAME_PREFIX "{d["ble"]["nus"]["device_name_prefix"]}"\n#define ECG_MODEL_NAME             "{d["ble"]["model"]}"

/* asrc 位定义（BLE 第 10 列 / ECGR v2 位图共用） */
{bits}

#define ECG_ASRC_KNOWN_MASK       0x1Fu

/* ECGR */
#define ECG_ECGR_MAGIC_0 0x45
#define ECG_ECGR_MAGIC_1 0x43
#define ECG_ECGR_MAGIC_2 0x47
#define ECG_ECGR_MAGIC_3 0x52
#define ECG_ECGR_HEADER_SIZE {d["ecgr"]["header_size"]}
#define ECG_ECGR_VERSION_1 1\n#define ECG_ECGR_VERSION_2 {d["ecgr"]["versions"]["current"]}\n#define ECG_ECGR_VERSION_CURRENT ECG_ECGR_VERSION_2
#define ECG_ECGR_SCALE_TO_VOLTS {d["ecgr"]["sample_scale"]["lsb"]:.1f}
#define ECG_ECGR_FLAG_HAS_ABNORMAL_BITMAP 0x01u
#define ECG_ECGR_RESERVED0_ABNORMAL_IS_ASRC 0x01u

#ifdef __cplusplus
}}
#endif

#endif /* ECG_PROTOCOL_GENERATED_H */
"""


def dart_file(d: dict) -> str:
    asrc = d["ble"]["asrc"]["bits"]
    priority = [int(x, 16) for x in d["ble"]["asrc"]["priority"]]
    asrc = sorted(asrc, key=lambda b: priority.index(int(b["bit"], 16)))
    names = "\n".join(
        "  static const int %s = %s; // %s" % (b["name"].lower(), b["bit"], b["ui_label"])
        for b in asrc
    )
    label_entries = ",\n".join(
        "    AsrcMeta(bit: %s, name: '%s', label: '%s', level: '%s')"
        % (b["bit"], b["name"], b["ui_label"], b["ui_level"])
        for b in asrc
    )
    return f"""// GENERATED FILE - DO NOT EDIT.
// Source: protocol/ecg_proto.json
// Regenerate: python scripts/gen_protocol_constants.py
// ignore_for_file: constant_identifier_names

/// ESP32-ECG 协议契约常量（P0-1 单一真值源）。
class ProtocolGenerated {{
  ProtocolGenerated._();

  static const int protoVersion = {d["ble"]["proto_ver"]};
  static const String fwVersion = '{d["ble"]["fw_ver"]}';
  static const String bleFramesPerNotifyHint = '2';
  static const int ecgrHeaderSize = {d["ecgr"]["header_size"]};
  static const int ecgrVersionCurrent = {d["ecgr"]["versions"]["current"]};
  static const double ecgrScaleToVolts = {d["ecgr"]["sample_scale"]["lsb"]:.1f};
  static const int ecgrFlagHasAbnormalBitmap = 0x01;
  static const int ecgrReserved0AbnormalIsAsrc = 0x01;
  static const int csvColumnsV1 = {len(d["ble"]["frame"]["columns_v1"])};
  static const int csvColumnsV2 = {len(d["ble"]["frame"]["columns_v2"])};

  /// asrc 位定义（BLE 第 10 列 / ECGR v2 位图共用）。
{names}

  /// UI 分级文案，顺序即优先级（高 -> 低）。
  static const List<AsrcMeta> asrcMeta = <AsrcMeta>[
{label_entries},
  ];

  static bool isKnownAsrc(int asrc) => (asrc & 0x1F) != 0;

  /// 返回最高优先级的 asrc 元数据；无命中返回 null。
  static AsrcMeta? primaryAsrc(int asrc) {{
    for (final m in asrcMeta) {{
      if ((asrc & m.bit) != 0) return m;
    }}
    return null;
  }}
}}

class AsrcMeta {{
  final int bit;
  final String name;
  final String label;
  final String level; // info | warning | critical
  const AsrcMeta({{required this.bit, required this.name, required this.label, required this.level}});
}}
"""


def js_file(d: dict) -> str:
    asrc = d["ble"]["asrc"]["bits"]
    priority = [int(x, 16) for x in d["ble"]["asrc"]["priority"]]
    asrc = sorted(asrc, key=lambda b: priority.index(int(b["bit"], 16)))
    entries = ",\n".join(
        "    { bit: %s, name: '%s', label: '%s', level: '%s' }"
        % (b["bit"], b["name"], b["ui_label"], b["ui_level"])
        for b in asrc
    )
    return f"""/* GENERATED FILE - DO NOT EDIT.
 * Source: protocol/ecg_proto.json
 * Regenerate: python scripts/gen_protocol_constants.py */
'use strict';
(function (global) {{
  var P = {{
    PROTO_VERSION: {d["ble"]["proto_ver"]},
    FW_VERSION: '{d["ble"]["fw_ver"]}',
    CSV_COLUMNS_V1: {len(d["ble"]["frame"]["columns_v1"])},
    CSV_COLUMNS_V2: {len(d["ble"]["frame"]["columns_v2"])},
    ECGR_HEADER_SIZE: {d["ecgr"]["header_size"]},
    ECGR_VERSION_CURRENT: {d["ecgr"]["versions"]["current"]},
    ECGR_SCALE_TO_VOLTS: {d["ecgr"]["sample_scale"]["lsb"]:.1f},
    ECGR_FLAG_HAS_ABNORMAL_BITMAP: 0x01,
    ECGR_RESERVED0_ABNORMAL_IS_ASRC: 0x01,
    ASRC: {{
{chr(10).join('      %s: %s,' % (b["name"], b["bit"]) for b in asrc)}
    }},
    ASRC_META: [
{entries}
    ],
    primaryAsrc: function (asrc) {{
      for (var i = 0; i < P.ASRC_META.length; i++) {{
        if (asrc & P.ASRC_META[i].bit) return P.ASRC_META[i];
      }}
      return null;
    }}
  }};
  if (typeof module !== 'undefined' && module.exports) module.exports = P;
  global.ECGProtocol = P;
}})(typeof window !== 'undefined' ? window : globalThis);
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查生成物是否为最新")
    args = ap.parse_args()

    d = load_contract()
    targets = {
        ROOT / "experiments/esp_idf_ecg_migration/components/ecg_storage/include/storage/ecg_protocol_generated.h": c_header(d),
        ROOT / "ecg_app/lib/services/protocol_generated.dart": dart_file(d),
        ROOT / "web/js/protocol_generated.js": js_file(d),
    }

    rc = 0
    for path, text in targets.items():
        if args.check:
            existing = path.read_text(encoding="utf-8") if path.exists() else None
            if existing != text:
                print(f"STALE: {path.relative_to(ROOT)}")
                rc = 1
            else:
                print(f"OK:    {path.relative_to(ROOT)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote: {path.relative_to(ROOT)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
