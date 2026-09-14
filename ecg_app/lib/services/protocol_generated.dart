// GENERATED FILE - DO NOT EDIT.
// Source: protocol/ecg_proto.json
// Regenerate: python scripts/gen_protocol_constants.py
// ignore_for_file: constant_identifier_names

/// ESP32-ECG 协议契约常量（P0-1 单一真值源）。
class ProtocolGenerated {
  ProtocolGenerated._();

  static const int protoVersion = 2;
  static const String fwVersion = '2.0.0';
  static const String bleFramesPerNotifyHint = '2';
  static const int ecgrHeaderSize = 32;
  static const int ecgrVersionCurrent = 2;
  static const double ecgrScaleToVolts = 8000.0;
  static const int ecgrFlagHasAbnormalBitmap = 0x01;
  static const int ecgrReserved0AbnormalIsAsrc = 0x01;
  static const int csvColumnsV1 = 9;
  static const int csvColumnsV2 = 10;

  /// asrc 位定义（BLE 第 10 列 / ECGR v2 位图共用）。
  static const int leadoff = 0x10; // 电极脱落
  static const int vf = 0x08; // VF / VT 疑似
  static const int flat = 0x04; // 时间停搏（疑似电极脱落）
  static const int rs = 0x02; // 停搏 / 过缓过速
  static const int ai = 0x01; // AI 异常

  /// UI 分级文案，顺序即优先级（高 -> 低）。
  static const List<AsrcMeta> asrcMeta = <AsrcMeta>[
    AsrcMeta(bit: 0x10, name: 'LEADOFF', label: '电极脱落', level: 'warning'),
    AsrcMeta(bit: 0x08, name: 'VF', label: 'VF / VT 疑似', level: 'critical'),
    AsrcMeta(bit: 0x04, name: 'FLAT', label: '时间停搏（疑似电极脱落）', level: 'critical'),
    AsrcMeta(bit: 0x02, name: 'RS', label: '停搏 / 过缓过速', level: 'critical'),
    AsrcMeta(bit: 0x01, name: 'AI', label: 'AI 异常', level: 'info'),
  ];

  static bool isKnownAsrc(int asrc) => (asrc & 0x1F) != 0;

  /// 返回最高优先级的 asrc 元数据；无命中返回 null。
  static AsrcMeta? primaryAsrc(int asrc) {
    for (final m in asrcMeta) {
      if ((asrc & m.bit) != 0) return m;
    }
    return null;
  }
}

class AsrcMeta {
  final int bit;
  final String name;
  final String label;
  final String level; // info | warning | critical
  const AsrcMeta({required this.bit, required this.name, required this.label, required this.level});
}
