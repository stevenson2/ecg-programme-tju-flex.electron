import 'dart:typed_data';

/**
 * @file ecg_record_codec.dart
 * @brief .ecgr 记录文件编解码（Contract C5）
 *
 * 字节布局与固件 include/storage/ecg_recorder_format.h 逐字节一致：
 *
 * ┌────────────────────────────────────────────┐
 * │ 32 字节小端头部：                            │
 * │   0-3   magic "ECGR"                        │
 * │   4     version = 1                         │
 * │   5     flags (bit0 = 含异常位图)             │
 * │   6-9   sampleRate   uint32 (=250)          │
 * │   10-13 startUnixTime uint32                │
 * │   14-17 durationSec  uint32                 │
 * │   18-21 totalSamples uint32                 │
 * │   22-25 abnormalSeconds uint32              │
 * │   26-31 reserved（零）                       │
 * ├────────────────────────────────────────────┤
 * │ 样本流：totalSamples × int16 LE              │
 * │   原始 int16 单位，固件标定 1.0V = 8000.0     │
 * │   → volts = int16 / 8000.0                  │
 * ├────────────────────────────────────────────┤
 * │ 异常位图（flags bit0=1 时）：durationSec × uint8 │
 * │   1 = 该秒异常                                │
 * └────────────────────────────────────────────┘
 *
 * 解码约定（文档化）：字节不足（截断）、魔数或版本非法时
 * decode 返回 null，不抛异常。
 */

/// 解码后的心电记录
class EcgRecord {
  /// 采样率（Hz），固件固定 250
  final int sampleRate;

  /// 录制起始 Unix 时间戳（秒）
  final int startUnixTime;

  /// 录制时长（秒）
  final int durationSec;

  /// 总样本数
  final int totalSamples;

  /// 异常秒数（头部统计字段）
  final int abnormalSeconds;

  /// 是否含异常位图（flags bit0）
  final bool hasBitmap;

  /// 逐秒异常标记，长度 = durationSec，取值 0/1（无位图时为空）
  final List<int> abnormalBySecond;

  /// 样本电压（V），长度 = totalSamples（int16 / 8000.0）
  final List<double> samplesV;

  const EcgRecord({
    required this.sampleRate,
    required this.startUnixTime,
    required this.durationSec,
    required this.totalSamples,
    required this.abnormalSeconds,
    required this.hasBitmap,
    required this.abnormalBySecond,
    required this.samplesV,
  });
}

/// .ecgr 二进制解码器（Contract C5）
class EcgRecordCodec {
  static const int headerSize = 32;
  static const int version = 1;
  static const List<int> _magic = [0x45, 0x43, 0x47, 0x52]; // 'ECGR'

  /// 固件 ADC 标定：1.0V = 8000.0 LSB（int16）
  static const double voltsPerLsb = 8000.0;

  static const int _flagHasAbnormalBitmap = 0x01;

  /// 仅校验头部：魔数 + 版本 + 头部长度
  static bool validateHeader(Uint8List bytes) {
    if (bytes.length < headerSize) return false;
    for (int i = 0; i < _magic.length; i++) {
      if (bytes[i] != _magic[i]) return false;
    }
    if (bytes[4] != version) return false;
    return true;
  }

  /// 解码完整记录。
  ///
  /// 返回 null 的情形（文档化约定，不抛异常）：
  /// - 头部非法（魔数/版本错误，或不足 32 字节）
  /// - 数据截断（实际长度 < 32 + 2×totalSamples + 位图字节数）
  /// 允许存在尾部多余字节（读取头部声明部分）。
  static EcgRecord? decode(Uint8List bytes) {
    if (!validateHeader(bytes)) return null;

    final flags = bytes[5];
    final hasBitmap = (flags & _flagHasAbnormalBitmap) != 0;

    final sampleRate = _readU32LE(bytes, 6);
    final startUnixTime = _readU32LE(bytes, 10);
    final durationSec = _readU32LE(bytes, 14);
    final totalSamples = _readU32LE(bytes, 18);
    final abnormalSeconds = _readU32LE(bytes, 22);

    // 长度自洽校验：位图字节数 = durationSec（有位图时）
    final bitmapBytes = hasBitmap ? durationSec : 0;
    final expectedLen = headerSize + totalSamples * 2 + bitmapBytes;
    if (bytes.length < expectedLen) return null;

    // 样本流：int16 LE → volts
    final samplesV = List<double>.generate(totalSamples, (i) {
      final off = headerSize + i * 2;
      final u = bytes[off] | (bytes[off + 1] << 8);
      final s = u >= 0x8000 ? u - 0x10000 : u;
      return s / voltsPerLsb;
    }, growable: false);

    // 异常位图（紧跟在样本流之后）
    final abnormalBySecond = <int>[];
    if (hasBitmap) {
      final base = headerSize + totalSamples * 2;
      for (int i = 0; i < durationSec; i++) {
        abnormalBySecond.add(bytes[base + i] & 0xFF);
      }
    }

    return EcgRecord(
      sampleRate: sampleRate,
      startUnixTime: startUnixTime,
      durationSec: durationSec,
      totalSamples: totalSamples,
      abnormalSeconds: abnormalSeconds,
      hasBitmap: hasBitmap,
      abnormalBySecond: abnormalBySecond,
      samplesV: samplesV,
    );
  }

  /// 小端读取 uint32
  static int _readU32LE(Uint8List b, int off) =>
      b[off] | (b[off + 1] << 8) | (b[off + 2] << 16) | (b[off + 3] << 24);
}
