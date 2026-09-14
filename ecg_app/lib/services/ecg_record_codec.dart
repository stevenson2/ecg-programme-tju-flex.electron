import 'dart:typed_data';

/* .ecgr 记录文件编解码（Contract C5，P0-3 双版本）。
 *
 * 字节布局与 protocol/ecg_proto.json / 固件
 * storage/ecg_recorder_format.h 一致：
 * - 32 字节小端头部：magic ECGR + version(1|2) + flags + 五个 uint32
 *   + reserved0(bit0 = 位图为 asrc 掩码) + 5 字节保留；
 * - 样本流 totalSamples x int16 LE，volts = int16 / 8000.0；
 * - 异常位图 durationSec x uint8：v1 = 0/1；v2 = asrc 位掩码。
 *
 * 截断策略统一为容忍：头部声明超过实际字节时解码到可用长度并置
 * truncated=true，绝不因截断返回 null；仅魔数/版本非法或头部不足
 * 32 字节返回 null。旧 v1 文件必须可解析。
 */
/// 解码后的心电记录（兼容 v1/v2）
class EcgRecord {
  final int version;
  final int sampleRate;
  final int startUnixTime;
  final int durationSec;
  final int totalSamples;
  final int abnormalSeconds;
  final bool hasBitmap;

  /// v2 位图是否为 asrc 位掩码（v1 恒 false）
  final bool bitmapIsAsrc;

  /// 逐秒异常标记，长度 = durationSec（无位图时为空）。
  /// v1: 0/1；v2: asrc 位掩码。
  final List<int> abnormalBySecond;
  final List<double> samplesV;
  final bool truncated;

  const EcgRecord({
    required this.version,
    required this.sampleRate,
    required this.startUnixTime,
    required this.durationSec,
    required this.totalSamples,
    required this.abnormalSeconds,
    required this.hasBitmap,
    required this.bitmapIsAsrc,
    required this.abnormalBySecond,
    required this.samplesV,
    this.truncated = false,
  });

  /// 规范化 asrc 视图：v1 的 0/1 -> 0x01 AI / 0x00。
  List<int> get asrcBySecond {
    if (!hasBitmap) return const [];
    if (!bitmapIsAsrc) {
      return abnormalBySecond.map((b) => b != 0 ? 0x01 : 0x00).toList();
    }
    return abnormalBySecond;
  }
}
/// .ecgr 二进制解码器（Contract C5）。
class EcgRecordCodec {
  static const int headerSize = 32;
  static const int version1 = 1;
  static const int version2 = 2;
  static const List<int> _magic = [0x45, 0x43, 0x47, 0x52]; // ECGR

  /// 固件 ADC 标定：1.0V = 8000.0 LSB（int16）
  static const double voltsPerLsb = 8000.0;

  static const int _flagHasAbnormalBitmap = 0x01;
  static const int _reserved0AbnormalIsAsrc = 0x01;

  /// 仅校验头部：魔数 + 版本 + 头部长度。
  static bool validateHeader(Uint8List bytes) {
    if (bytes.length < headerSize) return false;
    for (int i = 0; i < _magic.length; i++) {
      if (bytes[i] != _magic[i]) return false;
    }
    final v = bytes[4];
    return v == version1 || v == version2;
  }

  /// 解码完整记录；非法头部返回 null（不抛异常）。
  static EcgRecord? decode(Uint8List bytes) {
    if (!validateHeader(bytes)) return null;

    final version = bytes[4];
    final flags = bytes[5];
    final hasBitmap = (flags & _flagHasAbnormalBitmap) != 0;
    final reserved0 = bytes[26];
    final bitmapIsAsrc =
        version == version2 && (reserved0 & _reserved0AbnormalIsAsrc) != 0;

    final sampleRate = _readU32LE(bytes, 6);
    final startUnixTime = _readU32LE(bytes, 10);
    final durationSec = _readU32LE(bytes, 14);
    final headerSamples = _readU32LE(bytes, 18);
    final abnormalSeconds = _readU32LE(bytes, 22);

    /// P0-3 统一截断容忍（正向解码）：
    /// 样本流按头部 totalSamples 读，文件不足则读到字节上限；
    /// 位图从样本流之后读，缺失尾部按 0 补齐；任一处不足 truncated=true。
    final payload = bytes.length - headerSize;
    final int totalSamples;
    if (headerSamples > 0) {
      totalSamples = headerSamples * 2 <= payload ? headerSamples : payload ~/ 2;
    } else if (hasBitmap) {
      final rem = payload - durationSec;
      totalSamples = rem > 0 ? rem ~/ 2 : 0;
    } else {
      totalSamples = payload ~/ 2;
    }
    var truncated = headerSamples > 0 && totalSamples < headerSamples;

    final samplesV = List<double>.generate(totalSamples, (i) {
      final off = headerSize + i * 2;
      final u = bytes[off] | (bytes[off + 1] << 8);
      final s = u >= 0x8000 ? u - 0x10000 : u;
      return s / voltsPerLsb;
    }, growable: false);

    final abnormalBySecond = <int>[];
    if (hasBitmap) {
      final base = headerSize + totalSamples * 2;
      for (int i = 0; i < durationSec; i++) {
        final off = base + i;
        if (off < bytes.length) {
          abnormalBySecond.add(bytes[off] & 0xFF);
        } else {
          abnormalBySecond.add(0);
          truncated = true;
        }
      }
    }

    return EcgRecord(
      version: version,
      sampleRate: sampleRate,
      startUnixTime: startUnixTime,
      durationSec: durationSec,
      totalSamples: totalSamples,
      abnormalSeconds: abnormalSeconds,
      hasBitmap: hasBitmap,
      bitmapIsAsrc: bitmapIsAsrc,
      abnormalBySecond: abnormalBySecond,
      samplesV: samplesV,
      truncated: truncated,
    );
  }

  static int _readU32LE(Uint8List b, int off) =>
      b[off] | (b[off + 1] << 8) | (b[off + 2] << 16) | (b[off + 3] << 24);
}