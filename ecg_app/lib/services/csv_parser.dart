import '../models/ecg_data.dart';

/// BLE/串口 CSV 数据行解析（纯函数，便于单元测试）。
///
/// 与 ESP32 固件输出格式（protocol/ecg_proto.json 唯一真值源）：
/// - v1（9 列，旧固件/旧端）:
///   clean,noisy,filtered,bpm,true_bpm,sqi,motion,abnormal,confidence
/// - v2（10 列，第 10 列为 asrc 位图，向后兼容）:
///   ...,abnormal,confidence,asrc
///
/// 兼容策略（P0-2）：
/// - 至少 3 列（clean/noisy/filtered）才有效，不足 3 列返回 null；
/// - 第 8 列 abnormal 为非零即报警；第 10 列 asrc 缺省时按
///   abnormal != 0 ? 0x01 : 0x00 降级（等价 v1 行为）；
/// - 前三列解析失败（非数字）时返回 null，由调用方跳过该行。
ECGSample? parseEcgCsvLine(String line) {
  final str = line.trim();
  if (str.isEmpty) return null;

  final parts = str.split(',');
  if (parts.length < 3) return null;

  // records.idx / REC_LIST 行是恰好 5 个整数。波形帧带小数或不是 5 列。
  if (parts.length == 5 &&
      parts.every((p) => RegExp(r'^-?\d+$').hasMatch(p.trim()))) {
    return null;
  }

  try {
    final clean = double.parse(parts[0].trim());
    final noisy = double.parse(parts[1].trim());
    final filtered = double.parse(parts[2].trim());

    int bpm = 0;
    if (parts.length >= 4) {
      bpm = int.tryParse(parts[3].trim()) ?? 0;
    }

    double sqi = 0.0;
    if (parts.length >= 6) {
      sqi = double.tryParse(parts[5].trim()) ?? 0.0;
    }

    int abnormal = 0;
    if (parts.length >= 8) {
      abnormal = int.tryParse(parts[7].trim()) ?? 0;
      if (abnormal != 0) abnormal = 1;
    }

    double confidence = 0.0;
    if (parts.length >= 9) {
      confidence = double.tryParse(parts[8].trim()) ?? 0.0;
    }

    int asrc = abnormal != 0 ? 0x01 : 0x00;
    if (parts.length >= 10) {
      asrc = int.tryParse(parts[9].trim()) ?? 0;
    }

    return ECGSample(clean, noisy, filtered,
        bpm: bpm,
        sqi: sqi,
        abnormal: abnormal,
        confidence: confidence,
        asrc: asrc);
  } catch (_) {
    return null;
  }
}

/// 解析一次 BLE Notify 收到的批量帧数据（固件 2 帧以 ';' 拼接）。
/// 返回非 null 的样本列表（无效帧自动跳过）。
List<ECGSample> parseBleFrames(String raw) {
  final result = <ECGSample>[];
  for (final frame in raw.split(';')) {
    final sample = parseEcgCsvLine(frame);
    if (sample != null) result.add(sample);
  }
  return result;
}

/// 125 Hz 波形在两次到达之间少了多少个样本。
///
/// [elapsedUs] 是上一批完整帧到这一批的时间。[frameCount] 是这一批的帧数。
/// 允许多一个采样周期的抖动；缺口最多补 1 秒（125 点）。
int missingSampleCount({required int elapsedUs, required int frameCount}) {
  const periodUs = 8000; // 1e6 / 125
  if (frameCount <= 0) return 0;
  final expected = periodUs * frameCount;
  if (elapsedUs <= expected + periodUs) return 0;
  var missing = ((elapsedUs - expected) / periodUs).round();
  if (missing < 0) return 0;
  if (missing > 125) return 125;
  return missing;
}

/// 把多次 Notify 拼成以 ';'、换行或回车结束的帧。
///
/// 没有结束符的数字片段留在缓冲里，避免把 "0.125,-0.010,0." 当成一条样本。
/// 旧固件的字母命令（无结束符）如果下一块以数字、'-' 或 '.' 开头，先单独吐出，
/// 避免和波形粘成一行。
class BleNotifyAssembler {
  static const int cap = 2048;
  String _buf = '';

  void reset() => _buf = '';

  List<String> push(String chunk) {
    if (chunk.isEmpty) return const [];
    final out = <String>[];
    if (_buf.isNotEmpty &&
        !_hasTerminator(_buf) &&
        _isCommandLead(_buf) &&
        _looksWaveStart(chunk)) {
      out.add(_buf);
      _buf = '';
    }
    _buf += chunk;
    if (_buf.length > cap) {
      _buf = _buf.substring(_buf.length - cap);
    }
    out.addAll(_drain());
    return out;
  }

  List<String> _drain() {
    final out = <String>[];
    var start = 0;
    for (var i = 0; i < _buf.length; i++) {
      final c = _buf.codeUnitAt(i);
      if (c == 0x3B || c == 0x0A || c == 0x0D) {
        final frame = _buf.substring(start, i).trim();
        if (frame.isNotEmpty) out.add(frame);
        start = i + 1;
      }
    }
    _buf = _buf.substring(start);
    return out;
  }
}

bool _hasTerminator(String s) {
  return s.contains(';') || s.contains('\n') || s.contains('\r');
}

bool _isCommandLead(String s) {
  final t = s.trimLeft();
  if (t.isEmpty) return false;
  final c = t.codeUnitAt(0);
  final upper = c >= 0x41 && c <= 0x5A;
  final lower = c >= 0x61 && c <= 0x7A;
  return upper || lower;
}

bool _looksWaveStart(String chunk) {
  final c = chunk.codeUnitAt(0);
  return (c >= 0x30 && c <= 0x39) || c == 0x2D || c == 0x2E;
}
