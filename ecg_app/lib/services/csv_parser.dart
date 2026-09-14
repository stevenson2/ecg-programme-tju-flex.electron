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
