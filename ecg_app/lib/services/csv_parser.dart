import '../models/ecg_data.dart';

/**
 * @file csv_parser.dart
 * @brief BLE/串口 CSV 数据行解析（纯函数，便于单元测试）
 *
 * 与 ESP32 固件输出格式保持一致：
 * <clean>,<noisy>,<filtered>,<bpm>,<true_bpm>,<sqi>,<motion>,<abnormal_flag>,<confidence>
 *
 * 兼容性说明（与原 BLEService._onDataReceived 行为完全一致）：
 * - 至少 3 列（clean/noisy/filtered）才有效，不足 3 列返回 null；
 * - 第 4 列 bpm、第 8 列 abnormal、第 9 列 confidence 为可选，
 *   缺失或无法解析时取默认值（0 / 0 / 0.0）；
 * - 前三列解析失败（非数字）时返回 null，由调用方跳过该行。
 */

/// 解析一行 CSV 文本为 [ECGSample]；行无效或前三列非数字时返回 null
ECGSample? parseEcgCsvLine(String line) {
  final str = line.trim();
  if (str.isEmpty) return null;

  final parts = str.split(',');
  if (parts.length < 3) return null;

  try {
    final clean = double.parse(parts[0].trim());
    final noisy = double.parse(parts[1].trim());
    final filtered = double.parse(parts[2].trim());

    // 第 4 列：ESP32 板上心率 (可选)
    int bpm = 0;
    if (parts.length >= 4) {
      bpm = int.tryParse(parts[3].trim()) ?? 0;
    }

    // 第 8 列：AI 异常标志 (可选, 0=正常 1=异常)
    int abnormal = 0;
    if (parts.length >= 8) {
      abnormal = int.tryParse(parts[7].trim()) ?? 0;
    }

    // 第 9 列：AI 异常置信度 (可选, 0~1)
    double confidence = 0.0;
    if (parts.length >= 9) {
      confidence = double.tryParse(parts[8].trim()) ?? 0.0;
    }

    return ECGSample(clean, noisy, filtered,
        bpm: bpm, abnormal: abnormal, confidence: confidence);
  } catch (_) {
    // 解析失败，返回 null 由调用方跳过
    return null;
  }
}
