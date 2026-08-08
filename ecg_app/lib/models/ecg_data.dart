/**
 * @file ecg_data.dart
 * @brief 心电数据模型，一个样本点包含三通道值
 */

class ECGSample {
  final double clean;
  final double noisy;
  final double filtered;
  final int bpm;           /**< ESP32 板上心率检测值 */
  final int abnormal;      /**< AI 异常标志 (0=正常, 1=异常) */
  final double confidence; /**< AI 异常置信度 (0~1) */

  const ECGSample(this.clean, this.noisy, this.filtered,
      {this.bpm = 0, this.abnormal = 0, this.confidence = 0.0});

  @override
  String toString() =>
      'ECGSample(clean=$clean, noisy=$noisy, filtered=$filtered, bpm=$bpm, '
      'abnormal=$abnormal, confidence=$confidence)';
}

/**
 * @brief 心电统计信息（用于界面显示）
 */
class ECGStats {
  final double heartRate;
  final double signalQuality; // 0.0 ~ 1.0
  final int samplesPerSecond;

  const ECGStats({
    this.heartRate = 0,
    this.signalQuality = 0,
    this.samplesPerSecond = 0,
  });
}
