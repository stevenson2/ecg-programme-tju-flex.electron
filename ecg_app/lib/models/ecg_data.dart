/// 心电数据模型：一个样本点包含三通道值与板上状态。
class ECGSample {
  final double clean;
  final double noisy;
  final double filtered;

  /// ESP32 板上心率检测值 (bpm)
  final int bpm;

  /// 报警标志：v1=AI confirmed；v2=固件擎住报警位；非零即报警
  final int abnormal;

  /// SQI（0~1）
  final double sqi;

  /// AI 异常置信度（0~1）
  final double confidence;

  /// asrc 报警源位图（P0-2）：
  /// 0x01 AI / 0x02 RS / 0x04 FLAT / 0x08 VF / 0x10 LEADOFF。
  /// 旧 9 列帧降级为 abnormal != 0 ? 0x01 : 0x00。
  final int asrc;

  const ECGSample(this.clean, this.noisy, this.filtered,
      {this.bpm = 0,
      this.sqi = 0.0,
      this.abnormal = 0,
      this.confidence = 0.0,
      this.asrc = 0});

  @override
  String toString() =>
      'ECGSample(clean=$clean, noisy=$noisy, filtered=$filtered, bpm=$bpm, '
      'sqi=$sqi, abnormal=$abnormal, confidence=$confidence, asrc=$asrc)';
}

/// 心电统计信息（用于界面显示）。
class ECGStats {
  final double heartRate;

  /// 0.0 ~ 1.0
  final double signalQuality;
  final int samplesPerSecond;

  const ECGStats({
    this.heartRate = 0,
    this.signalQuality = 0,
    this.samplesPerSecond = 0,
  });
}
