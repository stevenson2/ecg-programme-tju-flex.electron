/**
 * @file waveform_data_source.dart
 * @brief 波形绘制数据源抽象（纯 Dart，无 Flutter 依赖）
 *
 * CustomPainter 绘制波形所需的全部只读输入。实时模式由 ECGProvider
 * 显式实现（implements），记录回放由 PlaybackProvider（playback_page.dart）
 * 实现——两者通过该接口与 ECGWaveform 解耦。
 */

abstract class WaveformDataSource {
  /// 当前显示窗口的数据点（伏特）
  List<double> get displayData;

  /// 显示窗口最大值（幅度缩放后的 Y 轴上限）
  double get maxValue;

  /// 显示窗口最小值（幅度缩放后的 Y 轴下限）
  double get minValue;

  /// 时间窗口（秒），决定横轴显示时长
  int get timeWindow;

  /// AI 异常告警（波形变红）；回放模式恒为 false（静默回放，不触发告警）
  bool get hasAbnormalAlert;
}
