import 'package:flutter/material.dart';
import '../models/waveform_data_source.dart';

/**
 * @file ecg_waveform.dart
 * @brief 心电波形绘制组件
 *
 * 波形绘制仅依赖数据源的 5 个只读属性（WaveformDataSource，见
 * models/waveform_data_source.dart）：实时模式由 ECGProvider 实现，
 * 记录回放由 playback_page.dart 的 PlaybackProvider 实现。
 */

class ECGWaveform extends StatelessWidget {
  final WaveformDataSource provider;

  const ECGWaveform({super.key, required this.provider});

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        return CustomPaint(
          size: Size(constraints.maxWidth, constraints.maxHeight),
          painter: _ECGWaveformPainter(
            data: provider.displayData,
            maxVal: provider.maxValue,
            minVal: provider.minValue,
            timeWindow: provider.timeWindow,
            alert: provider.hasAbnormalAlert,
          ),
        );
      },
    );
  }
}

class _ECGWaveformPainter extends CustomPainter {
  final List<double> data;
  final double maxVal;
  final double minVal;
  final int timeWindow;
  final bool alert; // AI 异常告警：波形变红 + 红色背景光晕

  _ECGWaveformPainter({
    required this.data,
    required this.maxVal,
    required this.minVal,
    required this.timeWindow,
    required this.alert,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final range = (maxVal - minVal).clamp(0.05, 10.0);
    final scaleY = size.height / range;
    final visibleCount = timeWindow * 250;
    final dx = visibleCount > 0 ? size.width / visibleCount : 1.0;

    _drawBackground(canvas, size, range);

    // AI 异常告警：叠加红色背景光晕，波形整体变红提示
    if (alert) {
      final alertPaint = Paint()
        ..color = const Color(0xFFFF5252).withValues(alpha: 0.07);
      canvas.drawRect(Offset.zero & size, alertPaint);
    }

    if (data.isNotEmpty) {
      _drawWaveform(canvas, size, data, dx, scaleY);
      _drawBaseline(canvas, size, range);
    }
  }

  void _drawBackground(Canvas canvas, Size size, double range) {
    final bgPaint = Paint()..color = const Color(0xFF0A0A0E);
    canvas.drawRect(Offset.zero & size, bgPaint);

    final gridPaint = Paint()
      ..color = const Color(0xFF1A1A30)
      ..strokeWidth = 0.5;

    // 竖网格线（每 200ms）
    final visibleCount = timeWindow * 250;
    final vLines = visibleCount > 0 ? (timeWindow * 5) : 10;
    for (int i = 1; i < vLines; i++) {
      final x = size.width * i / vLines;
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), gridPaint);
    }

    // 水平网格线（6 格）
    for (int i = 1; i < 6; i++) {
      final y = size.height * i / 6;
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }

    // 时间刻度标签
    final tp = TextPainter(textDirection: TextDirection.ltr);
    final smallFont = TextStyle(color: Color(0xFF445566), fontSize: 9);

    for (int i = 0; i <= vLines; i++) {
      if (i % (timeWindow > 3 ? 2 : 1) == 0) {
        final x = size.width * i / vLines;
        final sec = (timeWindow * i / vLines).toStringAsFixed(0);
        tp.text = TextSpan(text: '${sec}s', style: smallFont);
        tp.layout();
        tp.paint(canvas, Offset(x - tp.width / 2, size.height - 14));
      }
    }

    // 电压刻度标签
    for (int i = 0; i <= 6; i++) {
      final y = size.height * i / 6;
      final volt = (maxVal - range * i / 6).toStringAsFixed(2);
      tp.text = TextSpan(text: '${volt}V', style: smallFont);
      tp.layout();
      tp.paint(canvas, Offset(2, y - tp.height / 2));
    }
  }

  void _drawWaveform(Canvas canvas, Size size, List<double> data,
      double dx, double scaleY) {

    if (data.length < 2) return;

    // 主波形（异常告警时切换为红色）
    final waveColor = alert ? const Color(0xFFFF5252) : const Color(0xFF00E5FF);
    final paint = Paint()
      ..color = waveColor
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    // 光晕
    final glowPaint = Paint()
      ..color = waveColor.withValues(alpha: 0.15)
      ..strokeWidth = 4.0
      ..style = PaintingStyle.stroke;

    final path = Path();
    for (int i = 0; i < data.length; i++) {
      final val = data[i];
      final x = i * dx;
      final y = (size.height - (val - minVal) * scaleY).clamp(0.0, size.height);
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    canvas.drawPath(path, glowPaint);
    canvas.drawPath(path, paint);
  }

  void _drawBaseline(Canvas canvas, Size size, double range) {
    final zeroRatio = range > 0 ? maxVal / range : 0.5;
    final zeroY = size.height * (1.0 - zeroRatio);
    if (zeroY > 0 && zeroY < size.height) {
      final paint = Paint()
        ..color = const Color(0xFF334466)
        ..strokeWidth = 0.8;
      canvas.drawLine(Offset(0, zeroY), Offset(size.width, zeroY), paint);
    }
  }

  @override
  bool shouldRepaint(covariant _ECGWaveformPainter oldDelegate) => true;
}
