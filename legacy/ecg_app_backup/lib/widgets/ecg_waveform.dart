import 'package:flutter/material.dart';
import '../providers/ecg_provider.dart';

/**
 * @file ecg_waveform.dart
 * @brief 实时三通道心电波形绘制组件
 *
 * 使用 CustomPainter 直接绘制，支持：
 * - 三通道叠加显示（绿/红/蓝）
 * - 自动缩放 + 用户缩放
 * - 网格背景
 * - 200ms 延迟线
 */

class ECGWaveform extends StatelessWidget {
  final ECGProvider provider;

  const ECGWaveform({super.key, required this.provider});

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        return CustomPaint(
          size: Size(constraints.maxWidth, constraints.maxHeight),
          painter: _ECGWaveformPainter(
            samples: provider.samples,
            maxVal: provider.maxValue,
            minVal: provider.minValue,
          ),
        );
      },
    );
  }
}

class _ECGWaveformPainter extends CustomPainter {
  final List samples;
  final double maxVal;
  final double minVal;

  _ECGWaveformPainter({
    required this.samples,
    required this.maxVal,
    required this.minVal,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;

    _drawBackground(canvas, size, rect);
    if (samples.isEmpty) return;

    final range = (maxVal - minVal).clamp(0.1, 10.0);
    final scaleY = size.height / range;
    final dx = size.width / 500; // 每个样本占的像素宽度

    // 绘制三通道波形
    _drawChannel(canvas, size, samples, dx, scaleY, 'clean', Colors.green);
    _drawChannel(canvas, size, samples, dx, scaleY, 'noisy', Colors.red);
    _drawChannel(canvas, size, samples, dx, scaleY, 'filtered', Colors.blue);

    // 绘制200ms延迟标记线
    _drawDelayLine(canvas, size, dx);
  }

  void _drawBackground(Canvas canvas, Size size, Rect rect) {
    // 背景
    final bgPaint = Paint()..color = const Color(0xFF0A0A0A);
    canvas.drawRect(rect, bgPaint);

    // 网格
    final gridPaint = Paint()
      ..color = const Color(0xFF1A1A2E)
      ..strokeWidth = 0.5;

    // 竖线（每 50ms = 12.5 点 @250Hz）
    for (double x = 0; x < size.width; x += size.width / 10) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), gridPaint);
    }
    // 水平线
    for (double y = 0; y < size.height; y += size.height / 6) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }

    // 基线（y=0）
    final zeroY = size.height * maxVal / (maxVal - minVal);
    final baselinePaint = Paint()
      ..color = const Color(0xFF334455)
      ..strokeWidth = 1.0;
    canvas.drawLine(
      Offset(0, zeroY),
      Offset(size.width, zeroY),
      baselinePaint,
    );
  }

  void _drawChannel(Canvas canvas, Size size, List data,
      double dx, double scaleY, String field, Color color) {

    if (data.length < 2) return;

    final paint = Paint()
      ..color = color.withOpacity(0.85)
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;

    final path = Path();
    final startIdx = data.length > 500 ? data.length - 500 : 0;

    for (int i = startIdx; i < data.length; i++) {
      final sample = data[i];
      final val = _getField(sample, field);
      final x = (i - startIdx) * dx;
      final y = size.height - (val - minVal) * scaleY;

      if (i == startIdx) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    canvas.drawPath(path, paint);
  }

  void _drawDelayLine(Canvas canvas, Size size, double dx) {
    // 200ms = 50 点 @250Hz
    final delayX = 50 * dx;
    if (delayX <= 0 || delayX > size.width) return;

    final paint = Paint()
      ..color = Colors.yellow.withOpacity(0.3)
      ..strokeWidth = 1.0;

    canvas.drawLine(
      Offset(delayX, 0),
      Offset(delayX, size.height),
      paint,
    );
  }

  double _getField(dynamic sample, String field) {
    switch (field) {
      case 'clean':
        return (sample as dynamic).clean as double;
      case 'noisy':
        return (sample as dynamic).noisy as double;
      case 'filtered':
        return (sample as dynamic).filtered as double;
      default:
        return 0;
    }
  }

  @override
  bool shouldRepaint(covariant _ECGWaveformPainter oldDelegate) => true;
}
