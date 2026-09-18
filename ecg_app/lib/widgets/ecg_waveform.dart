import 'package:flutter/material.dart';
import '../config/app_theme.dart';
import '../models/waveform_data_source.dart';

/**
 * @file ecg_waveform.dart
 * @brief 心电波形绘制组件（P2-2 医用化重写）
 *
 * 医用标准（IEC 60601-2-25 显示约定）：
 *   - 走纸速度 25 mm/s：横轴 1 秒 = 25 个小格；
 *   - 增益 10 mm/mV：纵轴 1 mV = 10 个小格（垂直方向为自适应增益，
 *     见下方说明）；
 *   - 1 mm 细格 + 5 mm 粗格双层网格（方格为正方形）；
 *   - 走纸扫描头：新样本自左向右推进，未写入区域（右侧）留白；
 *   - 横轴标注绝对秒，纵轴标注 mV。
 *
 * 关于垂直增益：实时监测需要保证任何幅度下波形都可见，故纵向采用
 * 自适应增益（等效于监护仪 auto-gain）。网格方格仍保持正方形，
 * 纵向刻度按当前实际增益换算并标注真实 mV 值（不伪造"恰好 0.5mV"）。
 *
 * 性能（P2-2）：
 *   - 网格位于 RepaintBoundary 静态层，数据刷新不触发网格重绘；
 *   - shouldRepaint 差分判断（数据/尺度/告警态/窗口变化才重绘）；
 *   - 光晕与主描迹共用同一条 Path，只构造一次。
 */

/// 走纸与增益常量。
class _EcgPaper {
  _EcgPaper._();

  /// 走纸速度（mm/s）——医用标准。
  static const double speedMmPerSec = 25.0;

  /// 增益（mm/mV）——医用标准标注；纵向实际使用自适应增益。
  static const double gainMmPerMv = 10.0;

  /// 网格方格边长下限（像素），防止极限窗口下网格过密。
  /// 网格方格边长下限（像素）。手机屏上 1mm 若小于 ~6px 会糊成灰雾，
  /// 故设下限 6px；秒刻度仍按 25mm/s 准确换算。
  static const double minPxPerMm = 6.0;
}

class ECGWaveform extends StatelessWidget {
  final WaveformDataSource provider;

  const ECGWaveform({super.key, required this.provider});

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final size = Size(constraints.maxWidth, constraints.maxHeight);
        return Stack(
          fit: StackFit.expand,
          children: [
            // 静态层：ECG 标准网格（不随数据刷新重绘）
            RepaintBoundary(
              child: CustomPaint(
                painter: _EcgGridPainter(timeWindow: provider.timeWindow),
                size: size,
              ),
            ),
            // 动态层：波形描迹
            CustomPaint(
              painter: _EcgTracePainter(
                data: provider.displayData,
                maxVal: provider.maxValue,
                minVal: provider.minValue,
                timeWindow: provider.timeWindow,
                samplesPerSecond: provider.samplesPerSecond,
                alert: provider.hasAbnormalAlert,
              ),
              size: size,
            ),
          ],
        );
      },
    );
  }
}

/// ECG 标准网格 + 刻度（静态层：数据刷新不重绘）。
class _EcgGridPainter extends CustomPainter {
  final int timeWindow;

  const _EcgGridPainter({required this.timeWindow});

  /// 每毫米像素数：使横轴满足 25 mm/s（1 秒 = 25 小格）。
  double _pxPerMm(Size size) {
    final paperWidthMm = _EcgPaper.speedMmPerSec * timeWindow;
    if (paperWidthMm <= 0) return _EcgPaper.minPxPerMm;
    return (size.width / paperWidthMm).clamp(_EcgPaper.minPxPerMm, 100.0);
  }

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(
      Offset.zero & size,
      Paint()..color = AppColors.traceBackground,
    );

    final pxPerMm = _pxPerMm(size);
    final minor = pxPerMm; // 1 mm
    final major = pxPerMm * 5; // 5 mm

    // 真机 DPR 下 0.5px 会被抗锯齿摊薄到看不见；用 1.0px + 关闭抗锯齿保证实心细格。
    final minorPaint = Paint()
      ..color = AppColors.gridMinor
      ..isAntiAlias = false
      ..strokeWidth = 1.0;
    final majorPaint = Paint()
      ..color = AppColors.gridMajor
      ..isAntiAlias = false
      ..strokeWidth = 1.6;

    // 细格（竖 + 横）
    for (double x = 0; x <= size.width; x += minor) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), minorPaint);
    }
    for (double y = 0; y <= size.height; y += minor) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), minorPaint);
    }
    // 粗格（竖 + 横）
    for (double x = 0; x <= size.width; x += major) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), majorPaint);
    }
    for (double y = 0; y <= size.height; y += major) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), majorPaint);
    }

    // ---- 刻度文字 ----
    final labelStyle = TextStyle(
      color: AppColors.traceLabel,
      fontSize: AppFontSize.xxs,
    );
    final tp = TextPainter(textDirection: TextDirection.ltr);

    // 横轴：每秒一条（25 mm）
    final secPx = _EcgPaper.speedMmPerSec * pxPerMm;
    final totalSec = (size.width / secPx).floor();
    for (int s = 0; s <= totalSec; s++) {
      tp.text = TextSpan(text: '${s}s', style: labelStyle);
      tp.layout();
      tp.paint(canvas, Offset(s * secPx + 2, size.height - tp.height - 1));
    }

    // 走纸/增益注记（监护仪式角标；真实 mV 数值由波形层按实际增益标注）
    // 标准增益 10 mm/mV：据此换算"1 mV 等效高度"，供波形层标注参考。
    final mvEquivalentPx = _EcgPaper.gainMmPerMv * pxPerMm;
    tp.text = TextSpan(
      text: '${_EcgPaper.speedMmPerSec.toInt()}mm/s  '
          '${_EcgPaper.gainMmPerMv.toInt()}mm/mV'
          ' (1mV≈${mvEquivalentPx.round()}px)',
      style: labelStyle,
    );
    tp.layout();
    tp.paint(canvas, Offset(size.width - tp.width - 3, 2));
  }

  @override
  bool shouldRepaint(covariant _EcgGridPainter old) =>
      old.timeWindow != timeWindow;
}

/// 波形描迹（动态层）。
class _EcgTracePainter extends CustomPainter {
  final List<double> data;
  final double maxVal;
  final double minVal;
  final int timeWindow;
  final int samplesPerSecond;
  final bool alert;

  _EcgTracePainter({
    required this.data,
    required this.maxVal,
    required this.minVal,
    required this.timeWindow,
    required this.samplesPerSecond,
    required this.alert,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final span = maxVal - minVal;
    if (span <= 0) return;

    // 纵向自适应增益：可见范围铺满绘图区高度
    final pxPerMv = size.height / span;
    final zeroY = (1 - (0 - minVal) / span) * size.height;

    // 零基线
    if (zeroY >= 0 && zeroY <= size.height) {
      canvas.drawLine(
        Offset(0, zeroY),
        Offset(size.width, zeroY),
        Paint()
          ..color = AppColors.traceBaseline
          ..strokeWidth = 0.8,
      );
    }

    if (data.isEmpty) return;

    // 告警底色光晕
    if (alert) {
      canvas.drawRect(
        Offset.zero & size,
        Paint()..color = AppColors.alert.withValues(alpha: 0.07),
      );
    }

    // ---- 走纸扫描头：新样本自左向右推进，右侧留白 ----
    final totalSlots = timeWindow * samplesPerSecond;
    if (totalSlots <= 0) return;
    final dx = size.width / totalSlots;
    final n = data.length;

    // 填充期：锚定左端，头部在 n*dx 处，右侧留白；
    // 满窗后：显示最近 totalSlots 个样本（自左端铺满，头部在右端）。
    final int firstIndex = n > totalSlots ? n - totalSlots : 0;
    final int lastIndex = n;
    final double xBase = n > totalSlots ? 0.0 : 0.0;

    final traceColor = alert ? AppColors.alert : AppColors.traceLine;
    final path = Path();
    for (int i = firstIndex; i < lastIndex; i++) {
      final x = xBase + (i - firstIndex) * dx;
      final y = (size.height - (data[i] - minVal) * pxPerMv)
          .clamp(0.0, size.height);
      if (i == firstIndex) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    // 光晕 + 主描迹：同一 Path 画两遍
    canvas.drawPath(
      path,
      Paint()
        ..color = traceColor.withValues(alpha: 0.15)
        ..strokeWidth = AppSizes.traceGlowStroke
        ..style = PaintingStyle.stroke
        ..strokeJoin = StrokeJoin.round,
    );
    canvas.drawPath(
      path,
      Paint()
        ..color = traceColor
        ..strokeWidth = AppSizes.traceStroke
        ..style = PaintingStyle.stroke
        ..strokeJoin = StrokeJoin.round,
    );

    // ---- 纵轴 mV 标注（按当前实际增益换算，保证"标注 = 实际"）----
    // 纵向自适应增益下，绘图区高度对应 span（mV），中点为 0。
    final mvPerPx = span / size.height;
    final labelStyle = TextStyle(
      color: AppColors.traceLabel,
      fontSize: AppFontSize.xxs,
    );
    final tp = TextPainter(textDirection: TextDirection.ltr);
    // 每 5 mm 一格标注（与粗格对齐：粗格像素 = 5 * pxPerMm）
    final paperWidthMm = _EcgPaper.speedMmPerSec * timeWindow;
    final pxPerMm = paperWidthMm > 0
        ? (size.width / paperWidthMm).clamp(_EcgPaper.minPxPerMm, 100.0)
        : _EcgPaper.minPxPerMm;
    final stepPx = pxPerMm * 5;
    for (double y = 0; y <= size.height; y += stepPx) {
      final mv = (size.height / 2 - y) * mvPerPx;
      tp.text = TextSpan(text: mv.toStringAsFixed(2), style: labelStyle);
      tp.layout();
      tp.paint(canvas, Offset(2, y + 1));
    }
  }

  @override
  bool shouldRepaint(covariant _EcgTracePainter old) =>
      old.data.length != data.length ||
      old.maxVal != maxVal ||
      old.minVal != minVal ||
      old.alert != alert ||
      old.timeWindow != timeWindow ||
      old.samplesPerSecond != samplesPerSecond ||
      !identical(old.data, data);
}
