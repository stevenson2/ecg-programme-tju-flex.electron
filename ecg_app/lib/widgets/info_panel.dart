import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../providers/ecg_provider.dart';

class InfoPanel extends StatelessWidget {
  final ECGProvider provider;
  final int alarmCount;

  const InfoPanel({super.key, required this.provider, this.alarmCount = 0});

  @override
  Widget build(BuildContext context) {
    final last = provider.lastSample;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        children: [
          Row(
            children: [
              _heartRateWidget(),
              const Spacer(),
              _connectionWidget(),
            ],
          ),
          const SizedBox(height: 6),
          Row(
            children: [
              if (last != null)
                _valueChip('Filter', provider.displayChannel == 'clean'
                    ? last.clean : provider.displayChannel == 'noisy'
                    ? last.noisy : last.filtered, Colors.cyan),
              const SizedBox(width: 8),
              _aiStatusWidget(),
              const SizedBox(width: 8),
              Flexible(
                child: Text(
                  '${provider.timeWindow}s  |  ${(provider.amplitudeScale * 100).toStringAsFixed(0)}%',
                  style: const TextStyle(color: Colors.grey, fontSize: 12),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              const Spacer(),
              Text(
                '${provider.bufferSize} pts',
                style: const TextStyle(color: Colors.grey, fontSize: 11),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _heartRateWidget() {
    final hr = provider.heartRate;
    return Row(
      children: [
        Icon(
          Icons.favorite,
          color: hr > 0 ? Colors.red : Colors.grey,
          size: 28,
        ),
        const SizedBox(width: 8),
        Text(
          hr > 0 ? '${hr.toStringAsFixed(0)} BPM' : '-- BPM',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 28,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

  Widget _connectionWidget() {
    final connected = provider.isConnected;
    return Row(
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: connected ? Colors.green : Colors.red,
          ),
        ),
        const SizedBox(width: 6),
        Text(
          connected ? '已连接' : '未连接',
          style: TextStyle(
            color: connected ? Colors.green : Colors.red,
            fontSize: 14,
          ),
        ),
      ],
    );
  }

  Widget _valueChip(String label, double value, Color color) {
    final fmt = NumberFormat('0.000');
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        '$label: ${fmt.format(value)}V',
        style: TextStyle(
          color: color,
          fontSize: 12,
          fontFamily: 'monospace',
        ),
      ),
    );
  }

  /// AI 异常状态指示（ESP32 板上 TFLite Micro 推理结果）
  /// 异常：红色警告呼吸动画 + 置信度百分比 + 报警计数；正常：绿色；未连接：灰色
  Widget _aiStatusWidget() {
    final alert = provider.hasAbnormalAlert;
    final hasData = provider.lastSample != null;

    if (alert) {
      return _BreathingWarningChip(
        confidence: provider.abnormalConfidence,
        alarmCount: alarmCount,
      );
    }

    final color = hasData ? Colors.green : Colors.grey;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.check_circle,
            size: 14,
            color: color,
          ),
          const SizedBox(width: 4),
          Text(
            'AI 正常',
            style: TextStyle(
              color: color,
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }
}

/// 异常告警呼吸动画芯片（红色脉冲 + 置信度 + 报警次数）
class _BreathingWarningChip extends StatefulWidget {
  final double confidence;
  final int alarmCount;

  const _BreathingWarningChip({
    required this.confidence,
    required this.alarmCount,
  });

  @override
  State<_BreathingWarningChip> createState() => _BreathingWarningChipState();
}

class _BreathingWarningChipState extends State<_BreathingWarningChip>
    with SingleTickerProviderStateMixin {
  late final AnimationController _breathController;

  @override
  void initState() {
    super.initState();
    _breathController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _breathController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final confPct = (widget.confidence * 100).toStringAsFixed(0);
    final countSuffix =
        widget.alarmCount > 0 ? ' · 第 ${widget.alarmCount} 次' : '';

    return AnimatedBuilder(
      animation: _breathController,
      builder: (context, child) {
        final opacity = 0.55 + 0.45 * _breathController.value;
        final scale = 1.0 + 0.04 * _breathController.value;
        return Transform.scale(
          scale: scale,
          child: Opacity(
            opacity: opacity,
            child: child,
          ),
        );
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: BoxDecoration(
          color: Colors.red.withValues(alpha: 0.15),
          borderRadius: BorderRadius.circular(4),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.warning_amber, size: 14, color: Colors.red),
            const SizedBox(width: 4),
            Text(
              'AI 异常 $confPct%$countSuffix',
              style: const TextStyle(
                color: Colors.red,
                fontSize: 12,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
