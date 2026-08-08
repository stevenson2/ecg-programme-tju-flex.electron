import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../providers/ecg_provider.dart';

class InfoPanel extends StatelessWidget {
  final ECGProvider provider;

  const InfoPanel({super.key, required this.provider});

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
  /// 异常：红色警告 + 置信度百分比；正常：绿色；未连接：灰色
  Widget _aiStatusWidget() {
    final alert = provider.hasAbnormalAlert;
    final hasData = provider.lastSample != null;
    final color = alert
        ? Colors.red
        : (hasData ? Colors.green : Colors.grey);
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
            alert ? Icons.warning_amber : Icons.check_circle,
            size: 14,
            color: color,
          ),
          const SizedBox(width: 4),
          Text(
            alert
                ? 'AI 异常 ${(provider.abnormalConfidence * 100).toStringAsFixed(0)}%'
                : 'AI 正常',
            style: TextStyle(
              color: color,
              fontSize: 12,
              fontWeight: alert ? FontWeight.bold : FontWeight.normal,
            ),
          ),
        ],
      ),
    );
  }
}
