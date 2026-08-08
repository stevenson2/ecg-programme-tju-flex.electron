import 'dart:async';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../providers/ecg_provider.dart';
import '../providers/settings_provider.dart';

/// 与 App 一致的暗色主题（info_panel / settings_sheet 同款约定）
const Color _kCardBg = Color(0xFF1A1A2E);     // 卡片底色（弹窗背景）
const Color _kAlertRed = Color(0xFFE53935);   // 告警红色（标题/进度条/按钮）
const double _kDialogRadius = 16.0;           // 浮层圆角（对齐底部弹窗约定）

/**
 * @file alarm_dialog.dart
 * @brief 异常心律告警弹窗（红色主题）
 *
 * AlarmDialog 为无状态展示组件，数据全部来自注入的 provider：
 * - ECGProvider（Contract C1）：alarmPeakConfidence / lastBpm / alarmTriggerTime
 * - SettingsProvider（Contract C2）：autoCloseSeconds（自动关闭时长）
 *
 * 对外 API（集成任务按此名称接线）：
 * - AlarmDialog：弹窗内容组件
 * - showAlarmDialog：弹出模态告警弹窗（barrierDismissible=false，
 *   autoCloseSeconds 秒后自动关闭）
 *
 * 自动关闭定时器生命周期由内部 StatefulWidget 包装管理：弹窗被其他方式
 * 关闭（如点击确认）时在 dispose 中取消，避免定时器失效后误 pop 根路由。
 */
class AlarmDialog extends StatelessWidget {
  const AlarmDialog({super.key, required this.provider, required this.settings});

  final ECGProvider provider;
  final SettingsProvider settings;

  @override
  Widget build(BuildContext context) {
    // 置信度百分比（0.87 → 87%）
    final confidencePercent = (provider.alarmPeakConfidence * 100).round();

    return AlertDialog(
      backgroundColor: _kCardBg,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(_kDialogRadius),
      ),
      title: const Row(
        children: [
          Icon(Icons.warning_amber_rounded, color: _kAlertRed, size: 26),
          SizedBox(width: 8),
          Text(
            '⚠ 异常心律',
            style: TextStyle(
              color: _kAlertRed,
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 置信度：百分比文本 + 红色进度条（暗色背景上加浅色底槽保证可见）
          Text(
            '置信度 $confidencePercent%',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 15,
              fontFamily: 'monospace',
            ),
          ),
          const SizedBox(height: 8),
          LinearProgressIndicator(
            value: provider.alarmPeakConfidence.clamp(0.0, 1.0),
            color: _kAlertRed,
            backgroundColor: Colors.white24,
            minHeight: 8,
            borderRadius: BorderRadius.circular(4),
          ),
          const SizedBox(height: 16),
          // 心率（ESP32 板上算法锁定值）
          Text(
            '心率 ${provider.lastBpm} BPM',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 15,
              fontFamily: 'monospace',
            ),
          ),
          const SizedBox(height: 8),
          // 触发时间（intl HH:mm:ss）
          Text(
            '触发时间 ${DateFormat('HH:mm:ss').format(provider.alarmTriggerTime!)}',
            style: const TextStyle(
              color: Colors.grey,
              fontSize: 13,
              fontFamily: 'monospace',
            ),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () {
            Navigator.pop(context);
            provider.confirmAlarm();
          },
          child: const Text(
            '确认',
            style: TextStyle(
              color: _kAlertRed,
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
      ],
    );
  }
}

/// 弹出异常心律告警弹窗（模态，不可点外部关闭，autoCloseSeconds 秒后自动关闭）
Future<void> showAlarmDialog(
    BuildContext context, ECGProvider provider, SettingsProvider settings) {
  return showDialog<void>(
    context: context,
    barrierDismissible: false,
    builder: (_) => _AutoCloseAlarmDialog(provider: provider, settings: settings),
  );
}

/// 内部包装：负责自动关闭定时器生命周期
/// 弹窗被确认按钮等途径关闭时，State.dispose 取消定时器，防止残留定时器
/// 在路由已弹出后再次 Navigator.pop（会误弹根路由）。
class _AutoCloseAlarmDialog extends StatefulWidget {
  const _AutoCloseAlarmDialog({required this.provider, required this.settings});

  final ECGProvider provider;
  final SettingsProvider settings;

  @override
  State<_AutoCloseAlarmDialog> createState() => _AutoCloseAlarmDialogState();
}

class _AutoCloseAlarmDialogState extends State<_AutoCloseAlarmDialog> {
  Timer? _autoCloseTimer;

  @override
  void initState() {
    super.initState();
    _autoCloseTimer = Timer(
      Duration(seconds: widget.settings.autoCloseSeconds),
      () {
        if (mounted) Navigator.of(context).pop();
      },
    );
  }

  @override
  void dispose() {
    _autoCloseTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlarmDialog(provider: widget.provider, settings: widget.settings);
  }
}
