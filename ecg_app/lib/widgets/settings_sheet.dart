import 'package:flutter/material.dart';

import '../providers/settings_provider.dart';

/// 与 App 一致的主题色（info_panel 等控件同款暗色约定）
const Color _kScaffoldBg = Color(0xFF0D0D1A); // 弹窗底色
const Color _kCardBg = Color(0xFF1A1A2E);     // 卡片底色
const Color _kPrimary = Color(0xFF00BFFF);    // 高亮色（开关/滑块）

/**
 * @file settings_sheet.dart
 * @brief 告警与定时录制设置底部弹窗
 *
 * AlarmSettingsSheet 为无状态展示组件，通过 ListenableBuilder 监听
 * SettingsProvider，控件操作直接回写 provider（setDnd / setSound /
 * setVolume / setAutoClose / setRecScheduleEnabled /
 * setRecScheduleIntervalMin / setRecScheduleDurationSec），
 * 由 provider 持久化并通知刷新。
 */
class AlarmSettingsSheet extends StatelessWidget {
  final SettingsProvider settings;

  const AlarmSettingsSheet({super.key, required this.settings});

  /// 弹出告警设置模态底部弹窗
  static void show(BuildContext context, SettingsProvider settings) {
    showModalBottomSheet<void>(
      context: context,
      // 内容较高，允许撑满可用高度
      isScrollControlled: true,
      backgroundColor: _kScaffoldBg,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (_) => AlarmSettingsSheet(settings: settings),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: settings,
      builder: (context, _) {
        return SafeArea(
          child: SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // 标题 — 告警设置
                  const Row(
                    children: [
                      Icon(Icons.notifications_active_outlined,
                          color: _kPrimary, size: 22),
                      SizedBox(width: 8),
                      Text(
                        '告警设置',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  _buildSwitchTile(
                    icon: Icons.do_not_disturb_on_outlined,
                    title: '免打扰',
                    subtitle: '抑制弹窗与提示音（历史仍记录）',
                    value: settings.dndEnabled,
                    onChanged: settings.setDnd,
                  ),
                  _buildSwitchTile(
                    icon: Icons.volume_up_outlined,
                    title: '提示音',
                    subtitle: '异常告警时播放提示音',
                    value: settings.soundEnabled,
                    onChanged: settings.setSound,
                  ),
                  const SizedBox(height: 8),
                  // 音量滑块
                  _buildSliderRow(
                    icon: Icons.volume_down_outlined,
                    label: '音量',
                    valueLabel:
                        '${(settings.soundVolume * 100).round()}%',
                    slider: Slider(
                      key: const ValueKey('volume_slider'),
                      value: settings.soundVolume,
                      min: SettingsProvider.kMinVolume,
                      max: SettingsProvider.kMaxVolume,
                      activeColor: _kPrimary,
                      inactiveColor: Colors.white24,
                      onChanged: (v) => settings.setVolume(v),
                    ),
                  ),
                  // 自动关闭时长滑块
                  _buildSliderRow(
                    icon: Icons.timer_outlined,
                    label: '自动关闭时长',
                    valueLabel: '${settings.autoCloseSeconds} 秒',
                    slider: Slider(
                      key: const ValueKey('auto_close_slider'),
                      value: settings.autoCloseSeconds.toDouble(),
                      min: SettingsProvider.kMinAutoClose.toDouble(),
                      max: SettingsProvider.kMaxAutoClose.toDouble(),
                      divisions:
                          SettingsProvider.kMaxAutoClose -
                          SettingsProvider.kMinAutoClose,
                      activeColor: _kPrimary,
                      inactiveColor: Colors.white24,
                      onChanged: (v) => settings.setAutoClose(v.round()),
                    ),
                  ),
                  const Divider(height: 24, color: Color(0xFF2A2A3E)),
                  // ── 定时录制调度 ──
                  const Row(
                    children: [
                      Icon(Icons.schedule_outlined,
                          color: _kPrimary, size: 22),
                      SizedBox(width: 8),
                      Text(
                        '定时录制',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  _buildSwitchTile(
                    icon: Icons.timer_outlined,
                    title: '启用定时录制',
                    subtitle: '按固定间隔自动触发固件录制计划',
                    value: settings.recScheduleEnabled,
                    onChanged: settings.setRecScheduleEnabled,
                  ),
                  if (settings.recScheduleEnabled) ...[
                    const SizedBox(height: 4),
                    _buildSliderRow(
                      icon: Icons.hourglass_bottom_outlined,
                      label: '间隔',
                      valueLabel:
                          '${_formatIntervalLabel(settings.recScheduleIntervalMin)}',
                      slider: Slider(
                        key: const ValueKey('rec_interval_slider'),
                        value: settings.recScheduleIntervalMin.toDouble(),
                        min: SettingsProvider
                            .kMinRecScheduleIntervalMin
                            .toDouble(),
                        max: SettingsProvider
                            .kMaxRecScheduleIntervalMin
                            .toDouble(),
                        divisions:
                            (SettingsProvider.kMaxRecScheduleIntervalMin -
                                    SettingsProvider.kMinRecScheduleIntervalMin) ~/
                                10,
                        activeColor: _kPrimary,
                        inactiveColor: Colors.white24,
                        onChanged: (v) =>
                            settings.setRecScheduleIntervalMin(v.round()),
                      ),
                    ),
                    _buildSliderRow(
                      icon: Icons.fiber_manual_record_outlined,
                      label: '时长',
                      valueLabel:
                          '${settings.recScheduleDurationSec} 秒',
                      slider: Slider(
                        key: const ValueKey('rec_duration_slider'),
                        value: settings.recScheduleDurationSec.toDouble(),
                        min: SettingsProvider
                            .kMinRecScheduleDurationSec
                            .toDouble(),
                        max: SettingsProvider
                            .kMaxRecScheduleDurationSec
                            .toDouble(),
                        divisions:
                            (SettingsProvider.kMaxRecScheduleDurationSec -
                                    SettingsProvider.kMinRecScheduleDurationSec) ~/
                                5,
                        activeColor: _kPrimary,
                        inactiveColor: Colors.white24,
                        onChanged: (v) =>
                            settings.setRecScheduleDurationSec(v.round()),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  /// 开关设置行（卡片式，风格对齐 info_panel）
  Widget _buildSwitchTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required bool value,
    required ValueChanged<bool> onChanged,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      decoration: BoxDecoration(
        color: _kCardBg,
        borderRadius: BorderRadius.circular(8),
      ),
      child: SwitchListTile(
        secondary: Icon(icon, color: _kPrimary),
        title: Text(
          title,
          style: const TextStyle(color: Colors.white, fontSize: 15),
        ),
        subtitle: Text(
          subtitle,
          style: const TextStyle(color: Colors.grey, fontSize: 12),
        ),
        activeThumbColor: _kPrimary,
        value: value,
        onChanged: onChanged,
      ),
    );
  }

  /// 格式化间隔标签：≥60 分钟显示为小时
  static String _formatIntervalLabel(int minutes) {
    if (minutes >= 60) {
      final hours = minutes / 60;
      if (hours == hours.roundToDouble()) {
        return '${hours.round()} 小时';
      }
      return '${hours.toStringAsFixed(1)} 小时';
    }
    return '$minutes 分钟';
  }

  /// 滑块设置行（左侧标签 + 右侧当前值 + 滑块）
  Widget _buildSliderRow({
    required IconData icon,
    required String label,
    required String valueLabel,
    required Widget slider,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
      decoration: BoxDecoration(
        color: _kCardBg,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: _kPrimary, size: 20),
              const SizedBox(width: 8),
              Text(
                label,
                style: const TextStyle(color: Colors.white, fontSize: 15),
              ),
              const Spacer(),
              Text(
                valueLabel,
                style: const TextStyle(
                  color: _kPrimary,
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                  fontFamily: 'monospace',
                ),
              ),
            ],
          ),
          slider,
        ],
      ),
    );
  }
}

/// 弹出告警设置底部弹窗（全局辅助函数）
void showAlarmSettingsSheet(BuildContext context, SettingsProvider settings) {
  AlarmSettingsSheet.show(context, settings);
}
