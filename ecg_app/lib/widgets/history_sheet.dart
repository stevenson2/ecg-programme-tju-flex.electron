import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../models/alarm_event.dart';

/**
 * @file history_sheet.dart
 * @brief 告警历史底部弹窗 — 列表展示 + 清除全部
 *
 * 以模态底部弹窗形式展示 HistorySheet 组件。
 * 每条记录显示触发时间、持续时长、峰值置信度、平均心率、恢复方式。
 * 空状态显示"暂无报警记录"。
 */

/// 告警历史底部分页组件
///
/// 接收已加载的 [events] 列表（newest-first）与 [onClear] 回调。
/// 作为 [showHistorySheet] 的 `builder` 内容嵌入底部弹窗。
class HistorySheet extends StatelessWidget {
  /// 告警事件列表（已按最新优先排序）
  final List<AlarmEvent> events;

  /// 清除全部按钮回调 — 调用方负责 store.clear() + 刷新 UI
  final VoidCallback onClear;

  const HistorySheet({super.key, required this.events, required this.onClear});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      // 修复 2026-08-10: 缺 SafeArea 致标题顶到手机状态栏 (用户反馈)
      child: Container(
        // 底部弹窗背景色，与 App 暗色主题一致
        decoration: const BoxDecoration(
          color: Color(0xFF0D0D1A),
          borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // ── 标题栏 ──
            _buildHeader(),
            // ── 列表内容 ──
            Flexible(
              child: events.isEmpty ? _buildEmpty() : _buildList(),
            ),
          ],
        ),
      ),
    );
  }

  /// 标题栏：左侧标题 + 右侧清除按钮
  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: const BoxDecoration(
        border: Border(
          bottom: BorderSide(color: Color(0xFF2A2A3E), width: 0.5),
        ),
      ),
      child: Row(
        children: [
          const Icon(Icons.history, color: Color(0xFF00BFFF), size: 20),
          const SizedBox(width: 8),
          const Text(
            '告警历史',
            style: TextStyle(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          const Spacer(),
          if (events.isNotEmpty)
            TextButton.icon(
              onPressed: onClear,
              icon: const Icon(Icons.delete_outline, size: 16),
              label: const Text('清除记录'),
              style: TextButton.styleFrom(
                foregroundColor: const Color(0xFFE53935),
                padding:
                    const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                textStyle: const TextStyle(fontSize: 13),
              ),
            ),
        ],
      ),
    );
  }

  /// 空状态
  Widget _buildEmpty() {
    return const Padding(
      padding: EdgeInsets.symmetric(vertical: 48),
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.inbox_outlined, size: 48, color: Colors.grey),
            SizedBox(height: 12),
            Text(
              '暂无报警记录',
              style: TextStyle(color: Colors.grey, fontSize: 14),
            ),
          ],
        ),
      ),
    );
  }

  /// 事件列表
  Widget _buildList() {
    return ListView.builder(
      shrinkWrap: true,
      padding: const EdgeInsets.symmetric(vertical: 8),
      itemCount: events.length,
      itemBuilder: (context, index) => _buildCard(events[index]),
    );
  }

  /// 单条告警卡片
  Widget _buildCard(AlarmEvent event) {
    final timeStr =
        DateFormat('MM-dd HH:mm:ss').format(event.triggerTime.toLocal());
    final durMin = event.duration.inMinutes;
    final durSec = event.duration.inSeconds.remainder(60);
    final durStr =
        '${durMin.toString().padLeft(2, '0')}:${durSec.toString().padLeft(2, '0')}';
    final confPct = (event.peakConfidence * 100).toStringAsFixed(0);
    final bpmStr = event.avgBpm.toStringAsFixed(0);
    final isManual = event.recoveryMethod == 'user_confirm';

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── 警告图标 ──
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              color: const Color(0xFFE53935).withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Icon(
              Icons.warning_amber,
              color: Color(0xFFE53935),
              size: 20,
            ),
          ),
          const SizedBox(width: 12),
          // ── 信息区 ──
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // 第一行：触发时间 + 恢复标签
                Row(
                  children: [
                    Text(
                      timeStr,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 13,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    const Spacer(),
                    _recoveryChip(isManual),
                  ],
                ),
                const SizedBox(height: 6),
                // 第二行：三项指标
                Row(
                  children: [
                    _metricRow('时长', durStr),
                    const SizedBox(width: 16),
                    _metricRow('峰值置信度', '$confPct%'),
                    const SizedBox(width: 16),
                    _metricRow('平均心率', '$bpmStr BPM'),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// 恢复方式标签
  Widget _recoveryChip(bool isManual) {
    final label = isManual ? '手动确认' : '自动恢复';
    final chipColor = isManual
        ? const Color(0xFF4CAF50) // 绿色：手动确认
        : Colors.grey; // 灰色：自动恢复

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: chipColor.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: chipColor,
          fontSize: 11,
          fontWeight: FontWeight.w500,
        ),
      ),
    );
  }

  /// 指标行：label + value
  Widget _metricRow(String label, String value) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          '$label ',
          style: const TextStyle(color: Colors.grey, fontSize: 11),
        ),
        Text(
          value,
          style: const TextStyle(
            color: Color(0xFF00BFFF),
            fontSize: 11,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }
}

/// 弹出告警历史底部弹窗
///
/// 调用示例：
/// ```dart
/// showHistorySheet(context, events, () async {
///   await store.clear();
///   // 刷新 UI 等
/// });
/// ```
void showHistorySheet(
  BuildContext context,
  List<AlarmEvent> events,
  VoidCallback onClear,
) {
  showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    backgroundColor: Colors.transparent,
    builder: (_) => HistorySheet(events: events, onClear: onClear),
  );
}
