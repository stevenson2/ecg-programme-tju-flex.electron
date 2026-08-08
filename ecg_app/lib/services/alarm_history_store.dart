import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

import '../models/alarm_event.dart';

/**
 * @file alarm_history_store.dart
 * @brief 告警历史持久化存储器 — SharedPreferences 后端
 *
 * 以 JSON 数组格式存储在 SharedPreferences key 'alarm_history'。
 * 上限 100 条（最新优先），add 与 load 均带有截断保护。
 */

/// 告警历史持久化存储器
///
/// 提供加载、追加（含容量上限）、清空操作。
/// 所有方法均为异步，内部统一使用 `SharedPreferences.getInstance()`。
class AlarmHistoryStore {
  /// 存储 key — SharedPreferences 中的 JSON 数组键名
  static const String _key = 'alarm_history';

  /// 最大保留条数 — add 写入时截断、load 读取时防御性截断
  static const int maxEntries = 100;

  /// 加载告警历史
  ///
  /// 返回 newest-first 排列的 [AlarmEvent] 列表，最多 [maxEntries] 条。
  /// 若存储中无数据或数据格式异常，返回空列表（不抛异常）。
  Future<List<AlarmEvent>> load() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_key);
    if (raw == null || raw.isEmpty) {
      return [];
    }

    try {
      final List<dynamic> decoded = jsonDecode(raw) as List<dynamic>;
      final List<AlarmEvent> events = decoded
          .map((e) => AlarmEvent.fromJson(e as Map<String, dynamic>))
          .toList();
      // 防御性截断：即使存储中条目超过上限也只返回前 maxEntries
      if (events.length > maxEntries) {
        return events.sublist(0, maxEntries);
      }
      return events;
    } catch (_) {
      // 数据损坏时返回空列表，不阻塞 App
      return [];
    }
  }

  /// 追加一条告警事件
  ///
  /// 新条目插入列表最前端（最新优先），超出 [maxEntries] 时丢弃最旧条目，
  /// 然后写回 SharedPreferences。
  Future<void> add(AlarmEvent e) async {
    final prefs = await SharedPreferences.getInstance();
    final events = await load(); // 已有列表，已是 newest-first

    // 插入头部
    events.insert(0, e);
    // 截断超出上限的旧条目
    if (events.length > maxEntries) {
      events.removeRange(maxEntries, events.length);
    }

    // 序列化并持久化
    final encoded =
        jsonEncode(events.map((ev) => ev.toJson()).toList());
    await prefs.setString(_key, encoded);
  }

  /// 清空所有告警历史
  ///
  /// 删除 SharedPreferences 中的 `alarm_history` 键。
  Future<void> clear() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_key);
  }
}
