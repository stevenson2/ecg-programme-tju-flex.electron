import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:ecg_app/models/alarm_event.dart';
import 'package:ecg_app/services/alarm_history_store.dart';
import 'package:ecg_app/widgets/history_sheet.dart';

// @brief AlarmHistoryStore 与 HistorySheet 的 TDD 测试
//
// 覆盖：
// - add + load 往返（SharedPreferences 持久化，最新优先）
// - 容量上限 100（add 侧截断 + load 侧防御）
// - toJson/fromJson 保真（recoveryMethod 与 duration ms 保留）
// - clear() 清空
// - 控件渲染：2 条记录 / 空状态 / 恢复方式标签
// - showHistorySheet 模态弹窗
// - onClear 回调触发
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  /// 辅助：构造固定字段的 AlarmEvent（避免重复样板）
  AlarmEvent makeEvent({int minutesAgo = 0, String recovery = 'user_confirm'}) {
    return AlarmEvent(
      triggerTime: DateTime(2026, 8, 8, 14, 30, 0).subtract(
        Duration(minutes: minutesAgo),
      ),
      duration: const Duration(seconds: 12, milliseconds: 340),
      peakConfidence: 0.87,
      avgBpm: 75.0,
      recoveryMethod: recovery,
    );
  }

  // ════════════ (a) add + load 往返 ════════════
  group('AlarmHistoryStore add + load 往返', () {
    test('空存储时 load() 返回空列表', () async {
      SharedPreferences.setMockInitialValues({});
      final store = AlarmHistoryStore();
      final events = await store.load();
      expect(events, isEmpty);
    });

    test('add 2 条事件后 load() 返回最新优先', () async {
      SharedPreferences.setMockInitialValues({});
      final store = AlarmHistoryStore();
      final e1 = makeEvent(minutesAgo: 5); // 较早
      final e2 = makeEvent(minutesAgo: 1); // 较新

      await store.add(e1);
      await store.add(e2);

      final events = await store.load();
      expect(events.length, 2);
      // 最新优先：e2 在前，e1 在后
      // AlarmEvent.toJson 使用 toUtc() 序列化，往返后返回 UTC 时间
      expect(events[0].triggerTime, e2.triggerTime.toUtc());
      expect(events[1].triggerTime, e1.triggerTime.toUtc());
    });

    test('新 store 实例可读回已持久化数据（SharedPreferences 往返）', () async {
      SharedPreferences.setMockInitialValues({});
      final store1 = AlarmHistoryStore();
      await store1.add(makeEvent(minutesAgo: 10));
      await store1.add(makeEvent(minutesAgo: 3));

      // 新实例 load 应读回相同数据
      final store2 = AlarmHistoryStore();
      final events = await store2.load();
      expect(events.length, 2);
      // AlarmEvent.toJson → toUtc() 序列化，往返后返回 UTC
      expect(events[0].triggerTime,
          makeEvent(minutesAgo: 3).triggerTime.toUtc());
      expect(events[1].triggerTime,
          makeEvent(minutesAgo: 10).triggerTime.toUtc());
    });
  });

  // ════════════ (b) 容量上限 100 ════════════
  group('AlarmHistoryStore 容量上限', () {
    test('add 105 条 → load 返回恰好 100 条（最新 100 条保留）', () async {
      SharedPreferences.setMockInitialValues({});
      final store = AlarmHistoryStore();

      // 添加 105 条，index 越大越新
      for (int i = 0; i < 105; i++) {
        await store.add(AlarmEvent(
          triggerTime: DateTime(2026, 8, 8).add(Duration(minutes: i)),
          duration: const Duration(seconds: 10),
          peakConfidence: 0.5 + i * 0.001,
          avgBpm: 70.0 + i * 0.1,
          recoveryMethod: 'user_confirm',
        ));
      }

      final events = await store.load();
      expect(events.length, 100, reason: '应恰好保留 100 条');

      // 最老的 5 条（i=0~4）应被丢弃，最新的是 i=104
      // toJson → toUtc() 序列化，比较 epoch ms 避免时区歧义
      expect(
        events.first.triggerTime.millisecondsSinceEpoch,
        DateTime(2026, 8, 8)
            .add(const Duration(minutes: 104))
            .toUtc()
            .millisecondsSinceEpoch,
      );
      // 最老保留的应是 i=5（第 6 条，因为丢弃了前 5 条）
      expect(
        events.last.triggerTime.millisecondsSinceEpoch,
        DateTime(2026, 8, 8)
            .add(const Duration(minutes: 5))
            .toUtc()
            .millisecondsSinceEpoch,
      );
    });

    test('load 防御性截断：即使存储中多于 100 条也只返回 100', () async {
      SharedPreferences.setMockInitialValues({});
      // 直接用 SharedPreferences 写 110 条（绕过 add 的截断，测试 load 防御）
      final prefs = await SharedPreferences.getInstance();
      final List<Map<String, dynamic>> raw = [];
      for (int i = 0; i < 110; i++) {
        raw.add(AlarmEvent(
          triggerTime: DateTime(2026, 8, 8).add(Duration(minutes: i)),
          duration: const Duration(seconds: 10),
          peakConfidence: 0.8,
          avgBpm: 72.0,
          recoveryMethod: 'signal_normal',
        ).toJson());
      }
      await prefs.setString('alarm_history', _jsonEncode(raw));

      final store = AlarmHistoryStore();
      final events = await store.load();
      expect(events.length, 100, reason: 'load 防御性截断到 100');
    });
  });

  // ════════════ (c) toJson/fromJson 保真 ════════════
  group('AlarmHistoryStore JSON 保真', () {
    test('recoveryMethod 字段经 store 持久化往返后保留', () async {
      SharedPreferences.setMockInitialValues({});
      final store = AlarmHistoryStore();
      await store.add(makeEvent(minutesAgo: 0, recovery: 'signal_normal'));
      await store.add(makeEvent(minutesAgo: 1, recovery: 'user_confirm'));

      final events = await store.load();
      expect(events[0].recoveryMethod, 'user_confirm');
      expect(events[1].recoveryMethod, 'signal_normal');
    });

    test('duration 毫秒值经 store 持久化往返后保留', () async {
      SharedPreferences.setMockInitialValues({});
      final store = AlarmHistoryStore();
      const ms = 12340; // 12.34 秒
      await store.add(AlarmEvent(
        triggerTime: DateTime(2026, 8, 8),
        duration: const Duration(milliseconds: ms),
        peakConfidence: 0.91,
        avgBpm: 68.5,
        recoveryMethod: 'signal_normal',
      ));

      final events = await store.load();
      expect(events.first.duration.inMilliseconds, ms);
    });

    test('peakConfidence 与 avgBpm 双精度往返保留', () async {
      SharedPreferences.setMockInitialValues({});
      final store = AlarmHistoryStore();
      await store.add(AlarmEvent(
        triggerTime: DateTime(2026, 8, 8),
        duration: const Duration(seconds: 5),
        peakConfidence: 0.934567,
        avgBpm: 81.234,
        recoveryMethod: 'user_confirm',
      ));

      final events = await store.load();
      expect(events.first.peakConfidence, closeTo(0.934567, 1e-6));
      expect(events.first.avgBpm, closeTo(81.234, 1e-6));
    });
  });

  // ════════════ (d) clear() ════════════
  group('AlarmHistoryStore clear()', () {
    test('add 后 clear → load 返回空列表', () async {
      SharedPreferences.setMockInitialValues({});
      final store = AlarmHistoryStore();
      await store.add(makeEvent());
      await store.add(makeEvent(minutesAgo: 1));

      await store.clear();
      final events = await store.load();
      expect(events, isEmpty);
    });

    test('空存储 clear 不抛异常', () async {
      SharedPreferences.setMockInitialValues({});
      final store = AlarmHistoryStore();
      // 不应抛出异常
      await store.clear();
      final events = await store.load();
      expect(events, isEmpty);
    });
  });

  // ════════════ (e) 控件渲染 ════════════
  group('HistorySheet 控件渲染', () {
    testWidgets('空列表渲染 "暂无报警记录"', (tester) async {
      await tester.pumpWidget(const MaterialApp(
        home: Scaffold(
          body: HistorySheet(events: [], onClear: _noop),
        ),
      ));
      expect(find.text('暂无报警记录'), findsOneWidget);
    });

    testWidgets('2 条记录渲染触发时间 MM-dd HH:mm:ss 格式', (tester) async {
      final events = [
        makeEvent(minutesAgo: 0, recovery: 'user_confirm'),
        makeEvent(minutesAgo: 2, recovery: 'signal_normal'),
      ];

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(body: HistorySheet(events: events, onClear: _noop)),
      ));

      // 触发时间格式: MM-dd HH:mm:ss
      // 第一条: 2026-08-08 14:30:00
      expect(find.textContaining('08-08 14:30:00'), findsOneWidget);
      // 第二条: 2026-08-08 14:28:00 (minuteAgo=2)
      expect(find.textContaining('08-08 14:28:00'), findsOneWidget);
    });

    testWidgets('渲染 duration "mm:ss" 格式', (tester) async {
      final events = [makeEvent()]; // duration: 12s 340ms

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(body: HistorySheet(events: events, onClear: _noop)),
      ));

      // 12.34 秒 → "00:12"（指标值独立 Text widget）
      expect(find.text('00:12'), findsOneWidget);
    });

    testWidgets('渲染峰值置信度 "峰值置信度 87%"', (tester) async {
      final events = [makeEvent()]; // peakConfidence: 0.87

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(body: HistorySheet(events: events, onClear: _noop)),
      ));

      // peakConfidence 0.87 → 值 Text "87%"
      expect(find.text('87%'), findsOneWidget);
    });

    testWidgets('渲染平均心率 "平均心率 75 BPM"', (tester) async {
      final events = [makeEvent()]; // avgBpm: 75.0

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(body: HistorySheet(events: events, onClear: _noop)),
      ));

      // avgBpm 75.0 → 值 Text "75 BPM"
      expect(find.text('75 BPM'), findsOneWidget);
    });

    testWidgets('user_confirm 渲染 "手动确认" 标签', (tester) async {
      final events = [makeEvent(recovery: 'user_confirm')];

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(body: HistorySheet(events: events, onClear: _noop)),
      ));

      expect(find.text('手动确认'), findsOneWidget);
    });

    testWidgets('signal_normal 渲染 "自动恢复" 标签', (tester) async {
      final events = [makeEvent(recovery: 'signal_normal')];

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(body: HistorySheet(events: events, onClear: _noop)),
      ));

      expect(find.text('自动恢复'), findsOneWidget);
    });

    testWidgets('渲染 ⚠ 警告图标', (tester) async {
      final events = [makeEvent()];

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(body: HistorySheet(events: events, onClear: _noop)),
      ));

      // ⚠ 使用 Icons.warning_amber 图标
      expect(find.byIcon(Icons.warning_amber), findsOneWidget);
    });

    testWidgets('渲染 "清除记录" 按钮', (tester) async {
      final events = [makeEvent()];

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(body: HistorySheet(events: events, onClear: _noop)),
      ));

      expect(find.text('清除记录'), findsOneWidget);
    });
  });

  // ════════════ (f) showHistorySheet 模态 ════════════
  group('showHistorySheet 模态弹窗', () {
    testWidgets('弹窗渲染 HistorySheet 内容', (tester) async {
      final events = [makeEvent()];

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: Builder(
            builder: (context) => TextButton(
              onPressed: () => showHistorySheet(context, events, _noop),
              child: const Text('open'),
            ),
          ),
        ),
      ));

      // 点击触发弹窗
      await tester.tap(find.text('open'));
      await tester.pumpAndSettle();

      // 弹窗中应包含 HistorySheet 的内容
      expect(find.textContaining('08-08 14:30'), findsOneWidget);
      expect(find.text('暂无报警记录'), findsNothing);
    });

    testWidgets('空列表弹窗渲染空状态', (tester) async {
      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: Builder(
            builder: (context) => TextButton(
              onPressed: () => showHistorySheet(context, [], _noop),
              child: const Text('open'),
            ),
          ),
        ),
      ));

      await tester.tap(find.text('open'));
      await tester.pumpAndSettle();

      expect(find.text('暂无报警记录'), findsOneWidget);
    });
  });

  // ════════════ (g) onClear 回调 ════════════
  group('HistorySheet onClear 回调', () {
    testWidgets('点击 "清除记录" 按钮触发 onClear', (tester) async {
      var cleared = false;
      final events = [makeEvent()];

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: HistorySheet(
            events: events,
            onClear: () => cleared = true,
          ),
        ),
      ));

      await tester.tap(find.text('清除记录'));
      expect(cleared, isTrue);
    });
  });
}

// ════════════ 辅助工具 ════════════

/// VoidCallback 空操作（避免到处写 () {}）
void _noop() {}

/// 手动 JSON 编码 List<Map>（避免 dart:convert 的复杂度）
String _jsonEncode(List<Map<String, dynamic>> list) {
  final buffer = StringBuffer('[');
  for (int i = 0; i < list.length; i++) {
    if (i > 0) buffer.write(',');
    buffer.write(_jsonEncodeMap(list[i]));
  }
  buffer.write(']');
  return buffer.toString();
}

String _jsonEncodeMap(Map<String, dynamic> map) {
  final buffer = StringBuffer('{');
  var first = true;
  map.forEach((key, value) {
    if (!first) buffer.write(',');
    first = false;
    buffer.write('"$key":');
    if (value is String) {
      buffer.write('"$value"');
    } else if (value is int || value is double) {
      buffer.write('$value');
    } else if (value is bool) {
      buffer.write(value ? 'true' : 'false');
    }
  });
  buffer.write('}');
  return buffer.toString();
}
