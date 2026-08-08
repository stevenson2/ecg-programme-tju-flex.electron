import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:ecg_app/models/ecg_data.dart';
import 'package:ecg_app/models/alarm_event.dart';
import 'package:ecg_app/providers/ecg_provider.dart';
import 'package:ecg_app/providers/settings_provider.dart';
import 'package:ecg_app/services/alarm_sound_service.dart';
import 'package:ecg_app/main.dart';

/**
 * @file alarm_integration_test.dart
 * @brief 告警管线集成测试 — 验证 alarm pipeline 端到端行为
 *
 * 覆盖：
 * - (1) abnormal=1 → 弹窗 + 提示音精确一次（latch 防止重复）
 * - (2) 恢复（信号正常 3s）→ 停止提示音 + 写入历史
 * - (3) 2 分钟正常数据 → 零次告警转换
 * - (4) DND 免打扰：抑制弹窗与提示音，但历史仍记录
 */

/// 可注入的假提示音服务 — 重写播放/停止逻辑，仅计数不发声
class FakeAlarmSoundService extends AlarmSoundService {
  int startCount = 0;
  int stopCount = 0;

  FakeAlarmSoundService({required super.settings}) : super();

  @override
  Future<void> startAlarmLoop() async {
    startCount++;
  }

  @override
  Future<void> stopAlarmLoop() async {
    stopCount++;
  }

  @override
  void dispose() {
    // 不释放真实 AudioPlayer
  }
}

/// 便捷构造异常样本
ECGSample _abnormalSample({int bpm = 75, double confidence = 0.87}) =>
    ECGSample(0, 0, 0, bpm: bpm, abnormal: 1, confidence: confidence);

/// 便捷构造正常样本
ECGSample _normalSample({int bpm = 72}) =>
    ECGSample(0, 0, 0, bpm: bpm, abnormal: 0, confidence: 0.05);

/// 测试上下文：持有 provider / sound / settings 引用
class _TestContext {
  final ECGProvider provider;
  final FakeAlarmSoundService sound;
  final SettingsProvider settings;

  const _TestContext({
    required this.provider,
    required this.sound,
    required this.settings,
  });
}

void main() {
  SharedPreferences.setMockInitialValues({});

  /// 搭建测试 widget 树（MultiProvider + MaterialApp + ECGMonitorScreen）
  Future<_TestContext> pumpTestApp(WidgetTester tester) async {
    final ecgProvider = ECGProvider();
    final settings = SettingsProvider();
    final fakeSound = FakeAlarmSoundService(settings: settings);

    await tester.pumpWidget(
      MultiProvider(
        providers: [
          ChangeNotifierProvider<ECGProvider>.value(value: ecgProvider),
          ChangeNotifierProvider<SettingsProvider>.value(value: settings),
        ],
        child: MaterialApp(
          home: ECGMonitorScreen(soundService: fakeSound),
        ),
      ),
    );
    /// 等待 initState 异步初始化（settings.load / historyStore.load）
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 100));

    return _TestContext(
        provider: ecgProvider, sound: fakeSound, settings: settings);
  }

  group('告警管线集成', () {
    testWidgets('(1) 异常触发 → 弹窗 + 提示音精确各一次（arming不重复触发）',
        (tester) async {
      final ctx = await pumpTestApp(tester);
      final provider = ctx.provider;
      final sound = ctx.sound;

      // 注入异常样本 → 触发告警
      provider.debugAddSample(_abnormalSample());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 100));

      // 弹窗出现
      expect(find.text('⚠ 异常心律'), findsOneWidget);
      // 提示音启动
      expect(sound.startCount, 1);

      // 注入更多异常样本（arming 状态）→ 不新增弹窗、不新增提示音调用
      for (int i = 0; i < 10; i++) {
        provider.debugAddSample(_abnormalSample(confidence: 0.9));
      }
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 100));

      // 弹窗仍仅一个（latched）
      expect(find.text('⚠ 异常心律'), findsOneWidget);
      // startAlarmLoop 仅调用一次
      expect(sound.startCount, 1);
    });

    testWidgets('(2) 信号恢复 → 3s后停止提示音 + 写入历史',
        (tester) async {
      final ctx = await pumpTestApp(tester);
      final provider = ctx.provider;
      final sound = ctx.sound;

      // 触发告警
      provider.debugAddSample(_abnormalSample(bpm: 80, confidence: 0.92));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 100));

      expect(find.text('⚠ 异常心律'), findsOneWidget);
      expect(sound.startCount, 1);

      // 注入正常样本 — 启动 3s 恢复定时器
      provider.debugAddSample(_normalSample());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 100));

      // 推进 4 秒（超过 3 秒恢复窗口）
      await tester.pump(const Duration(seconds: 4));
      await tester.pump(const Duration(milliseconds: 100));

      // 状态回到 idle
      expect(provider.alarmState, AlarmState.idle);
      // 提示音停止
      expect(sound.stopCount, greaterThanOrEqualTo(1));
      // 历史事件已生成
      expect(provider.lastCompletedAlarm, isNotNull);
      expect(provider.lastCompletedAlarm!.recoveryMethod, 'signal_normal');
      expect(provider.lastCompletedAlarm!.peakConfidence, 0.92);
      expect(provider.lastCompletedAlarm!.avgBpm, 76.0); // (80+72)/2
    });

    testWidgets('(3) 2分钟正常信号 → 零次告警转换', (tester) async {
      final ctx = await pumpTestApp(tester);
      final provider = ctx.provider;
      final sound = ctx.sound;

      // 持续注入正常样本
      for (int i = 0; i < 120; i++) {
        provider.debugAddSample(_normalSample());
      }
      await tester.pump();
      await tester.pump(const Duration(minutes: 2));
      await tester.pump(const Duration(milliseconds: 100));

      // 始终处于 idle
      expect(provider.alarmState, AlarmState.idle);
      // 从未调用 startAlarmLoop
      expect(sound.startCount, 0);
      // 无人告警历史事件（正常信号不会触发告警状态机）
      expect(provider.lastCompletedAlarm, isNull);

      // 弹窗从未出现
      expect(find.text('⚠ 异常心律'), findsNothing);
    });

    testWidgets('(4) DND免打扰：抑制弹窗与提示音，但历史仍记录',
        (tester) async {
      final ctx = await pumpTestApp(tester);
      final provider = ctx.provider;
      final sound = ctx.sound;
      final settings = ctx.settings;

      // 开启免打扰
      await settings.setDnd(true);
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 100));

      expect(settings.dndEnabled, true);

      // 触发告警
      provider.debugAddSample(_abnormalSample(bpm: 72, confidence: 0.88));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 100));

      // DND 抑制弹窗
      expect(find.text('⚠ 异常心律'), findsNothing);
      // DND 抑制提示音启动
      expect(sound.startCount, 0);

      // 注入正常样本 → 恢复
      provider.debugAddSample(_normalSample());
      await tester.pump();
      await tester.pump(const Duration(seconds: 4));
      await tester.pump(const Duration(milliseconds: 100));

      // 历史仍记录（DND 仅抑制 UX，不抑制记录）
      expect(provider.lastCompletedAlarm, isNotNull);
      expect(provider.lastCompletedAlarm!.recoveryMethod, 'signal_normal');
    });
  });
}
