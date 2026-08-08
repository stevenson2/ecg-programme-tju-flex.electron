import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:intl/intl.dart';

import 'package:ecg_app/models/ecg_data.dart';
import 'package:ecg_app/models/alarm_event.dart';
import 'package:ecg_app/providers/ecg_provider.dart';
import 'package:ecg_app/providers/settings_provider.dart';
import 'package:ecg_app/widgets/alarm_dialog.dart';

/**
 * @file alarm_dialog_test.dart
 * @brief 告警弹窗测试（AlarmDialog / showAlarmDialog）
 *
 * 覆盖：
 * - (a) 字段渲染：红色标题 / 置信度百分比 + 进度条 / 心率 BPM / 触发时间
 * - (b) 自动关闭：settings.autoCloseSeconds 到点后弹窗自动消失
 * - (c) 确认按钮：调用 provider.confirmAlarm() 且关闭弹窗
 * - (d) barrierDismissible=false：点击弹窗外部不关闭
 *
 * 测试方式：直接构造 ECGProvider()（不连接 BLE），用 debugAddSample 注入
 * 异常样本触发告警状态机（alarming），再模拟主界面点击按钮调用
 * showAlarmDialog。注入后不再喂样本，provider 的 3 秒恢复定时器不会启动，
 * 不会与弹窗自身的自动关闭定时器混淆。
 */

void main() {
  /** 便捷构造异常样本（触发告警用） */
  ECGSample abnormalSample({int bpm = 75, double confidence = 0.87}) {
    return ECGSample(0, 0, 0, bpm: bpm, abnormal: 1, confidence: confidence);
  }

  /** 搭建宿主页面：按钮点击后调用 showAlarmDialog（模拟主界面告警监听） */
  Future<void> pumpHost(
    WidgetTester tester,
    ECGProvider provider,
    SettingsProvider settings,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: Builder(
            builder: (context) => Center(
              child: TextButton(
                onPressed: () => showAlarmDialog(context, provider, settings),
                child: const Text('open'),
              ),
            ),
          ),
        ),
      ),
    );
  }

  /** 触发告警并打开弹窗（返回已注入异常样本的 provider） */
  Future<ECGProvider> openAlarmDialog(
    WidgetTester tester, {
    int bpm = 75,
    double confidence = 0.87,
  }) async {
    final provider = ECGProvider();
    provider.debugAddSample(abnormalSample(bpm: bpm, confidence: confidence));
    // 前置条件：异常样本已把状态机推到 alarming
    expect(provider.alarmState, AlarmState.alarming);

    final settings = SettingsProvider();
    await pumpHost(tester, provider, settings);
    await tester.tap(find.text('open'));
    await tester.pumpAndSettle();
    return provider;
  }

  group('AlarmDialog 渲染', () {
    testWidgets('(a) 渲染全部字段：标题 / 置信度 / 进度条 / 心率 / 触发时间',
        (tester) async {
      final provider = await openAlarmDialog(tester);

      // 红色主题标题
      expect(find.text('⚠ 异常心律'), findsOneWidget);

      // 置信度百分比文本（0.87 → 87%）
      expect(find.text('置信度 87%'), findsOneWidget);

      // 进度条值与峰值置信度一致，且使用红色主题色
      final progress = tester.widget<LinearProgressIndicator>(
          find.byType(LinearProgressIndicator));
      expect(progress.value, 0.87);
      expect(progress.color, const Color(0xFFE53935));

      // 心率（provider.lastBpm）
      expect(find.text('心率 75 BPM'), findsOneWidget);

      // 触发时间（intl DateFormat HH:mm:ss）
      final expected = DateFormat('HH:mm:ss').format(provider.alarmTriggerTime!);
      expect(find.text('触发时间 $expected'), findsOneWidget);
    });
  });

  group('AlarmDialog 自动关闭', () {
    testWidgets('(b) autoCloseSeconds 到点后弹窗自动消失', (tester) async {
      final provider = ECGProvider();
      provider.debugAddSample(abnormalSample());
      final settings = SettingsProvider();
      // 默认自动关闭时长为 10 秒
      expect(settings.autoCloseSeconds, SettingsProvider.kDefaultAutoClose);

      await pumpHost(tester, provider, settings);
      await tester.tap(find.text('open'));
      await tester.pumpAndSettle();
      expect(find.text('⚠ 异常心律'), findsOneWidget);

      // 未到点（5s < 10s）：弹窗仍在
      await tester.pump(const Duration(seconds: 5));
      expect(find.text('⚠ 异常心律'), findsOneWidget);

      // 到点（再 5s = 10s）：弹窗自身的定时器触发自动关闭
      await tester.pump(const Duration(seconds: 5));
      await tester.pumpAndSettle();
      expect(find.text('⚠ 异常心律'), findsNothing);
    });
  });

  group('AlarmDialog 确认按钮', () {
    testWidgets('(c) 点击确认：调用 confirmAlarm 且关闭弹窗', (tester) async {
      final provider = await openAlarmDialog(tester);

      await tester.tap(find.text('确认'));
      await tester.pumpAndSettle();

      // provider 状态回到 idle，恢复方式为 user_confirm
      expect(provider.alarmState, AlarmState.idle);
      expect(provider.lastCompletedAlarm, isNotNull);
      expect(provider.lastCompletedAlarm!.recoveryMethod, 'user_confirm');
      expect(provider.alarmPeakConfidence, 0.0);
      expect(provider.alarmTriggerTime, isNull);

      // 弹窗已关闭
      expect(find.text('⚠ 异常心律'), findsNothing);
    });
  });

  group('AlarmDialog 遮罩行为', () {
    testWidgets('(d) barrierDismissible=false：点击弹窗外部不关闭', (tester) async {
      await openAlarmDialog(tester);

      // 点击弹窗外部（左上角遮罩层）
      await tester.tapAt(const Offset(10, 10));
      await tester.pump();

      // 弹窗仍存在
      expect(find.text('⚠ 异常心律'), findsOneWidget);
    });
  });
}
