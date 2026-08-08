import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:ecg_app/providers/settings_provider.dart';
import 'package:ecg_app/widgets/settings_sheet.dart';

/**
 * @brief SettingsProvider 与 AlarmSettingsSheet 的 TDD 测试
 *
 * 覆盖：
 * - 无存储时的默认值（dnd=false, sound=true, volume=0.8, autoClose=10）
 * - set + SharedPreferences 持久化往返（新实例 load 可读回）
 * - 音量 clamp [0,1]、自动关闭时长 clamp [3,30]
 * - 每次 setter 触发 notifyListeners
 * - 设置弹窗渲染 + 控件操作回写 provider
 */
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('SettingsProvider 默认值', () {
    test('SharedPreferences 为空时 load() 采用默认值', () async {
      SharedPreferences.setMockInitialValues({});
      final p = SettingsProvider();
      await p.load();
      expect(p.dndEnabled, isFalse);
      expect(p.soundEnabled, isTrue);
      expect(p.soundVolume, 0.8);
      expect(p.autoCloseSeconds, 10);
    });
  });

  group('SettingsProvider 持久化往返', () {
    test('setDnd(true) 后新实例 load() 读回 true', () async {
      SharedPreferences.setMockInitialValues({});
      final p = SettingsProvider();
      await p.setDnd(true);

      final p2 = SettingsProvider();
      await p2.load();
      expect(p2.dndEnabled, isTrue);
    });

    test('setSound(false) 后新实例 load() 读回 false', () async {
      SharedPreferences.setMockInitialValues({});
      final p = SettingsProvider();
      await p.setSound(false);

      final p2 = SettingsProvider();
      await p2.load();
      expect(p2.soundEnabled, isFalse);
    });

    test('setVolume(0.5) 后新实例 load() 读回 0.5', () async {
      SharedPreferences.setMockInitialValues({});
      final p = SettingsProvider();
      await p.setVolume(0.5);

      final p2 = SettingsProvider();
      await p2.load();
      expect(p2.soundVolume, 0.5);
    });

    test('setAutoClose(15) 后新实例 load() 读回 15', () async {
      SharedPreferences.setMockInitialValues({});
      final p = SettingsProvider();
      await p.setAutoClose(15);

      final p2 = SettingsProvider();
      await p2.load();
      expect(p2.autoCloseSeconds, 15);
    });
  });

  group('SettingsProvider 数值收敛', () {
    test('setVolume 超出 [0,1] 时收敛到边界', () async {
      SharedPreferences.setMockInitialValues({});
      final p = SettingsProvider();
      await p.setVolume(1.5);
      expect(p.soundVolume, 1.0);
      await p.setVolume(-0.2);
      expect(p.soundVolume, 0.0);
    });

    test('setAutoClose 超出 [3,30] 时收敛到边界', () async {
      SharedPreferences.setMockInitialValues({});
      final p = SettingsProvider();
      await p.setAutoClose(99);
      expect(p.autoCloseSeconds, 30);
      await p.setAutoClose(1);
      expect(p.autoCloseSeconds, 3);
    });
  });

  group('SettingsProvider notifyListeners', () {
    test('每次 setter 都触发一次 notifyListeners', () async {
      SharedPreferences.setMockInitialValues({});
      final p = SettingsProvider();
      var count = 0;
      p.addListener(() => count++);
      await p.setDnd(true);
      await p.setSound(false);
      await p.setVolume(0.5);
      await p.setAutoClose(15);
      expect(count, 4);
    });
  });

  group('AlarmSettingsSheet 控件', () {
    Future<SettingsProvider> pumpSheet(WidgetTester tester,
        {bool viaHelper = false}) async {
      SharedPreferences.setMockInitialValues({});
      final p = SettingsProvider();
      await p.load();
      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: viaHelper
              ? Builder(
                  builder: (context) => TextButton(
                    onPressed: () => showAlarmSettingsSheet(context, p),
                    child: const Text('open'),
                  ),
                )
              : AlarmSettingsSheet(settings: p),
        ),
      ));
      if (viaHelper) {
        await tester.tap(find.text('open'));
        await tester.pumpAndSettle();
      }
      return p;
    }

    testWidgets('弹窗渲染免打扰/提示音开关与两条滑块及标签', (tester) async {
      await pumpSheet(tester);
      expect(find.text('免打扰'), findsOneWidget);
      expect(find.text('提示音'), findsOneWidget);
      expect(find.byKey(const ValueKey('volume_slider')), findsOneWidget);
      expect(find.byKey(const ValueKey('auto_close_slider')), findsOneWidget);
      expect(find.text('80%'), findsOneWidget);   // 默认音量百分比
      expect(find.text('10 秒'), findsOneWidget); // 默认自动关闭时长
    });

    testWidgets('切换开关调用对应 setter 并更新 provider', (tester) async {
      final p = await pumpSheet(tester);
      expect(p.dndEnabled, isFalse);
      expect(p.soundEnabled, isTrue);

      // 第一个 SwitchListTile = 免打扰
      await tester.tap(find.byType(SwitchListTile).first);
      await tester.pump();
      expect(p.dndEnabled, isTrue);

      // 第二个 SwitchListTile = 提示音
      await tester.tap(find.byType(SwitchListTile).at(1));
      await tester.pump();
      expect(p.soundEnabled, isFalse);

      // 再点一次免打扰恢复
      await tester.tap(find.byType(SwitchListTile).first);
      await tester.pump();
      expect(p.dndEnabled, isFalse);
    });

    testWidgets('拖动音量滑块回写 provider 并更新百分比标签', (tester) async {
      final p = await pumpSheet(tester);
      // 向右拖满 → 1.0 → 100%
      await tester.drag(
          find.byKey(const ValueKey('volume_slider')), const Offset(500, 0));
      await tester.pump();
      expect(p.soundVolume, 1.0);
      expect(find.text('100%'), findsOneWidget);
    });

    testWidgets('拖动自动关闭滑块回写 provider 并更新时长标签', (tester) async {
      final p = await pumpSheet(tester);
      // 向右拖满 → 30 秒
      await tester.drag(find.byKey(const ValueKey('auto_close_slider')),
          const Offset(500, 0));
      await tester.pump();
      expect(p.autoCloseSeconds, 30);
      expect(find.text('30 秒'), findsOneWidget);
    });

    testWidgets('showAlarmSettingsSheet 弹出模态底部弹窗', (tester) async {
      await pumpSheet(tester, viaHelper: true);
      expect(find.byType(AlarmSettingsSheet), findsOneWidget);
      expect(find.text('告警设置'), findsOneWidget);
    });
  });
}
