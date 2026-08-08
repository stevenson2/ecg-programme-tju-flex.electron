import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:ecg_app/main.dart';
import 'package:ecg_app/providers/ecg_provider.dart';
import 'package:ecg_app/providers/settings_provider.dart';

void main() {
  SharedPreferences.setMockInitialValues({});

  testWidgets('App 冒烟测试：主界面正常渲染（无 BLE 连接）', (WidgetTester tester) async {
    // 与 main() 相同的 MultiProvider 注入方式；测试环境不触发 BLE 扫描/连接
    await tester.pumpWidget(
      MultiProvider(
        providers: [
          ChangeNotifierProvider(create: (_) => ECGProvider()),
          ChangeNotifierProvider(create: (_) => SettingsProvider()),
        ],
        child: const ECGApp(),
      ),
    );
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 100));

    // 标题栏
    expect(find.text('ESP32-ECG 心电监测'), findsOneWidget);
    // 连接按钮（未连接状态）
    expect(find.text('连接'), findsOneWidget);
    // 速度 / 幅度控制
    expect(find.text('2s'), findsOneWidget);
    expect(find.text('1.0x'), findsOneWidget);
  });
}
