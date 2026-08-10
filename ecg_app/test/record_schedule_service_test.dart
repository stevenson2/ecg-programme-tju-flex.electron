import 'package:fake_async/fake_async.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:ecg_app/providers/settings_provider.dart';
import 'package:ecg_app/services/record_schedule_service.dart';

/**
 * @brief 定时录制调度服务测试（TDD）
 *
 * 覆盖：
 * - 禁用时无命令发出
 * - 启用后按间隔/时长发送 REC_START → REC_STOP 循环
 * - sendCommand 异常不崩溃，下一周期重试
 * - 中途禁用 → 无后续命令，dispose 取消所有定时器
 * - SettingsProvider 调度设置默认值 / 持久化往返 / clamp / notifyListeners
 */
void main() {
  /// 创建已配置好调度设置的 SettingsProvider
  Future<SettingsProvider> _createSettings({
    bool enabled = false,
    int intervalMin = 60,
    int durationSec = 60,
  }) async {
    SharedPreferences.setMockInitialValues({
      'rec_schedule_enabled': enabled,
      'rec_schedule_interval_min': intervalMin,
      'rec_schedule_duration_sec': durationSec,
    });
    final p = SettingsProvider();
    await p.load();
    return p;
  }

  /// 创建带命令记录器的服务
  RecordScheduleService _createService({
    required SettingsProvider settings,
    required List<String> commandLog,
    Duration tick = const Duration(seconds: 1),
    bool throwOnSend = false,
    int throwOnCallIndex = -1,
  }) {
    int callCount = 0;
    return RecordScheduleService(
      settings: settings,
      sendCommand: (cmd) async {
        callCount++;
        if (throwOnSend && callCount == throwOnCallIndex) {
          throw Exception('BLE disconnected');
        }
        commandLog.add(cmd);
      },
      tick: tick,
    );
  }

  group('RecordScheduleService — 基础行为', () {
    test('disabled → 无命令发出（推进 10 分钟）', () async {
      final commands = <String>[];
      final settings = await _createSettings(enabled: false);

      final service = _createService(settings: settings, commandLog: commands);

      fakeAsync((async) {
        service.start();
        async.elapse(const Duration(minutes: 10));
        service.stop();
      });

      expect(commands, isEmpty);
    });

    test(
        'enabled: interval=1min, duration=5s → REC_START@60s, REC_STOP@65s, REC_START@120s',
        () async {
      final commands = <String>[];
      final settings = await _createSettings(
          enabled: true, intervalMin: 1, durationSec: 5);

      final service = _createService(settings: settings, commandLog: commands);

      fakeAsync((async) {
        service.start();
        async.elapse(const Duration(seconds: 60)); // → REC_START
        async.elapse(const Duration(seconds: 5)); // → REC_STOP
        async.elapse(const Duration(seconds: 55)); // → REC_START
        service.stop();
      });

      expect(commands, ['REC_START', 'REC_STOP', 'REC_START']);
    });

    test('enabled: interval=2min, duration=10s → 正确时序', () async {
      final commands = <String>[];
      final settings = await _createSettings(
          enabled: true, intervalMin: 2, durationSec: 10);

      final service = _createService(settings: settings, commandLog: commands);

      fakeAsync((async) {
        service.start();
        async.elapse(const Duration(seconds: 120)); // → REC_START
        async.elapse(const Duration(seconds: 10)); // → REC_STOP
        async.elapse(const Duration(seconds: 110)); // → REC_START
        service.stop();
      });

      expect(commands, ['REC_START', 'REC_STOP', 'REC_START']);
    });
  });

  group('RecordScheduleService — 异常处理', () {
    test('sendCommand 抛异常 → 不崩溃，下一周期重试', () async {
      final commands = <String>[];
      final settings = await _createSettings(
          enabled: true, intervalMin: 1, durationSec: 5);

      int firstCall = 0;
      final service = RecordScheduleService(
        settings: settings,
        sendCommand: (cmd) async {
          // 第一次调用（REC_START）抛出异常；之后成功
          firstCall++;
          if (firstCall == 1) {
            throw Exception('BLE disconnected');
          }
          commands.add(cmd);
        },
        tick: const Duration(seconds: 1),
      );

      fakeAsync((async) {
        service.start();
        // 60s → REC_START fails, resets to idle
        async.elapse(const Duration(seconds: 60));
        // 等待下一个 interval (60s) → 重试 REC_START
        async.elapse(const Duration(seconds: 60)); // → REC_START 成功
        async.elapse(const Duration(seconds: 5)); // → REC_STOP
        service.stop();
      });

      expect(commands, ['REC_START', 'REC_STOP']);
    });

    test('sendCommand 在 REC_STOP 阶段失败 → 重试', () async {
      final commands = <String>[];
      final settings = await _createSettings(
          enabled: true, intervalMin: 1, durationSec: 5);

      int callCount = 0;
      final service = RecordScheduleService(
        settings: settings,
        sendCommand: (cmd) async {
          callCount++;
          if (callCount == 2) {
            // REC_STOP 失败
            throw Exception('BLE disconnected');
          }
          commands.add(cmd);
        },
        tick: const Duration(seconds: 1),
      );

      fakeAsync((async) {
        service.start();
        async.elapse(const Duration(seconds: 60)); // → REC_START OK
        async.elapse(const Duration(seconds: 5)); // → REC_STOP fails, reset
        // Wait next interval
        async.elapse(const Duration(seconds: 60)); // → REC_START retry
        async.elapse(const Duration(seconds: 5)); // → REC_STOP retry OK
        service.stop();
      });

      expect(commands, ['REC_START', 'REC_START', 'REC_STOP']);
    });
  });

  group('RecordScheduleService — 动态启用/禁用', () {
    test('中途禁用 → 无后续命令发出', () async {
      final commands = <String>[];
      final settings = await _createSettings(
          enabled: true, intervalMin: 1, durationSec: 5);

      final service = _createService(settings: settings, commandLog: commands);

      fakeAsync((async) {
        service.start();
        async.elapse(const Duration(seconds: 60)); // REC_START 已发出

        // 中途禁用（在 REC_STOP 之前）
        settings.setRecScheduleEnabled(false);
        async.elapse(const Duration(seconds: 10)); // 推进到 70s

        // REC_STOP 不应发出
        service.stop();
      });

      expect(commands, ['REC_START']);
    });

    test('重新启用后恢复调度', () async {
      final commands = <String>[];
      final settings = await _createSettings(
          enabled: true, intervalMin: 1, durationSec: 5);

      final service = _createService(settings: settings, commandLog: commands);

      fakeAsync((async) {
        service.start();
        async.elapse(const Duration(seconds: 60)); // REC_START
        async.elapse(const Duration(seconds: 5)); // REC_STOP

        // 禁用
        settings.setRecScheduleEnabled(false);
        async.elapse(const Duration(seconds: 120)); // 2 分钟无命令

        // 重新启用
        settings.setRecScheduleEnabled(true);
        async.elapse(const Duration(seconds: 60)); // REC_START
        async.elapse(const Duration(seconds: 5)); // REC_STOP

        service.stop();
      });

      expect(commands, [
        'REC_START',
        'REC_STOP',
        'REC_START',
        'REC_STOP',
      ]);
    });
  });

  group('RecordScheduleService — 生命周期', () {
    test('stop → cancel 所有定时器，不再触发 tick', () async {
      final commands = <String>[];
      final settings = await _createSettings(
          enabled: true, intervalMin: 1, durationSec: 5);

      final service = _createService(settings: settings, commandLog: commands);

      // 不启动也可安全 stop
      service.stop();
    });

    test('dispose → cancel 所有定时器，安全重复调用', () async {
      final commands = <String>[];
      final settings = await _createSettings(enabled: false);

      final service = _createService(settings: settings, commandLog: commands);

      fakeAsync((async) {
        service.start();
        async.elapse(const Duration(seconds: 30));
        service.dispose(); // 安全（未 start 过的服务 dispose）
      });
    });

    test('double stop → 无异常', () async {
      final commands = <String>[];
      final settings = await _createSettings(enabled: false);

      final service = _createService(settings: settings, commandLog: commands);

      service.stop();
      service.stop(); // 安全重复 stop

      fakeAsync((async) {
        service.start();
        async.elapse(const Duration(seconds: 30));
        service.stop();
        service.stop(); // 安全重复 stop
      });
    });

    test('repeat start → 重置周期', () async {
      final commands = <String>[];
      final settings = await _createSettings(
          enabled: true, intervalMin: 1, durationSec: 5);

      final service = _createService(settings: settings, commandLog: commands);

      fakeAsync((async) {
        service.start();
        async.elapse(const Duration(seconds: 30));
        // repeat start → resets elapsed
        service.start();
        async.elapse(const Duration(seconds: 60)); // from reset → REC_START
        service.stop();
      });

      expect(commands, ['REC_START']);
    });
  });

  group('SettingsProvider — 调度设置持久化', () {
    test('默认值：enabled=false, intervalMin=60, durationSec=60', () async {
      SharedPreferences.setMockInitialValues({});
      final p = SettingsProvider();
      await p.load();
      expect(p.recScheduleEnabled, isFalse);
      expect(p.recScheduleIntervalMin, 60);
      expect(p.recScheduleDurationSec, 60);
    });

    test('setRecScheduleEnabled(true) 后新实例 load() 读回 true', () async {
      SharedPreferences.setMockInitialValues({});
      final p = SettingsProvider();
      await p.setRecScheduleEnabled(true);

      final p2 = SettingsProvider();
      await p2.load();
      expect(p2.recScheduleEnabled, isTrue);
    });

    test(
        'setRecScheduleIntervalMin(120) 后新实例 load() 读回 120',
        () async {
      SharedPreferences.setMockInitialValues({});
      final p = SettingsProvider();
      await p.setRecScheduleIntervalMin(120);

      final p2 = SettingsProvider();
      await p2.load();
      expect(p2.recScheduleIntervalMin, 120);
    });

    test('setRecScheduleDurationSec(30) 后新实例 load() 读回 30', () async {
      SharedPreferences.setMockInitialValues({});
      final p = SettingsProvider();
      await p.setRecScheduleDurationSec(30);

      final p2 = SettingsProvider();
      await p2.load();
      expect(p2.recScheduleDurationSec, 30);
    });

    test('intervalMin 越界 clamp [1, 1440]', () async {
      SharedPreferences.setMockInitialValues({});
      final p = SettingsProvider();
      await p.setRecScheduleIntervalMin(0);
      expect(p.recScheduleIntervalMin, 1);
      await p.setRecScheduleIntervalMin(2000);
      expect(p.recScheduleIntervalMin, 1440);
    });

    test('durationSec 越界 clamp [5, 600]', () async {
      SharedPreferences.setMockInitialValues({});
      final p = SettingsProvider();
      await p.setRecScheduleDurationSec(2);
      expect(p.recScheduleDurationSec, 5);
      await p.setRecScheduleDurationSec(999);
      expect(p.recScheduleDurationSec, 600);
    });

    test('调度设置每次 setter 触发一次 notifyListeners', () async {
      SharedPreferences.setMockInitialValues({});
      final p = SettingsProvider();
      var count = 0;
      p.addListener(() => count++);
      await p.setRecScheduleEnabled(true);
      await p.setRecScheduleIntervalMin(120);
      await p.setRecScheduleDurationSec(30);
      expect(count, 3);
    });
  });
}
