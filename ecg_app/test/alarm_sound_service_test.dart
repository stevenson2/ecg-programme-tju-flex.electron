import 'dart:async';

import 'package:audioplayers/audioplayers.dart';
import 'package:ecg_app/providers/settings_provider.dart';
import 'package:ecg_app/services/alarm_sound_service.dart';
import 'package:fake_async/fake_async.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

/**
 * @brief AlarmSoundService 的 TDD 测试
 *
 * 覆盖（规范用例 a~e + 回退路径 f）：
 * - (a) startAlarmLoop：从设置读取音量并播放 beep（setVolume + play）
 * - (b) onPlayerComplete 事件后经 loopInterval 再次播放（循环不中断）
 * - (c) stopAlarmLoop：停止播放器且完成事件不再触发重播
 * - (d) soundEnabled=false 时 startAlarmLoop 不播放（正常返回）
 * - (e) dispose：停止播放、取消订阅、之后完成事件不再重播
 * - (f) 播放失败时回退系统提示音 SystemSound.alert
 *
 * 说明：
 * - 循环调度使用 Timer(loopInterval)，测试通过 fakeAsync 控制虚拟时钟，
 *   无需等待真实 2 秒。
 * - HapticFeedback.vibrate() 在测试绑定下为 no-op，仅验证不抛错。
 */

/// 忽略 complete/completeError 的“静默”Completer。
/// 基类构造期 _create() 会调用平台通道（测试环境抛 MissingPluginException），
/// 覆写 creatingCompleter 使 completeError 成为 no-op，避免未处理异步错误。
class _NoopCompleter implements Completer<void> {
  @override
  Future<void> get future => Future<void>.value();

  @override
  bool get isCompleted => true;

  @override
  void complete([FutureOr<void>? value]) {}

  @override
  void completeError(Object error, [StackTrace? stackTrace]) {}
}

/// 测试用假播放器：继承 AudioPlayer 并仅覆写服务用到的成员，
/// 记录调用次数/参数，并可手动触发播放完成事件。
class FakeAudioPlayer extends AudioPlayer {
  FakeAudioPlayer() : super(playerId: 'fake-player');

  static final Completer<void> _silentCompleter = _NoopCompleter();

  final StreamController<void> _completeController =
      StreamController<void>.broadcast();

  // ── 调用记录（供断言）──
  int playCount = 0;
  int stopCount = 0;
  int setVolumeCount = 0;
  double? lastVolume;
  bool disposed = false;

  /// 置 true 时 play() 抛异常，用于回退路径测试
  bool throwOnPlay = false;

  // 构造期平台通道不可用：用静默 Completer 覆盖基类字段
  @override
  Completer<void> get creatingCompleter => _silentCompleter;

  @override
  Stream<void> get onPlayerComplete => _completeController.stream;

  @override
  Future<void> play(
    Source source, {
    double? volume,
    double? balance,
    AudioContext? ctx,
    Duration? position,
    PlayerMode? mode,
  }) async {
    if (throwOnPlay) {
      throw Exception('fake play failure');
    }
    playCount++;
    if (volume != null) {
      lastVolume = volume;
      setVolumeCount++;
    }
  }

  @override
  Future<void> setVolume(double volume) async {
    setVolumeCount++;
    lastVolume = volume;
  }

  @override
  Future<void> stop() async {
    stopCount++;
  }

  @override
  Future<void> dispose() async {
    disposed = true;
  }

  // 基类 onPlayerComplete 处理器会调用 getCurrentPosition 走平台通道，
  // 覆写为直接返回 null，避免测试环境下的 MissingPluginException。
  @override
  Future<Duration?> getCurrentPosition() async => null;

  /// 模拟一次播放完成事件
  void fireComplete() => _completeController.add(null);
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('AlarmSoundService', () {
    test('(a) startAlarmLoop 从设置读取音量并播放 beep', () async {
      final settings = SettingsProvider(); // 默认 sound=true, volume=0.8
      final fake = FakeAudioPlayer();
      final svc = AlarmSoundService(settings: settings, player: fake);

      await svc.startAlarmLoop();

      expect(svc.isLooping, isTrue);
      expect(fake.playCount, 1);
      expect(fake.setVolumeCount, 1);
      expect(fake.lastVolume, settings.soundVolume); // 0.8
    });

    test('(b) onPlayerComplete 后经 loopInterval 再次播放（循环持续）', () {
      fakeAsync((async) {
        final settings = SettingsProvider();
        final fake = FakeAudioPlayer();
        final svc = AlarmSoundService(settings: settings, player: fake);

        svc.startAlarmLoop();
        async.flushMicrotasks();
        expect(fake.playCount, 1);

        // 完成事件 → 调度 2s 定时器 → 定时器触发后重播
        fake.fireComplete();
        async.flushMicrotasks();
        expect(fake.playCount, 1); // 定时器未到，尚未重播
        async.elapse(const Duration(seconds: 2));
        async.flushMicrotasks();
        expect(fake.playCount, 2);

        // 再来一轮，验证循环可持续
        fake.fireComplete();
        async.flushMicrotasks();
        async.elapse(const Duration(seconds: 2));
        async.flushMicrotasks();
        expect(fake.playCount, 3);
      });
    });

    test('(c) stopAlarmLoop 停止播放且完成事件不再触发重播', () {
      fakeAsync((async) {
        final settings = SettingsProvider();
        final fake = FakeAudioPlayer();
        final svc = AlarmSoundService(settings: settings, player: fake);

        svc.startAlarmLoop();
        async.flushMicrotasks();
        expect(fake.playCount, 1);

        fake.fireComplete();
        async.flushMicrotasks();
        async.elapse(const Duration(seconds: 2));
        async.flushMicrotasks();
        expect(fake.playCount, 2);

        svc.stopAlarmLoop();
        async.flushMicrotasks();
        expect(svc.isLooping, isFalse);
        expect(fake.stopCount, 1);

        // 停止后再来完成事件，不应重播
        fake.fireComplete();
        async.flushMicrotasks();
        async.elapse(const Duration(seconds: 2));
        async.flushMicrotasks();
        expect(fake.playCount, 2);
        expect(fake.stopCount, 1);
      });
    });

    test('(d) soundEnabled=false 时 startAlarmLoop 不播放', () async {
      SharedPreferences.setMockInitialValues({});
      final settings = SettingsProvider();
      await settings.setSound(false);
      final fake = FakeAudioPlayer();
      final svc = AlarmSoundService(settings: settings, player: fake);

      await svc.startAlarmLoop();

      expect(fake.playCount, 0);
      expect(fake.setVolumeCount, 0);
      expect(fake.lastVolume, isNull);
    });

    test('(e) dispose 清理：停止播放、取消订阅、之后不再重播', () {
      fakeAsync((async) {
        final settings = SettingsProvider();
        final fake = FakeAudioPlayer();
        final svc = AlarmSoundService(settings: settings, player: fake);

        svc.startAlarmLoop();
        async.flushMicrotasks();
        expect(fake.playCount, 1);

        fake.fireComplete();
        async.flushMicrotasks();
        async.elapse(const Duration(seconds: 2));
        async.flushMicrotasks();
        expect(fake.playCount, 2);

        svc.dispose();
        async.flushMicrotasks();
        expect(fake.stopCount, greaterThanOrEqualTo(1));
        expect(fake.disposed, isTrue);
        expect(svc.isLooping, isFalse);

        // dispose 后再来完成事件，不应重播
        fake.fireComplete();
        async.flushMicrotasks();
        async.elapse(const Duration(seconds: 2));
        async.flushMicrotasks();
        expect(fake.playCount, 2);
      });
    });

    test('(f) 播放失败时回退系统提示音 SystemSound.alert', () {
      // 拦截平台通道，记录 SystemSound / HapticFeedback 调用
      final calls = <String>[];
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(SystemChannels.platform,
              (MethodCall call) async {
        calls.add(call.method);
        return null;
      });
      addTearDown(() {
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
            .setMockMethodCallHandler(SystemChannels.platform, null);
      });

      fakeAsync((async) {
        final settings = SettingsProvider();
        final fake = FakeAudioPlayer()..throwOnPlay = true;
        final svc = AlarmSoundService(settings: settings, player: fake);

        svc.startAlarmLoop();
        async.flushMicrotasks();

        expect(fake.playCount, 0);
        expect(calls, contains('SystemSound.play'));
      });
    });
  });
}
