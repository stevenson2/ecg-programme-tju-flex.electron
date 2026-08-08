import 'dart:async';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/services.dart';

import '../providers/settings_provider.dart';

/**
 * @file alarm_sound_service.dart
 * @brief 告警提示音服务：循环播放 beep.wav，支持音量/开关/触觉反馈/回退提示音
 *
 * 设计要点：
 * - 与 UI 解耦：不持有 BuildContext，仅依赖 SettingsProvider 与 AudioPlayer。
 * - 循环调度：play() → onPlayerComplete 事件 → Timer(loopInterval) → play()，
 *   每次播放完成后再等一个间隔重播，避免与真实播放时长耦合。
 * - 可测性：AudioPlayer（player）与 loopInterval 均可注入；
 *   isLooping 暴露循环状态供 UI/测试判定。
 * - 设置联动：音量取自 settings.soundVolume（默认 0.8）；
 *   settings.soundEnabled=false 时静默跳过播放（调用仍正常返回）。
 * - 回退路径：资源播放失败（play 抛异常 / 完成事件流错误）时播放系统提示音
 *   SystemSound.alert，保证告警仍有可闻提示。
 * - 触觉反馈：每次实际播报伴随 HapticFeedback.vibrate()。
 */
class AlarmSoundService {
  /// 默认提示音资源路径（相对 assets/，已注册于 pubspec assets: - assets/audio/）
  static const String kDefaultAssetPath = 'audio/beep.wav';

  /**
   * @param settings     告警设置源（soundEnabled / soundVolume）
   * @param player       播放器实例，测试可注入 Fake；缺省创建真实 AudioPlayer
   * @param loopInterval 每次播放完成后的重播间隔（默认 2 秒）
   * @param assetPath    提示音资源路径
   */
  AlarmSoundService({
    required SettingsProvider settings,
    AudioPlayer? player,
    this.loopInterval = const Duration(seconds: 2),
    this.assetPath = kDefaultAssetPath,
  })  : _settings = settings,
        _player = player ?? AudioPlayer() {
    // 播放完成 → 继续循环；事件流错误 → 回退系统提示音
    _completeSub = _player.onPlayerComplete.listen(
      _onPlayerComplete,
      onError: (Object _, [StackTrace? __]) => _playFallback(),
    );
  }

  final SettingsProvider _settings;

  final AudioPlayer _player;

  /// 重播间隔（测试可注入短间隔或借助 fakeAsync 控制虚拟时钟）
  final Duration loopInterval;

  /// 提示音资源路径
  final String assetPath;

  Timer? _loopTimer; // 重播定时器
  bool _looping = false; // 循环开关（stop/dispose 后为 false）
  bool _disposed = false; // 防止重复 dispose
  late final StreamSubscription<void> _completeSub; // 播放完成订阅

  /// 当前是否处于告警循环状态
  bool get isLooping => _looping;

  /// 开始告警循环：设置音量并播放；soundEnabled=false 时静默跳过（正常返回）
  Future<void> startAlarmLoop() async {
    if (_looping || _disposed) return;
    _looping = true;
    await _playBeep();
  }

  /// 停止告警循环：取消定时器并停止播放器
  Future<void> stopAlarmLoop() async {
    _looping = false;
    _loopTimer?.cancel();
    _loopTimer = null;
    await _player.stop();
  }

  /// 释放资源：停止循环、取消订阅并释放播放器（可重复调用）
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    unawaited(stopAlarmLoop());
    _completeSub.cancel();
    unawaited(_player.dispose());
  }

  /// 单次播报：应用音量 → 播放 → 触觉反馈；播放失败时回退系统提示音
  Future<void> _playBeep() async {
    if (!_looping) return;
    if (!_settings.soundEnabled) return; // 关闭提示音：不发声也不振动
    try {
      await _player.setVolume(_settings.soundVolume);
      await _player.play(AssetSource(assetPath));
    } on Exception {
      _playFallback(); // 资源播放失败 → 系统提示音
      return;
    }
    HapticFeedback.vibrate(); // 每次实际播报伴随触觉反馈（测试环境为 no-op）
  }

  /// 播放完成事件：仍在循环中则安排下一次播报
  void _onPlayerComplete(void _) {
    if (!_looping) return;
    _loopTimer?.cancel();
    _loopTimer = Timer(loopInterval, _playBeep);
  }

  /// 回退：播放系统提示音（audioplayers 失败时保证仍有可闻告警）
  void _playFallback() {
    unawaited(SystemSound.play(SystemSoundType.alert));
  }
}
