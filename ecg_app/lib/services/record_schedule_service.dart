import 'dart:async';

import '../providers/settings_provider.dart';

/**
 * @file record_schedule_service.dart
 * @brief 定时录制调度服务（追加需求 1：App 端定时录制调度）
 *
 * 基于递归单次 Timer 驱动的状态机（兼容 fake_async）：
 *   enabled & idle → wait recScheduleIntervalMin 分钟 → REC_START
 *   → wait recScheduleDurationSec 秒 → REC_STOP
 *   → idle → 下一周期
 *
 * 异常处理：sendCommand 失败（BLE 断连）→ 记录日志，相位重置为 idle，
 * 下一周期重试。禁用时 tick 不发送任何命令。设置变更时重置周期。
 */
class RecordScheduleService {
  final SettingsProvider _settings;
  final Future<void> Function(String cmd) _sendCommand;
  final Duration _tick;

  Timer? _tickTimer;
  int _secondsElapsed = 0;
  bool _isRecording = false; // false = 等待开始，true = 录制中
  bool _lastEnabled = false;
  int _lastIntervalMin = 0;
  int _lastDurationSec = 0;

  /**
   * @param settings 设置提供者（含 recScheduleEnabled / recScheduleIntervalMin /
   *                 recScheduleDurationSec）
   * @param sendCommand 异步命令发送回调（通常为 BLEService.sendCommand）
   * @param tick tick 间隔，默认 1 秒（测试可注入更短值）
   */
  RecordScheduleService({
    required SettingsProvider settings,
    required Future<void> Function(String cmd) sendCommand,
    Duration tick = const Duration(seconds: 1),
  })  : _settings = settings,
        _sendCommand = sendCommand,
        _tick = tick;

  /// 启动调度循环（监听设置变更，创建 tick Timer）
  void start() {
      _lastEnabled = _settings.recScheduleEnabled;
      _lastIntervalMin = _settings.recScheduleIntervalMin;
      _lastDurationSec = _settings.recScheduleDurationSec;
    stop(); // 先取消已有 Timer
    _secondsElapsed = 0;
    _isRecording = false;
    _settings.addListener(_onSettingsChanged);
    _scheduleNextTick();
  }

  /// 停止调度循环（取消 Timer，移除监听），可安全重复调用
  void stop() {
    _tickTimer?.cancel();
    _tickTimer = null;
    _settings.removeListener(_onSettingsChanged);
  }

  /// 释放资源，同 stop()
  void dispose() {
    stop();
  }

  /// 设置变更回调：重置周期，重新使用当前值
  void _onSettingsChanged() {
      final enabled = _settings.recScheduleEnabled;
      final intervalMin = _settings.recScheduleIntervalMin;
      final durationSec = _settings.recScheduleDurationSec;
      if (enabled == _lastEnabled &&
          intervalMin == _lastIntervalMin &&
          durationSec == _lastDurationSec) {
        return; // 与调度无关的设置 (免打扰/音量等) 不重置录制周期
      }
      _lastEnabled = enabled;
      _lastIntervalMin = intervalMin;
      _lastDurationSec = durationSec;
    _secondsElapsed = 0;
    _isRecording = false;
  }

  /// 调度下一次 tick
  void _scheduleNextTick() {
    _tickTimer = Timer(_tick, _onTick);
  }

  /// 每秒 tick：驱动状态机
  void _onTick() {
    // 已停止 → 不再调度下一 tick
    if (_tickTimer == null) return;

    // 禁用时自旋等待（不发送命令，但继续 tick 以便启用时恢复）
    if (_settings.recScheduleEnabled) {
      _secondsElapsed++;

      if (!_isRecording) {
        // 等待到达间隔 → 发送 REC_START
        final intervalSec = _settings.recScheduleIntervalMin * 60;
        if (_secondsElapsed >= intervalSec) {
          _trySend('REC_START');
          _secondsElapsed = 0;
          _isRecording = true;
        }
      } else {
        // 录制中等待到达时长 → 发送 REC_STOP
        if (_secondsElapsed >= _settings.recScheduleDurationSec) {
          _trySend('REC_STOP');
          // 不重置 _secondsElapsed：继续从上次 REC_START 计时，
          // 以保持 REC_START 之间的间隔恒定。
          _isRecording = false;
        }
      }
    }

    // 调度下一次 tick（若未被 stop）
    if (_tickTimer != null) {
      _scheduleNextTick();
    }
  }

  /// 尝试发送命令，异步异常时重置相位为 idle（下一周期重试）
  void _trySend(String cmd) {
    try {
      // 同步部分（多数 sendCommand 实现在 await 之前无同步异常，
      // 但保留 try/catch 防范同步崩溃）
      _sendCommand(cmd).catchError((_) {
        _secondsElapsed = 0;
        _isRecording = false;
      });
    } catch (_) {
      // 万一 sendCommand 本身不是 async 且同步抛异常，也兜底
      _secondsElapsed = 0;
      _isRecording = false;
    }
  }
}
