import 'dart:async';
import 'package:flutter/foundation.dart';
import '../models/ecg_data.dart';
import '../models/alarm_event.dart';
import '../models/waveform_data_source.dart';
import '../services/ble_service.dart';

/**
 * @file ecg_provider.dart
 * @brief 心电数据状态管理（ChangeNotifier + Provider）
 *
 * 管理：
 * - 历史数据环形缓冲区（最新 1500 点 = 6 秒 @250Hz）
 * - 连接状态
 * - 心率计算
 * - 显示速度/幅度控制
 */

class ECGProvider extends ChangeNotifier implements WaveformDataSource {
  // ── BLE 服务 ──
  final BLEService _bleService = BLEService();

  // ── 环形缓冲区 ──
  static const int kBufferSize = 1500; // 6 秒 @ 250Hz
  final List<ECGSample> _samples = [];
  int _droppedCount = 0;

  // ── 心率 (来自 ESP32 板上算法) ──
  double _heartRate = 0;

  // ── AI 异常检测 (来自 ESP32 板上 TFLite Micro 推理) ──
  static const int kAbnormalWindow = 10; // 防闪烁窗口：最近 10 样本内任一异常即告警
  double _abnormalConfidence = 0.0;      // 最近一次异常样本的置信度

  // ── 状态 ──
  bool _isConnected = false;
  bool _isScanning = false;
  String _statusMessage = '未连接';

  // ── 显示控制 ──
  int _timeWindow = 2;   // 时间窗口：1~6 秒
  double _amplitudeScale = 1.0;  // 幅度缩放：0.5x ~ 3.0x
  String _displayChannel = 'filtered';  // 'clean' | 'noisy' | 'filtered'

  // ── 数据流订阅 ──
  StreamSubscription<ECGSample>? _subscription;

  // ── Getter ──

  List<ECGSample> get samples => _samples;
  bool get isConnected => _isConnected;
  bool get isScanning => _isScanning;
  String get statusMessage => _statusMessage;
  double get heartRate => _heartRate;
  int get droppedCount => _droppedCount;
  int get bufferSize => _samples.length;
  BLEService get bleService => _bleService;
  ECGSample? get lastSample => _samples.isEmpty ? null : _samples.last;

  /// 最近 kAbnormalWindow 个样本内是否存在 AI 异常（防单样本闪烁）
  bool get hasAbnormalAlert {
    final start = _samples.length > kAbnormalWindow
        ? _samples.length - kAbnormalWindow
        : 0;
    for (int i = start; i < _samples.length; i++) {
      if (_samples[i].abnormal == 1) return true;
    }
    return false;
  }

  /// 最近一次异常样本的置信度 (0~1)
  double get abnormalConfidence => _abnormalConfidence;

  // ── 显示控制 Getter/Setter ──

  /// 时间窗口（秒），决定屏幕显示多长时间的波形
  int get timeWindow => _timeWindow;
  set timeWindow(int val) {
    _timeWindow = val.clamp(1, 6);
    notifyListeners();
  }

  /// 当前时间窗口对应的样本数
  int get visibleSamples => _timeWindow * 250;

  /// 幅度缩放系数
  double get amplitudeScale => _amplitudeScale;
  set amplitudeScale(double val) {
    _amplitudeScale = val.clamp(0.5, 3.0);
    notifyListeners();
  }

  /// 显示的通道
  String get displayChannel => _displayChannel;
  set displayChannel(String val) {
    if (['clean', 'noisy', 'filtered'].contains(val)) {
      _displayChannel = val;
      notifyListeners();
    }
  }

  // ── 连接管理 ──

  /// 扫描并连接 ESP32
  Future<void> connect() async {
    _isScanning = true;
    _statusMessage = '正在扫描 ESP32-ECG...';
    notifyListeners();

    final success = await _bleService.connect();

    _isScanning = false;

    if (success) {
      _isConnected = true;
      _statusMessage = '已连接';

      // 监听数据流
      _subscription = _bleService.dataStream.listen(_addSample);

      // 监听断开
      _bleService.onDisconnected = () {
        _isConnected = false;
        _statusMessage = '连接已断开';
        _subscription?.cancel();
        notifyListeners();
      };
    } else {
      _statusMessage = '连接失败：未找到 ESP32-ECG 设备';
    }

    notifyListeners();
  }

  /// 手动断开连接
  Future<void> disconnect() async {
    _subscription?.cancel();
    _resetAlarmStateMachine(); // 立即重置告警（在异步 BLE 断开之前，不组装事件）
    await _bleService.disconnect();
    _isConnected = false;
    _statusMessage = '未连接';
    notifyListeners();
  }

  // ── 数据管理 ──

  /// 测试专用：直接向环形缓冲区注入样本（仅测试使用，不改变任何既有逻辑）
  @visibleForTesting
  void debugAddSample(ECGSample sample) => _addSample(sample);

  /// 添加一个新样本到环形缓冲区
  void _addSample(ECGSample sample) {
    if (_samples.length >= kBufferSize) {
      _samples.removeAt(0);
    }
    _samples.add(sample);

    // 使用 ESP32 板上算法计算的 BPM (来自 CSV 第4列)
    if (sample.bpm > 0) {
      _heartRate = sample.bpm.toDouble();
    }

    // 记录 AI 异常置信度 (来自 CSV 第9列)，供 UI 显示
    if (sample.abnormal == 1) {
      _abnormalConfidence = sample.confidence;
    }

    // 评估告警状态机（基于原始 sample.abnormal，非 hasAbnormalAlert 窗口）
    final alarmChanged = _evaluateAlarm(sample);

    // 告警状态变更时立即通知；否则沿用每 15 样本批量刷新
    if (alarmChanged || _samples.length % 15 == 0 || _samples.length < 15) {
      notifyListeners();
    }
  }

  /// 获取当前显示通道的数值
  double _getChannelValue(ECGSample sample) {
    switch (_displayChannel) {
      case 'clean':
        return sample.clean;
      case 'noisy':
        return sample.noisy;
      case 'filtered':
      default:
        return sample.filtered;
    }
  }

  /// ★★★ 幅度缩放原理 ★★★
  /// 数据本身不变，通过缩小/扩大 Y 轴电压窗口实现视觉缩放
  /// ampScale=2.0 → 窗口缩小一半 → 波形视觉上放大 2 倍
  /// ampScale=0.5 → 窗口扩大一倍 → 波形视觉上缩小一半

  /// 可见范围内原始数据的极值（辅助计算）
  (double, double) _getVisibleRange() {
    if (_samples.isEmpty) return (1.0, -0.2);
    final start = _samples.length > visibleSamples
        ? _samples.length - visibleSamples
        : 0;
    double max = -999, min = 999;
    for (int i = start; i < _samples.length; i++) {
      final val = _getChannelValue(_samples[i]);
      if (val > max) max = val;
      if (val < min) min = val;
    }
    return (max == -999 ? 1.0 : max, min == 999 ? -0.2 : min);
  }

  /// 当前显示范围内的最大值（受 ampScale 控制窗口大小）
  double get maxValue {
    final (rawMax, rawMin) = _getVisibleRange();
    final center = (rawMax + rawMin) / 2;
    final halfRange = ((rawMax - rawMin) / 2).clamp(0.1, 10.0);
    final zoomedHalf = halfRange / _amplitudeScale;  // ÷系数 = 缩小窗口
    return center + zoomedHalf;
  }

  /// 当前显示范围内的最小值
  double get minValue {
    final (rawMax, rawMin) = _getVisibleRange();
    final center = (rawMax + rawMin) / 2;
    final halfRange = ((rawMax - rawMin) / 2).clamp(0.1, 10.0);
    final zoomedHalf = halfRange / _amplitudeScale;
    return center - zoomedHalf;
  }

  /// 获取绘图用的数据点列表（不缩放，原始数值）
  List<double> get displayData {
    if (_samples.isEmpty) return [];
    final start = _samples.length > visibleSamples
        ? _samples.length - visibleSamples
        : 0;
    return _samples
        .sublist(start)
        .map((s) => _getChannelValue(s))
        .toList();
  }

  // ── 告警状态机 (Contract C1) ──

  AlarmState _alarmState = AlarmState.idle;
  DateTime? _alarmTriggerTime;
  double _alarmPeakConfidence = 0.0;

  /// 告警周期内心率运行均值簿记
  double _alarmBpmSum = 0.0;
  int _alarmBpmCount = 0;

  AlarmEvent? _lastCompletedAlarm;
  Timer? _recoveryTimer; // 3 秒干净窗口恢复定时器

  /// 当前告警状态（idle / alarming / arming）
  AlarmState get alarmState => _alarmState;

  /// 当前告警周期的触发时刻（idle 时为 null）
  DateTime? get alarmTriggerTime => _alarmTriggerTime;

  /// 当前告警周期内异常置信度峰值
  double get alarmPeakConfidence => _alarmPeakConfidence;

  /// 最近一次锁定的心率值（来自 ESP32 板上算法，由 _addSample 更新）
  int get lastBpm => _heartRate.round();

  /// 上一次完成的告警事件（IDLE 转换时组装，新触发时清除）
  AlarmEvent? get lastCompletedAlarm => _lastCompletedAlarm;

  /// 用户确认告警 → 立即转入 IDLE（恢复方式 = user_confirm）
  void confirmAlarm() {
    if (_alarmState != AlarmState.idle) {
      _completeEpisode('user_confirm');
      notifyListeners();
    }
  }

  /// 在每次 _addSample 中评估告警状态转换（基于原始 sample.abnormal）
  /// 返回 true 表示发生了需要立即通知的状态变更
  bool _evaluateAlarm(ECGSample sample) {
    bool changed = false;
    final bool isAbnormal = sample.abnormal == 1;

    switch (_alarmState) {
      case AlarmState.idle:
        if (isAbnormal) {
          // 上升沿 → 触发告警（唯一触发点）
          _alarmState = AlarmState.alarming;
          _alarmTriggerTime = DateTime.now();
          _alarmPeakConfidence = sample.confidence;
          _alarmBpmSum = sample.bpm > 0 ? sample.bpm.toDouble() : 0.0;
          _alarmBpmCount = sample.bpm > 0 ? 1 : 0;
          _lastCompletedAlarm = null; // 新周期清除旧事件
          changed = true;
        }
        break;

      case AlarmState.alarming:
      case AlarmState.arming:
        if (isAbnormal) {
          // 取消待处理的恢复定时器
          _recoveryTimer?.cancel();
          _recoveryTimer = null;

          // 锁存到 arming（alarming→arming 不重新通知）
          if (_alarmState == AlarmState.alarming) {
            _alarmState = AlarmState.arming;
            // 不设置 changed = true：NO re-notify of transition
          }

          // 追踪峰值置信度（峰值变化时需要通知 UI 更新显示）
          if (sample.confidence > _alarmPeakConfidence) {
            _alarmPeakConfidence = sample.confidence;
            changed = true;
          }

          // 计入 BPM 运行均值（仅 bpm>0）
          if (sample.bpm > 0) {
            _alarmBpmSum += sample.bpm.toDouble();
            _alarmBpmCount++;
          }
        } else {
          // abnormal==0：启动 3 秒恢复定时器（仅首次）
          if (_recoveryTimer == null) {
            _recoveryTimer = Timer(const Duration(seconds: 3), () {
              _completeEpisode('signal_normal');
              notifyListeners();
            });
          }

          // 恢复倒计时期间 BPM 仍纳入均值
          if (sample.bpm > 0) {
            _alarmBpmSum += sample.bpm.toDouble();
            _alarmBpmCount++;
          }
        }
        break;
    }

    return changed;
  }

  /// 完成当前告警周期，组装 AlarmEvent
  void _completeEpisode(String recoveryMethod) {
    _recoveryTimer?.cancel();
    _recoveryTimer = null;

    final now = DateTime.now();
    final duration = now.difference(_alarmTriggerTime!);
    final avgBpm =
        _alarmBpmCount > 0 ? _alarmBpmSum / _alarmBpmCount : 0.0;

    _lastCompletedAlarm = AlarmEvent(
      triggerTime: _alarmTriggerTime!,
      duration: duration,
      peakConfidence: _alarmPeakConfidence,
      avgBpm: avgBpm,
      recoveryMethod: recoveryMethod,
    );

    _alarmState = AlarmState.idle;
    _alarmTriggerTime = null;
    _alarmPeakConfidence = 0.0;
    _alarmBpmSum = 0.0;
    _alarmBpmCount = 0;
  }

  /// 重置告警状态机（断开 / dispose 时调用，不组装事件）
  void _resetAlarmStateMachine() {
    _recoveryTimer?.cancel();
    _recoveryTimer = null;
    _alarmState = AlarmState.idle;
    _alarmTriggerTime = null;
    _alarmPeakConfidence = 0.0;
    _alarmBpmSum = 0.0;
    _alarmBpmCount = 0;
    _lastCompletedAlarm = null;
  }

  /// 清除数据（覆写以包含告警状态重置）
  void clear() {
    _samples.clear();
    _heartRate = 0;
    _abnormalConfidence = 0.0;
    _resetAlarmStateMachine(); // 同时重置告警状态机
    notifyListeners();
  }

  @override
  void dispose() {
    _subscription?.cancel();
    _bleService.dispose();
    _resetAlarmStateMachine(); // 释放前重置告警（不组装事件）
    super.dispose();
  }
}
