import 'dart:async';
import 'package:flutter/foundation.dart';
import '../models/ecg_data.dart';
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

class ECGProvider extends ChangeNotifier {
  // ── BLE 服务 ──
  final BLEService _bleService = BLEService();

  // ── 环形缓冲区 ──
  static const int kBufferSize = 1500; // 6 秒 @ 250Hz
  final List<ECGSample> _samples = [];
  int _droppedCount = 0;

  // ── 心率 (来自 ESP32 板上算法) ──
  double _heartRate = 0;

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
    await _bleService.disconnect();
    _isConnected = false;
    _statusMessage = '未连接';
    notifyListeners();
  }

  // ── 数据管理 ──

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

    // 每 15 个样本通知一次 UI 刷新
    if (_samples.length % 15 == 0 || _samples.length < 15) {
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

  /// 清除数据
  void clear() {
    _samples.clear();
    _heartRate = 0;
    notifyListeners();
  }

  @override
  void dispose() {
    _subscription?.cancel();
    _bleService.dispose();
    super.dispose();
  }
}
