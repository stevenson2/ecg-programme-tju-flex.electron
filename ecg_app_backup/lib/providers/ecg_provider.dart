import 'dart:async';
import 'package:flutter/foundation.dart';
import '../models/ecg_data.dart';
import '../services/ble_service.dart';

/**
 * @file ecg_provider.dart
 * @brief 心电数据状态管理（ChangeNotifier + Provider）
 *
 * 管理：
 * - 历史数据环形缓冲区（最新 500 点 = 2 秒）
 * - 连接状态
 * - 心率计算
 * - 信号质量评估
 */

class ECGProvider extends ChangeNotifier {
  // ── BLE 服务 ──
  final BLEService _bleService = BLEService();

  // ── 环形缓冲区 ──
  static const int kBufferSize = 500; // 2 秒 @ 250Hz
  final List<ECGSample> _samples = [];
  int _droppedCount = 0;

  // ── 心率计算 ──
  static const int kMinHR = 30; // bpm
  static const int kMaxHR = 220;
  static const double kRPeakThreshold = 0.3; // V
  int _lastRPeakIndex = -kBufferSize;
  double _heartRate = 0;

  // ── 状态 ──
  bool _isConnected = false;
  bool _isScanning = false;
  String _statusMessage = '未连接';

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

    // 检测 R 波并计算心率
    _detectRPeak(sample);

    // 每 20 个样本通知一次 UI 刷新（性能优化）
    if (_samples.length % 20 == 0 || _samples.length < 20) {
      notifyListeners();
    }
  }

  /// 心电信号最大值最小值（用于绘图缩放）
  double get maxValue {
    if (_samples.isEmpty) return 1.2;
    return _samples.map((s) => [s.clean, s.noisy, s.filtered]
        .reduce((a, b) => a > b ? a : b))
        .reduce((a, b) => a > b ? a : b);
  }

  double get minValue {
    if (_samples.isEmpty) return -0.2;
    return _samples.map((s) => [s.clean, s.noisy, s.filtered]
        .reduce((a, b) => a < b ? a : b))
        .reduce((a, b) => a < b ? a : b);
  }

  // ── R 波检测与心率计算 ──

  void _detectRPeak(ECGSample sample) {
    // 用滤波后的信号检测
    final val = sample.filtered;

    // 阈值检测
    if (val > kRPeakThreshold) {
      final currentIndex = _samples.length - 1;
      final interval =
          (currentIndex - _lastRPeakIndex).abs();

      // 最小间隔防止误检（> 200ms = 50 点 @250Hz）
      if (interval > 50 && interval < 500) {
        final hr = 60.0 / (interval / 250.0);
        if (hr >= kMinHR && hr <= kMaxHR) {
          // 滑动平均
          _heartRate = _heartRate == 0
              ? hr
              : _heartRate * 0.7 + hr * 0.3;
        }
        _lastRPeakIndex = currentIndex;
      }
    }
  }

  /// 清除数据
  void clear() {
    _samples.clear();
    _heartRate = 0;
    _lastRPeakIndex = -kBufferSize;
    notifyListeners();
  }

  @override
  void dispose() {
    _subscription?.cancel();
    _bleService.dispose();
    super.dispose();
  }
}
