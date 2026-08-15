import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import '../models/ecg_data.dart';
import 'csv_parser.dart';

/**
 * @file ble_service.dart
 * @brief BLE NUS 连接与数据接收服务
 *
 * 对接 ESP32 Nordic UART Service (NUS)：
 * - Service: 6E400001-B5A3-F393-E0A9-E50E24DCCA9E
 * - TX Char: 6E400002-B5A3-F393-E0A9-E50E24DCCA9E (Notify)
 * - RX Char: 6E400003-B5A3-F393-E0A9-E50E24DCCA9E (Write)
 */

class BLEService {
  // NUS UUID
  static const String _nusServiceUuid = '6e400001-b5a3-f393-e0a9-e50e24dcca9e';
  static const String _nusTxUuid = '6e400002-b5a3-f393-e0a9-e50e24dcca9e';
  static const String _nusRxUuid = '6e400003-b5a3-f393-e0a9-e50e24dcca9e';

  // ESP32 广播名称
  static const String _deviceName = 'ESP32-ECG';

  // 扫描超时
  static const Duration _scanTimeout = Duration(seconds: 15);

  BluetoothDevice? _device;
  BluetoothCharacteristic? _txChar;
  BluetoothCharacteristic? _rxChar;

  /**
   * @brief 测试接缝：注入写回调替代真实 BLE write
   *
   * 非 null 时，sendCommand() 优先使用此回调而非 _rxChar.write()，
   * 便于在不依赖 flutter_blue_plus 的单元测试中验证命令发送行为。
   */
  @visibleForTesting
  Future<void> Function(List<int> data)? commandWriteCallback;

  final StreamController<ECGSample> _dataController =
      StreamController<ECGSample>.broadcast();

  /// 数据流：手机端订阅此流以获取实时心电数据
  Stream<ECGSample> get dataStream => _dataController.stream;

  /// 是否已连接
  bool get isConnected => _device?.isConnected ?? false;

  /// 断开回调
  VoidCallback? onDisconnected;
  StreamSubscription<BluetoothConnectionState>? _connectionSub;
  StreamSubscription<List<int>>? _notifySub;
  int _connectionEpoch = 0; // 连接代次：旧设备/旧订阅回调到达时直接忽略

  /// 清理上一连接的订阅与设备引用（重连防泄漏，2026-08-14 阶梯感修复）。
  /// 旧 connectionState / onValueReceived 订阅若不取消，重连后会继续向
  /// dataStream 注入样本，并可能在新连接建立后被旧 disconnected 事件误清空。
  Future<void> _teardownCurrentConnection() async {
    _connectionEpoch++;
    _connectionSub?.cancel();
    _connectionSub = null;
    _notifySub?.cancel();
    _notifySub = null;
    if (_device != null) {
      try {
        await _device!.disconnect();
      } catch (_) {
        // 设备已不可达时忽略断开异常，继续清理引用
      }
    }
    _device = null;
    _txChar = null;
    _rxChar = null;
  }

  /// 扫描并连接 ESP32-ECG 设备（含重试机制）
  Future<bool> connect() async {
    // 确保蓝牙已开启（内含权限请求处理）
    try {
      await FlutterBluePlus.turnOn();
    } catch (_) {
      return false;
    }

    await Future.delayed(const Duration(milliseconds: 300));

    // 尝试 3 次扫描
    for (int attempt = 0; attempt < 3; attempt++) {
      if (attempt > 0) {
        await Future.delayed(const Duration(milliseconds: 500));
      }

      BluetoothDevice? targetDevice = await _scanForDevice();

      if (targetDevice != null) {
        return _connectToDevice(targetDevice);
      }
    }

    return false;
  }

  /// ★★★ 关键修复：移除 withServices 过滤，按设备名称匹配 ★★★
  /// 原因：Realme/小米等手机对自定义 128-bit UUID 过滤支持不完整
  Future<BluetoothDevice?> _scanForDevice() async {
    // 开始扫描所有 BLE 设备
    await FlutterBluePlus.startScan(
      timeout: _scanTimeout,
    );

    BluetoothDevice? targetDevice;
    final stopWatch = Stopwatch()..start();

    await for (final scanResult in FlutterBluePlus.scanResults) {
      // 超时保护
      if (stopWatch.elapsedMilliseconds > _scanTimeout.inMilliseconds + 2000) {
        break;
      }

      for (final result in scanResult) {
        final name = result.device.platformName;
        if (name == _deviceName) {
          targetDevice = result.device;
          break;
        }
      }
      if (targetDevice != null) break;
    }

    await FlutterBluePlus.stopScan();
    return targetDevice;
  }

  /// 连接到指定设备
  Future<bool> _connectToDevice(BluetoothDevice device) async {
    try {
      // 重连前彻底清理上一连接 (2026-08-14 阶梯感修复): 旧订阅/旧设备残留
      // 会让 dataStream 重复注入样本, 或旧 disconnected 事件误清新连接。
      await _teardownCurrentConnection();
      final epoch = ++_connectionEpoch;
      _device = device;
      await device.connect();

      // ★ 2026-08-14 修复 (阶梯感根因之一): MTU 协商。默认 MTU 23 → 每帧
      // ~50B 拆成 ~3 个 ATT 包 → 250Hz notify 包量 ×3 → 链路拥塞丢帧 →
      // App 按固定 250Hz 时间轴绘制 → 波形阶梯/粗糙。MTU 185 后单帧 1 包。
      try {
        await device.requestMtu(185);
      } catch (_) {
        // 平台不支持或协商失败 → 保持默认 MTU (无副作用)
      }

      // 发现服务
      await device.discoverServices();
      for (final svc in device.servicesList) {
        if (svc.uuid.toString().toLowerCase() == _nusServiceUuid) {
          for (final chr in svc.characteristics) {
            if (chr.uuid.toString().toLowerCase() == _nusTxUuid) {
              _txChar = chr;
              // 监听 Notify
              await chr.setNotifyValue(true);
              _notifySub?.cancel();
                _notifySub = chr.onValueReceived.listen(_onDataReceived);
            } else if (chr.uuid.toString().toLowerCase() == _nusRxUuid) {
              _rxChar = chr;
            }
          }
        }
      }

      // 监听断开事件
      _connectionSub?.cancel();
        _connectionSub = device.connectionState.listen((state) {
          if (epoch != _connectionEpoch) return; // 旧代次回调：忽略
        if (state == BluetoothConnectionState.disconnected) {
            _connectionEpoch++;
            _connectionSub?.cancel();
            _connectionSub = null;
            _notifySub?.cancel();
            _notifySub = null;
          _device = null;
          _txChar = null;
          _rxChar = null;
          onDisconnected?.call();
        }
      });

      // ★ 2026-08-10 候选修复 (遗留A: 重连后波形分辨率低) + 2026-08-14 强化:
      // 请求高优先级连接参数 (Android), 收紧连接间隔提升 notify 实际数据率。
      // 重连累积变粗糙根因: 连接刚建立时 GATT 未稳定, requestConnectionPriority
      // 会静默失败 → Android 回落到默认大间隔 (且重连次数越多间隔越退化),
      // 退出 App 重建 BLE 栈才恢复。改为: 等 200ms 稳定后再请求, 失败重试一次。
      // 注意: flutter_blue_plus 1.32+/1.36+ API 为命名参数
      // requestConnectionPriority({required connectionPriorityRequest})
      await Future.delayed(const Duration(milliseconds: 200));
      for (int attempt = 0; attempt < 2; attempt++) {
        try {
          await device.requestConnectionPriority(
            connectionPriorityRequest: ConnectionPriority.high,
          );
          break; // 成功即退出
        } catch (_) {
          // 平台不支持 (iOS) 或协商失败 → 间隔 300ms 后重试一次
          if (attempt == 0) {
            await Future.delayed(const Duration(milliseconds: 300));
          }
        }
      }

      return _txChar != null;
    } catch (e) {
        _connectionSub?.cancel();
        _connectionSub = null;
        _notifySub?.cancel();
        _notifySub = null;
        _device = null;
        _txChar = null;
        _rxChar = null;
      return false;
    }
  }

  /// 收到 CSV 数据：解析为 ECGSample
  /// CSV 格式: clean,noisy,filtered,bpm,true_bpm,sqi,motion,abnormal_flag,confidence
  /// 固件 4 帧以 ';' 批量拼接，按帧分割解析（2026-08-10 修复多帧错位）
  void _onDataReceived(List<int> value) {
    final str = utf8.decode(value).trim();
    if (str.isEmpty) return;

    // 按 ';' 分割批量帧，逐帧解析（parseBleFrames 内跳过无效帧）
    final samples = parseBleFrames(str);
    for (final sample in samples) {
      _dataController.add(sample);
    }
  }

  /**
   * @brief 发送命令到 ESP32 NUS RX 特征值
   *
   * 通过 NUS RX Characteristic (6E400003-...) 以 Write Without Response
   * 方式发送字符串命令。测试模式下优先使用 commandWriteCallback。
   *
   * ★ 2026-08-10 修复: 命令追加 '\n' 结束符。
   * 固件 BLE 命令解析器 (ble.cpp RxCallbacks) 仅收到 '\n'/'\r'/'\0' 时才提交
   * 命令——此前 App 发送裸字节 (无结束符), 命令永久卡在固件行缓冲,
   * 定时录制等 BLE 命令从未送达设备 (真机 count=0, 仅单元测试 mock 通过)。
   *
   * @param cmd 待发送的命令字符串（如 "REC_START"）
   */
  Future<void> sendCommand(String cmd) async {
    final writer = commandWriteCallback;
    // 统一追加 '\n' (0x0A) 命令结束符, 真实 BLE 与测试接缝行为一致
    final payload = <int>[...cmd.codeUnits, 0x0A];
    if (writer != null) {
      await writer(payload);
      return;
    }
    if (_rxChar == null) {
      return;
    }
    await _rxChar!.write(payload, withoutResponse: true);
  }

  /// 断开连接
  Future<void> disconnect() async {
    await _teardownCurrentConnection();
    _device = null;
    _txChar = null;
    _rxChar = null;
  }

  /// 释放资源
  void dispose() {
    _connectionSub?.cancel();
      _notifySub?.cancel();
      _connectionEpoch++;
    _dataController.close();
  }
}
