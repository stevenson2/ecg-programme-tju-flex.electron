import 'dart:async';
import 'dart:convert';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import '../models/ecg_data.dart';

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

  // ESP32 广播名称
  static const String _deviceName = 'ESP32-ECG';

  // 扫描超时
  static const Duration _scanTimeout = Duration(seconds: 15);

  BluetoothDevice? _device;
  BluetoothCharacteristic? _txChar;

  final StreamController<ECGSample> _dataController =
      StreamController<ECGSample>.broadcast();

  /// 数据流：手机端订阅此流以获取实时心电数据
  Stream<ECGSample> get dataStream => _dataController.stream;

  /// 是否已连接
  bool get isConnected => _device != null && _device!.isConnected;

  /// 断开回调
  VoidCallback? onDisconnected;

  /// 扫描并连接 ESP32-ECG 设备
  Future<bool> connect() async {
    // 请求蓝牙权限
    try {
      await FlutterBluePlus.turnOn();
    } catch (_) {}

    // 开始扫描
    await FlutterBluePlus.startScan(
      withServices: [Guid(_nusServiceUuid)],
      timeout: _scanTimeout,
    );

    // 等待找到目标设备
    BluetoothDevice? targetDevice;

    await for (final scanResult in FlutterBluePlus.scanResults) {
      for (final result in scanResult) {
        final name = result.device.remoteName;
        if (name == _deviceName) {
          targetDevice = result.device;
          break;
        }
      }
      if (targetDevice != null) break;
    }

    await FlutterBluePlus.stopScan();

    if (targetDevice == null) {
      return false;
    }

    return _connectToDevice(targetDevice);
  }

  /// 连接到指定设备
  Future<bool> _connectToDevice(BluetoothDevice device) async {
    try {
      _device = device;
      await device.connect();

      // 发现服务
      await device.discoverServices();
      for (final svc in device.services) {
        if (svc.uuid.toString().toLowerCase() == _nusServiceUuid) {
          for (final chr in svc.characteristics) {
            if (chr.uuid.toString().toLowerCase() == _nusTxUuid) {
              _txChar = chr;
              // 监听 Notify
              await chr.setNotifyValue(true);
              chr.onValueReceived.listen(_onDataReceived);
              break;
            }
          }
        }
      }

      // 监听断开事件
      device.onDisconnected.listen((_) {
        _device = null;
        _txChar = null;
        onDisconnected?.call();
      });

      return _txChar != null;
    } catch (e) {
      return false;
    }
  }

  /// 收到 CSV 数据：解析为 ECGSample
  void _onDataReceived(List<int> value) {
    final str = utf8.decode(value).trim();
    if (str.isEmpty) return;

    final parts = str.split(',');
    if (parts.length < 3) return;

    try {
      final clean = double.parse(parts[0].trim());
      final noisy = double.parse(parts[1].trim());
      final filtered = double.parse(parts[2].trim());

      _dataController.add(ECGSample(clean, noisy, filtered));
    } catch (_) {
      // 解析失败，跳过
    }
  }

  /// 断开连接
  Future<void> disconnect() async {
    await _device?.disconnect();
    _device = null;
    _txChar = null;
  }

  /// 释放资源
  void dispose() {
    _dataController.close();
  }
}
