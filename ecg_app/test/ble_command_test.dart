import 'package:flutter_test/flutter_test.dart';

import 'package:ecg_app/services/ble_service.dart';

/**
 * @file ble_command_test.dart
 * @brief BLEService sendCommand() 单元测试
 *
 * 通过 commandWriteCallback 测试接缝验证命令发送行为，
 * 不依赖 flutter_blue_plus 真实 BLE 硬件。
 */
void main() {
  group('BLEService sendCommand', () {
    test('通过 commandWriteCallback 接缝写入命令字节', () async {
      final ble = BLEService();
      String? receivedCmd;
      List<int>? receivedBytes;

      // 注入测试写回调
      ble.commandWriteCallback = (data) async {
        receivedBytes = List<int>.from(data);
        receivedCmd = String.fromCharCodes(data);
      };

      await ble.sendCommand('RECORDS:LIST');

      expect(receivedCmd, 'RECORDS:LIST');
      expect(receivedBytes, 'RECORDS:LIST'.codeUnits);
    });

    test('sendCommand 调用多次各自独立', () async {
      final ble = BLEService();
      final commands = <String>[];

      ble.commandWriteCallback = (data) async {
        commands.add(String.fromCharCodes(data));
      };

      await ble.sendCommand('CMD1');
      await ble.sendCommand('CMD2');
      await ble.sendCommand('CMD3');

      expect(commands, ['CMD1', 'CMD2', 'CMD3']);
    });

    test('空字符串也能发送', () async {
      final ble = BLEService();
      List<int>? receivedBytes;

      ble.commandWriteCallback = (data) async {
        receivedBytes = data;
      };

      await ble.sendCommand('');

      expect(receivedBytes, isEmpty);
    });

    test('无 callback 且无连接时不抛异常（优雅降级）', () async {
      final ble = BLEService();
      // commandWriteCallback == null, _rxChar == null → 应静默返回
      await ble.sendCommand('ANY');
      // 不抛异常即通过
    });
  });
}
