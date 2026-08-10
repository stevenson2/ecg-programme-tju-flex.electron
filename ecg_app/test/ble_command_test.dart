import 'package:flutter_test/flutter_test.dart';

import 'package:ecg_app/services/ble_service.dart';

/**
 * @file ble_command_test.dart
 * @brief BLEService sendCommand() 单元测试
 *
 * 通过 commandWriteCallback 测试接缝验证命令发送行为，
 * 不依赖 flutter_blue_plus 真实 BLE 硬件。
 *
 * 2026-08-10: sendCommand 追加 '\n' (0x0A) 命令结束符 (固件 BLE 解析器
 * 仅以 '\n'/'\r'/'\0' 提交命令), 断言同步更新。
 */
void main() {
  group('BLEService sendCommand', () {
    test('通过 commandWriteCallback 接缝写入命令字节(含 \\n 结束符)', () async {
      final ble = BLEService();
      String? receivedCmd;
      List<int>? receivedBytes;

      // 注入测试写回调
      ble.commandWriteCallback = (data) async {
        receivedBytes = List<int>.from(data);
        receivedCmd = String.fromCharCodes(data);
      };

      await ble.sendCommand('RECORDS:LIST');

      expect(receivedCmd, 'RECORDS:LIST\n');
      expect(receivedBytes, [...'RECORDS:LIST'.codeUnits, 0x0A]);
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

      expect(commands, ['CMD1\n', 'CMD2\n', 'CMD3\n']);
    });

    test('空字符串也能发送(仅结束符, 固件忽略空行)', () async {
      final ble = BLEService();
      List<int>? receivedBytes;

      ble.commandWriteCallback = (data) async {
        receivedBytes = data;
      };

      await ble.sendCommand('');

      // 空命令追加 '\n' 后为单个 0x0A; 固件行缓冲为空时忽略该结束符
      expect(receivedBytes, [0x0A]);
    });

    test('无 callback 且无连接时不抛异常（优雅降级）', () async {
      final ble = BLEService();
      // commandWriteCallback == null, _rxChar == null → 应静默返回
      await ble.sendCommand('ANY');
      // 不抛异常即通过
    });
  });
}
