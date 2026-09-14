import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';

import 'package:ecg_app/services/csv_parser.dart';
import 'package:ecg_app/services/ecg_record_codec.dart';
import 'package:ecg_app/services/protocol_generated.dart';

/// P0-1 金样测试（App 端）：与 Web/Node、C++ 测试共享
/// protocol/golden/* fixture，逐字段断言三端一致。
void main() {
  late Map<String, dynamic> framesDoc;
  late Map<String, dynamic> ecgrDoc;

  late String repoRoot;

  setUpAll(() {
    // flutter test 的 cwd 是 ecg_app/，协议 fixture 在仓库根 protocol/。
    final cwd = Directory.current.path;
    repoRoot = Directory('$cwd/../protocol').existsSync() ? '$cwd/..' : cwd;
    framesDoc = jsonDecode(
        File('$repoRoot/protocol/golden/ble_frames.json').readAsStringSync())
        as Map<String, dynamic>;
    ecgrDoc = jsonDecode(
        File('$repoRoot/protocol/golden/ecgr_expected.json').readAsStringSync())
        as Map<String, dynamic>;
  });

  group('BLE CSV 金样', () {
    test('v1 / v2 帧逐字段一致', () {
      final frames = framesDoc['frames'] as Map<String, dynamic>;
      final expected = framesDoc['expected'] as Map<String, dynamic>;

      for (final entry in expected.entries) {
        final name = entry.key;
        final exp = entry.value as Map<String, dynamic>;
        final sample = parseEcgCsvLine(frames[name] as String);
        expect(sample, isNotNull, reason: name);
        expect(sample!.clean, exp['clean'] ?? sample.clean, reason: name);
        expect(sample.bpm, exp['bpm'] ?? sample.bpm, reason: name);
        expect(sample.abnormal, exp['abnormal'], reason: name);
        expect(sample.asrc, exp['asrc'], reason: name);
      }

      // v1 缺第 10 列 -> asrc 降级 0x01；v2 多比特保留
      expect(parseEcgCsvLine(frames['v1_frame'] as String)!.asrc, 0);
      expect(parseEcgCsvLine(frames['v1_frame_abnormal'] as String)!.asrc, 0x01);
      expect(parseEcgCsvLine(frames['v2_frame_multibit'] as String)!.asrc, 0x14);
      expect(parseEcgCsvLine(frames['diagnostic_line'] as String), isNull);
    });

    test('批量帧按分号拆分（含 HELLO 不进入样本流）', () {
      final frames = framesDoc['frames'] as Map<String, dynamic>;
      final samples = parseBleFrames(frames['v2_batch'] as String);
      expect(samples.length, 2);
      expect(samples[0].asrc, 0);
      expect(samples[1].asrc, 0x01);
    });
  });

  group('ECGR v1/v2 金样', () {
    test('v1 与 v2 解码逐字段一致 + 截断容忍', () {
      for (final entry in ecgrDoc.entries) {
        final file = entry.key;
        final exp = entry.value as Map<String, dynamic>;
        final bytes = File('$repoRoot/protocol/golden/$file')
            .readAsBytesSync();

        final record = EcgRecordCodec.decode(Uint8List.fromList(bytes));
        expect(record, isNotNull, reason: file);
        expect(record!.version, exp['version'], reason: file);
        expect(record.sampleRate, exp['sampleRate'], reason: file);
        expect(record.durationSec, exp['durationSec'], reason: file);
        expect(record.totalSamples, exp['totalSamples'], reason: file);
        expect(record.abnormalSeconds, exp['abnormalSeconds'], reason: file);
        expect(record.hasBitmap, exp['hasBitmap'], reason: file);
        expect(record.bitmapIsAsrc, exp['bitmapIsAsrc'], reason: file);
        expect(record.abnormalBySecond, exp['bitmap'], reason: file);
        expect(record.truncated, exp['truncated'], reason: file);
        // v1 的 asrc 视图：0/1 -> 0x01/0x00
        final asrcExp = (exp['asrcBySecond'] as List)
            .map((e) => exp['bitmapIsAsrc'] == true ? e : (e != 0 ? 1 : 0))
            .toList();
        expect(record.asrcBySecond, asrcExp, reason: file);
      }
    });

    test('非法头部拒收（魔数/版本）', () {
      final bytes = Uint8List(32);
      expect(EcgRecordCodec.decode(bytes), isNull);
      final valid = File('$repoRoot/protocol/golden/ecgr_v1.bin')
          .readAsBytesSync();
      final badMagic = Uint8List.fromList(valid)..[0] = 0x58;
      expect(EcgRecordCodec.decode(badMagic), isNull);
    });
  });

  group('asrc 分级', () {
    test('协议生成常量提供优先级文案', () {
      expect(ProtocolGenerated.primaryAsrc(0x10)!.name, 'LEADOFF');
      expect(ProtocolGenerated.primaryAsrc(0x01)!.name, 'AI');
      expect(ProtocolGenerated.primaryAsrc(0x1C)!.name, 'LEADOFF');
      expect(ProtocolGenerated.primaryAsrc(0), isNull);
    });
  });
}