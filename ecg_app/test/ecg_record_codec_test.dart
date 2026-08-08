import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:ecg_app/pages/playback_page.dart';
import 'package:ecg_app/services/ecg_record_codec.dart';
import 'package:ecg_app/widgets/ecg_waveform.dart';

/**
 * @brief ECGR 记录编解码 + 回放页测试
 *
 * 字节布局以固件 include/storage/ecg_recorder_format.h 为准（Contract C5）：
 * 32 字节小端头部 + totalSamples × int16 LE 样本流 + 可选 durationSec 字节异常位图。
 *
 * 覆盖：
 * - 解码：正常 3s fixture、异常位图、坏魔数/坏版本/截断、空样本
 * - 回放页：波形渲染、播放/暂停、seek、异常标识、自动停止、定时器无泄漏
 */

// ─────────────────────────── 测试内 fixture 构建 ───────────────────────────
// fixture 构建器仅存在于测试文件（不为生产代码增加测试 API）。

/// 确定性样本序列（覆盖正/负/极值 int16）
const List<int> _sampleCycle = [0, 1000, -2000, 3000, -4000, 32767, -32768];

void _putU32LE(Uint8List b, int off, int v) {
  b[off] = v & 0xFF;
  b[off + 1] = (v >> 8) & 0xFF;
  b[off + 2] = (v >> 16) & 0xFF;
  b[off + 3] = (v >> 24) & 0xFF;
}

void _putI16LE(Uint8List b, int off, int v) {
  b[off] = v & 0xFF;
  b[off + 1] = (v >> 8) & 0xFF;
}

/// 按 Contract C5 合成 .ecgr 字节（小端）
Uint8List _buildFixture({
  int sampleRate = 250,
  int startUnixTime = 1700000000,
  int durationSec = 3,
  List<int>? abnormalBitmap,
  List<int>? samples,
}) {
  final totalSamples = samples?.length ?? sampleRate * durationSec;
  final bitmap = abnormalBitmap;
  final size = 32 + totalSamples * 2 + (bitmap != null ? durationSec : 0);
  final bytes = Uint8List(size);

  // 头部
  bytes[0] = 0x45;
  bytes[1] = 0x43;
  bytes[2] = 0x47;
  bytes[3] = 0x52; // 'ECGR'
  bytes[4] = 1; // version
  bytes[5] = bitmap != null ? 0x01 : 0x00; // flags: bit0 = 异常位图
  _putU32LE(bytes, 6, sampleRate);
  _putU32LE(bytes, 10, startUnixTime);
  _putU32LE(bytes, 14, durationSec);
  _putU32LE(bytes, 18, totalSamples);
  _putU32LE(bytes, 22, bitmap == null ? 0 : bitmap.where((b) => b == 1).length);

  // 样本流
  for (int i = 0; i < totalSamples; i++) {
    final v = samples != null ? samples[i] : _sampleCycle[i % _sampleCycle.length];
    _putI16LE(bytes, 32 + i * 2, v);
  }

  // 异常位图
  if (bitmap != null) {
    final base = 32 + totalSamples * 2;
    for (int i = 0; i < durationSec; i++) {
      bytes[base + i] = bitmap[i] & 0xFF;
    }
  }
  return bytes;
}

// ─────────────────────────── 解码单元测试 ───────────────────────────

void main() {
  group('EcgRecordCodec.decode', () {
    test('3 秒 fixture（750 样本，位图全 0）：字段完整 + volts = int16/8000', () {
      final bytes = _buildFixture(abnormalBitmap: [0, 0, 0]);
      final record = EcgRecordCodec.decode(bytes);

      expect(record, isNotNull);
      expect(record!.sampleRate, 250);
      expect(record.startUnixTime, 1700000000);
      expect(record.durationSec, 3);
      expect(record.totalSamples, 750);
      expect(record.abnormalSeconds, 0);
      expect(record.hasBitmap, isTrue);
      expect(record.abnormalBySecond, [0, 0, 0]);

      expect(record.samplesV.length, 750);
      // int16 → volts（固件标定 1.0V = 8000.0）
      expect(record.samplesV[0], closeTo(0, 1e-9));
      expect(record.samplesV[1], closeTo(1000 / 8000.0, 1e-9));
      expect(record.samplesV[2], closeTo(-2000 / 8000.0, 1e-9));
      expect(record.samplesV[3], closeTo(3000 / 8000.0, 1e-9));
      expect(record.samplesV[4], closeTo(-4000 / 8000.0, 1e-9));
      expect(record.samplesV[5], closeTo(32767 / 8000.0, 1e-9));
      expect(record.samplesV[6], closeTo(-32768 / 8000.0, 1e-9));
      expect(record.samplesV[749], closeTo(0, 1e-9)); // 749 % 7 == 0
    });

    test('无位图 record：hasBitmap=false，abnormalBySecond 为空', () {
      final record = EcgRecordCodec.decode(_buildFixture())!;
      expect(record.hasBitmap, isFalse);
      expect(record.abnormalBySecond, isEmpty);
      expect(record.abnormalSeconds, 0);
      expect(record.totalSamples, 750);
    });

    test('异常位图 [1,0,1]：abnormalBySecond 逐秒正确，abnormalSeconds=2', () {
      final record = EcgRecordCodec.decode(
          _buildFixture(abnormalBitmap: [1, 0, 1]))!;
      expect(record.hasBitmap, isTrue);
      expect(record.abnormalBySecond, [1, 0, 1]);
      expect(record.abnormalSeconds, 2);
      // 位图字节紧跟在样本流之后，样本解码不受位图偏移影响
      expect(record.samplesV.length, 750);
      expect(record.samplesV[1], closeTo(1000 / 8000.0, 1e-9));
    });

    test('坏魔数 / 坏版本 / 头部不足 32 字节 → null', () {
      final valid = _buildFixture();

      final badMagic = Uint8List.fromList(valid);
      badMagic[0] = 0x58; // 'X'
      expect(EcgRecordCodec.decode(badMagic), isNull);
      expect(EcgRecordCodec.validateHeader(badMagic), isFalse);

      final badVersion = Uint8List.fromList(valid);
      badVersion[4] = 2;
      expect(EcgRecordCodec.decode(badVersion), isNull);
      expect(EcgRecordCodec.validateHeader(badVersion), isFalse);

      // 不足 32 字节
      expect(EcgRecordCodec.decode(Uint8List(31)), isNull);
      expect(EcgRecordCodec.validateHeader(Uint8List(10)), isFalse);
    });

    test('样本流截断（size < 32 + 2×totalSamples）→ null（文档化约定）', () {
      final valid = _buildFixture(); // 750 样本 = 1532 字节
      // 仅保留 500 个样本
      final truncated = Uint8List.sublistView(valid, 0, 32 + 2 * 500);
      expect(EcgRecordCodec.decode(truncated), isNull);
      // 位图模式截断同样拒绝
      final withBitmap = _buildFixture(abnormalBitmap: [1, 0, 1]);
      final truncatedBitmap =
          Uint8List.sublistView(withBitmap, 0, 32 + 2 * 750); // 缺 3 字节位图
      expect(EcgRecordCodec.decode(truncatedBitmap), isNull);
    });

    test('validateHeader：合法头部返回 true', () {
      expect(EcgRecordCodec.validateHeader(_buildFixture()), isTrue);
    });

    test('totalSamples=0 边界：空样本流可正常解码', () {
      // 无位图：仅 32 字节头部
      final empty = _buildFixture(durationSec: 0, samples: []);
      final r0 = EcgRecordCodec.decode(empty)!;
      expect(r0.totalSamples, 0);
      expect(r0.samplesV, isEmpty);
      expect(r0.hasBitmap, isFalse);
      expect(r0.durationSec, 0);

      // 有位图：位图从 32+0 偏移处开始
      final emptyBitmap =
          _buildFixture(durationSec: 3, samples: [], abnormalBitmap: [0, 1, 0]);
      final r1 = EcgRecordCodec.decode(emptyBitmap)!;
      expect(r1.totalSamples, 0);
      expect(r1.samplesV, isEmpty);
      expect(r1.hasBitmap, isTrue);
      expect(r1.abnormalBySecond, [0, 1, 0]);
      expect(r1.abnormalSeconds, 1);
    });
  });

  // ─────────────────────────── 回放页 Widget 测试 ───────────────────────────

  group('PlaybackPage', () {
    testWidgets('渲染波形 + 播放/暂停切换 + 250Hz 定时喂样推进时间', (tester) async {
      final record = EcgRecordCodec.decode(_buildFixture())!; // 3s / 750 样本
      await tester.pumpWidget(MaterialApp(home: PlaybackPage(record: record)));

      // 初始：波形 + 控制项 + 无异常标识（无位图 record）
      expect(find.byType(ECGWaveform), findsOneWidget);
      expect(find.byType(Slider), findsOneWidget);
      expect(find.byIcon(Icons.play_arrow), findsOneWidget);
      expect(find.text('记录回放'), findsOneWidget);
      expect(find.text('00:00 / 00:03'), findsOneWidget);
      expect(find.text('异常'), findsNothing);

      // 播放：图标切换为暂停；1 秒后（250 tick @4ms）时间标签前进到 00:01
      await tester.tap(find.byIcon(Icons.play_arrow));
      await tester.pump();
      expect(find.byIcon(Icons.pause), findsOneWidget);

      await tester.pump(const Duration(seconds: 1));
      expect(find.text('00:01 / 00:03'), findsOneWidget);

      // 暂停：再等 1 秒时间不再前进
      await tester.tap(find.byIcon(Icons.pause));
      await tester.pump();
      await tester.pump(const Duration(seconds: 1));
      expect(find.text('00:01 / 00:03'), findsOneWidget);
      expect(find.byIcon(Icons.play_arrow), findsOneWidget);

      // 卸载页面：定时器在 dispose 中取消，测试结束时无 pending timer
      await tester.pumpWidget(const SizedBox());
    });

    testWidgets('seek 滑块跳转 + 播完自动停止 + 末尾重播从头开始', (tester) async {
      final record = EcgRecordCodec.decode(_buildFixture())!;
      await tester.pumpWidget(MaterialApp(home: PlaybackPage(record: record)));

      // 拖动滑块到最右 → seek 到 3 秒（样本索引跳到 750）
      await tester.drag(find.byType(Slider), const Offset(500, 0));
      await tester.pump();
      expect(find.text('00:03 / 00:03'), findsOneWidget);

      // 末尾按播放 → 从头重播
      await tester.tap(find.byIcon(Icons.play_arrow));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 250));
      expect(find.text('00:00 / 00:03'), findsOneWidget);

      // 播完自动停止：回到播放图标，时间到末尾，定时器已取消
      await tester.pump(const Duration(seconds: 3));
      expect(find.text('00:03 / 00:03'), findsOneWidget);
      expect(find.byIcon(Icons.play_arrow), findsOneWidget);

      await tester.pumpWidget(const SizedBox());
    });

    testWidgets('异常秒显示红色异常标识（位图 [1,0,1]）', (tester) async {
      final record = EcgRecordCodec.decode(
          _buildFixture(abnormalBitmap: [1, 0, 1]))!;
      await tester.pumpWidget(MaterialApp(home: PlaybackPage(record: record)));

      // 初始停在 0 秒（异常）→ 显示标识
      expect(find.text('异常'), findsOneWidget);

      // 播放 1.5 秒 → 第 1 秒（正常）→ 标识消失
      await tester.tap(find.byIcon(Icons.play_arrow));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 1500));
      expect(find.text('异常'), findsNothing);

      // 播放到 2.5 秒 → 第 2 秒（异常）→ 标识再次出现
      await tester.pump(const Duration(milliseconds: 1000));
      expect(find.text('异常'), findsOneWidget);

      await tester.pumpWidget(const SizedBox());
    });
  });
}
