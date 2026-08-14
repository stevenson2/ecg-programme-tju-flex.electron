import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:ecg_app/services/record_api.dart';
import 'package:ecg_app/services/ecg_record_codec.dart';
import 'package:ecg_app/pages/record_list_page.dart';

/**
 * @file record_list_page_test.dart
 * @brief RecordListPage 组件测试（注入假 RecordApi + 临时目录）
 *
 * 覆盖：AP 引导横幅、记录列表渲染、下载写文件、删除刷新、空状态、本地回放跳转。
 */

// ─────────────────────────── 测试内 fixture 构建 ───────────────────────────
// 与 ecg_record_codec_test.dart 同构：按 Contract C5 合成 .ecgr 字节（小端）。

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

/// 合成最小合法 .ecgr（1 秒 / 250 样本，无异常位图）
Uint8List _buildFixture({int durationSec = 1}) {
  final totalSamples = 250 * durationSec;
  final bytes = Uint8List(32 + totalSamples * 2);
  bytes[0] = 0x45; // 'E'
  bytes[1] = 0x43; // 'C'
  bytes[2] = 0x47; // 'G'
  bytes[3] = 0x52; // 'R'
  bytes[4] = 1; // version
  bytes[5] = 0; // flags: 无异常位图
  _putU32LE(bytes, 6, 250); // sampleRate
  _putU32LE(bytes, 10, 1700000000); // startUnixTime
  _putU32LE(bytes, 14, durationSec);
  _putU32LE(bytes, 18, totalSamples);
  _putU32LE(bytes, 22, 0); // abnormalSeconds
  for (int i = 0; i < totalSamples; i++) {
    _putI16LE(bytes, 32 + i * 2, i % 7 == 0 ? 0 : 1000);
  }
  return bytes;
}

void main() {
  /// 构造标准列表数据
  List<RecordInfo> _sampleRecords() => [
        RecordInfo(
          id: 1,
          duration: 60,
          size: 146000,
          abnormalSeconds: 3,
          start: '2026-08-08T10:00:00Z',
        ),
        RecordInfo(
          id: 2,
          duration: 30,
          size: 73000,
          abnormalSeconds: 0,
          start: '2026-08-08T10:05:00Z',
        ),
      ];

  Map<String, dynamic> _testMetaJson() => {
        'id': 1,
        'sample_rate': 250,
        'start_unix': 1723104000,
        'duration': 60,
        'total_samples': 15000,
        'abnormal_seconds': 3,
      };

  /// 创建假 RecordApi（MockClient 注入）
  RecordApi _fakeApi({
    List<RecordInfo>? records,
    bool deleteSuccess = true,
    Uint8List? downloadBytes,
  }) {
    final list = records ?? _sampleRecords();
    final client = MockClient((req) async {
      final path = req.url.path;
      final method = req.method;

      // list
      if (path == '/api/records' && method == 'GET') {
        final json = {
          'records': list
              .map((r) => {
                    'id': r.id,
                    'duration': r.duration,
                    'size': r.size,
                    'abnormal_seconds': r.abnormalSeconds,
                    'start': r.start,
                  })
              .toList(),
        };
        return http.Response(
            '{"records":${jsonEncode(json['records'])}}', 200);
      }
      // meta
      if (path.startsWith('/api/records/') && path.endsWith('/meta')) {
        return http.Response(jsonEncode(_testMetaJson()), 200);
      }
      // download
      if (path.startsWith('/api/records/') && path.endsWith('/data')) {
        final bytes = downloadBytes ?? Uint8List.fromList([1, 2, 3, 4, 5]);
        return http.Response.bytes(bytes, 200,
            headers: {'Content-Length': '${bytes.length}'});
      }
      // delete
      if (method == 'DELETE') {
        if (deleteSuccess) {
          return http.Response('{"deleted":true}', 200);
        }
        return http.Response('error', 500);
      }
      return http.Response('not found', 404);
    });
    return RecordApi(client: client);
  }

  group('RecordListPage 渲染', () {
    testWidgets('显示 AP 引导横幅与中文提示', (tester) async {
      final api = _fakeApi(records: []);
      await tester.pumpWidget(MaterialApp(home: RecordListPage(api: api)));
      await tester.pumpAndSettle();

      expect(find.textContaining('请连接手机 WiFi'), findsOneWidget);
      expect(find.textContaining('ESP32-ECG'), findsOneWidget);
      expect(find.textContaining('12345678'), findsOneWidget);
      // 刷新按钮
      expect(find.text('刷新'), findsOneWidget);
    });

    testWidgets('记录列表渲染 id 与时长', (tester) async {
      final api = _fakeApi();
      await tester.pumpWidget(MaterialApp(home: RecordListPage(api: api)));
      await tester.pumpAndSettle();

      // 记录 1
      expect(find.textContaining('#1'), findsOneWidget);
      expect(find.textContaining('60s'), findsOneWidget);
      // 记录 2
      expect(find.textContaining('#2'), findsOneWidget);
      expect(find.textContaining('30s'), findsOneWidget);
    });

    testWidgets('异常记录显示异常徽章', (tester) async {
      final api = _fakeApi();
      await tester.pumpWidget(MaterialApp(home: RecordListPage(api: api)));
      await tester.pumpAndSettle();

      // 记录 1 有 3 异常秒
      expect(find.text('3'), findsOneWidget); // 异常徽章数字
    });

    testWidgets('空记录列表显示暂且无记录提示', (tester) async {
      final api = _fakeApi(records: []);
      await tester.pumpWidget(MaterialApp(home: RecordListPage(api: api)));
      await tester.pumpAndSettle();

      expect(find.text('暂无记录'), findsOneWidget);
    });

    testWidgets('下载按钮点击后触发下载流程', (tester) async {
      final api = _fakeApi();
      await tester.pumpWidget(MaterialApp(home: RecordListPage(api: api)));
      await tester.pumpAndSettle();

      // 下载按钮可见
      final downloadBtn = find.byIcon(Icons.cloud_download);
      expect(downloadBtn, findsWidgets);

      // 点击下载
      await tester.tap(downloadBtn.first);
      await tester.pump();

      // 进入下载状态：显示 CircularProgressIndicator
      expect(find.byType(CircularProgressIndicator), findsOneWidget);
    });

    test('下载记录字节写入文件', () async {
      final tmpDir = Directory.systemTemp.createTempSync('ecg_test_');
      try {
        final testBytes = Uint8List.fromList([10, 20, 30, 40, 50]);
        final api = _fakeApi(downloadBytes: testBytes);

        // 直接通过 API 下载并写入文件
        final data = await api.downloadData(1);
        expect(data, testBytes);

        final recordFile = File('${tmpDir.path}/1.ecgr');
        await recordFile.writeAsBytes(data);
        expect(recordFile.existsSync(), true);
        expect(recordFile.readAsBytesSync(), testBytes);
      } finally {
        try {
          tmpDir.deleteSync(recursive: true);
        } catch (_) {}
      }
    });

    testWidgets('删除按钮调用 API 后列表刷新', (tester) async {
      final api = _fakeApi(deleteSuccess: true);
      await tester.pumpWidget(MaterialApp(home: RecordListPage(api: api)));
      await tester.pumpAndSettle();

      // 初始 2 条记录
      expect(find.textContaining('#1'), findsOneWidget);
      expect(find.textContaining('#2'), findsOneWidget);

      // 点击第一条记录的删除按钮
      final deleteBtns = find.byIcon(Icons.delete_outline);
      expect(deleteBtns, findsWidgets);
      await tester.tap(deleteBtns.first);
      await tester.pumpAndSettle();

      // 应显示删除成功提示
      expect(find.textContaining('已删除'), findsOneWidget);
    });
  });

  group('loadEcgrFile（默认加载器，真实文件 IO）', () {
    test('文件不存在 → null', () async {
      final tmpDir = Directory.systemTemp.createTempSync('ecg_test_');
      addTearDown(() {
        try {
          tmpDir.deleteSync(recursive: true);
        } catch (_) {}
      });

      expect(await loadEcgrFile('${tmpDir.path}/missing.ecgr'), isNull);
    });

    test('合法 .ecgr → 解码成功', () async {
      final tmpDir = Directory.systemTemp.createTempSync('ecg_test_');
      addTearDown(() {
        try {
          tmpDir.deleteSync(recursive: true);
        } catch (_) {}
      });

      final path = '${tmpDir.path}/1.ecgr';
      File(path).writeAsBytesSync(_buildFixture());

      final record = await loadEcgrFile(path);
      expect(record, isNotNull);
      expect(record!.sampleRate, 250);
      expect(record.totalSamples, 250);
    });

    test('损坏 .ecgr（坏魔数）→ null', () async {
      final tmpDir = Directory.systemTemp.createTempSync('ecg_test_');
      addTearDown(() {
        try {
          tmpDir.deleteSync(recursive: true);
        } catch (_) {}
      });

      final corrupt = _buildFixture();
      corrupt[0] = 0x58; // 'X'
      final path = '${tmpDir.path}/bad.ecgr';
      File(path).writeAsBytesSync(corrupt);

      expect(await loadEcgrFile(path), isNull);
    });
  });

  group('本地回放', () {
    // 回放路径统一注入假加载器（真实文件 IO 在 FakeAsync 区域无法完成，
    // 已由 loadEcgrFile 普通 test 覆盖）。
    Future<EcgRecord?> Function(int) _fakeLoader() {
      final record = EcgRecordCodec.decode(_buildFixture());
      return (_) async => record;
    }

    Future<void> _settleRoute(WidgetTester tester) async {
      // 路由转场动画用固定时长推进
      await tester.pump(const Duration(milliseconds: 400));
      await tester.pump(const Duration(milliseconds: 400));
    }

    /// 下载含真实文件写入：Windows 文件 IO 分多步送达事件循环，
    /// 交替 runAsync（放行真实事件循环）/ pump（冲刷 fake 微任务）3 跳。
    Future<void> _completeDownload(WidgetTester tester) async {
      for (var i = 0; i < 3; i++) {
        await tester.runAsync(
            () => Future<void>.delayed(const Duration(milliseconds: 50)));
        await tester.pump();
      }
    }

    testWidgets('加载器返回 null 时提示先下载', (tester) async {
      final api = _fakeApi();
      await tester.pumpWidget(MaterialApp(
          home: RecordListPage(api: api, ecgrLoader: (_) async => null)));
      await tester.pumpAndSettle();

      await tester.tap(find.byIcon(Icons.play_circle_outline).first);
      await tester.pumpAndSettle();

      expect(find.text('无法回放（请先下载记录文件）'), findsOneWidget);
    });

    testWidgets('下载成功后 SnackBar 动作「回放」跳转回放页', (tester) async {
      final api = _fakeApi(downloadBytes: _buildFixture());
      final tmpDir = Directory.systemTemp.createTempSync('ecg_test_');
      addTearDown(() {
        try {
          tmpDir.deleteSync(recursive: true);
        } catch (_) {}
      });

      await tester.pumpWidget(MaterialApp(
          home: RecordListPage(
              api: api, downloadDir: tmpDir, ecgrLoader: _fakeLoader())));
      await tester.pumpAndSettle();

      // 下载第一条记录（含真实文件写入）
      await tester.tap(find.byIcon(Icons.cloud_download).first);
      await _completeDownload(tester);
      await tester.pumpAndSettle();

      // 下载成功提示 + 回放动作
      expect(find.textContaining('下载成功'), findsOneWidget);
      expect(find.text('回放'), findsOneWidget);

      // 点动作跳转回放页（假加载器 → PlaybackPage）
      await tester.tap(find.text('回放'));
      await tester.pump();
      await _settleRoute(tester);

      expect(find.text('记录回放'), findsOneWidget);
      expect(find.byType(Slider), findsOneWidget);
    });

    testWidgets('回放按钮直接跳转回放页', (tester) async {
      final api = _fakeApi();
      await tester.pumpWidget(MaterialApp(
          home: RecordListPage(api: api, ecgrLoader: _fakeLoader())));
      await tester.pumpAndSettle();

      await tester.tap(find.byIcon(Icons.play_circle_outline).first);
      await tester.pump();
      await _settleRoute(tester);

      expect(find.text('记录回放'), findsOneWidget);
      expect(find.byType(Slider), findsOneWidget);
    });
  });
}
