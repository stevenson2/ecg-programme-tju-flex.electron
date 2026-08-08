import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:ecg_app/services/record_api.dart';
import 'package:ecg_app/pages/record_list_page.dart';

/**
 * @file record_list_page_test.dart
 * @brief RecordListPage 组件测试（注入假 RecordApi + 临时目录）
 *
 * 覆盖：AP 引导横幅、记录列表渲染、下载写文件、删除刷新、空状态。
 */
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
}
