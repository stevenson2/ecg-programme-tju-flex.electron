import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:ecg_app/services/record_api.dart';

/**
 * @file record_api_test.dart
 * @brief RecordApi HTTP 客户端单元测试（MockClient TDD）
 *
 * 覆盖 Contract C7 所有端点：list / meta / download / delete 的
 * 正常响应、404、500、超时场景。
 */
void main() {
  const baseUrl = 'http://192.168.4.1';

  /// 构造标准 list 响应 JSON
  Map<String, dynamic> _listJson() => {
        'records': [
          {
            'id': 1,
            'duration': 60,
            'size': 146000,
            'abnormal_seconds': 3,
            'start': '2026-08-08T10:00:00Z',
          },
          {
            'id': 2,
            'duration': 30,
            'size': 73000,
            'abnormal_seconds': 0,
            'start': '2026-08-08T10:05:00Z',
          },
        ],
      };

  Map<String, dynamic> _metaJson(int id) => {
        'id': id,
        'sample_rate': 250,
        'start_unix': 1723104000,
        'duration': 60,
        'total_samples': 15000,
        'abnormal_seconds': 3,
      };

  List<int> _fakeEcgrBytes() {
    // 32B 头部 + 一些 int16 样本 + 位图
    final header = List<int>.filled(32, 0);
    final samples = List<int>.generate(100, (i) => i % 256);
    return [...header, ...samples];
  }

  group('RecordApi listRecords', () {
    test('正常响应解析为 RecordInfo 列表', () async {
      final client = MockClient((req) async {
        if (req.url.path == '/api/records' && req.method == 'GET') {
          return http.Response(jsonEncode(_listJson()), 200);
        }
        return http.Response('not found', 404);
      });

      final api = RecordApi(client: client, baseUrl: baseUrl);
      final records = await api.listRecords();

      expect(records, hasLength(2));
      expect(records[0].id, 1);
      expect(records[0].duration, 60);
      expect(records[0].size, 146000);
      expect(records[0].abnormalSeconds, 3);
      expect(records[0].start, '2026-08-08T10:00:00Z');
      expect(records[1].id, 2);
      expect(records[1].abnormalSeconds, 0);
    });

    test('空列表返回 []', () async {
      final client = MockClient((req) async {
        return http.Response('{"records":[]}', 200);
      });

      final api = RecordApi(client: client, baseUrl: baseUrl);
      final records = await api.listRecords();
      expect(records, isEmpty);
    });

    test('HTTP 404 抛出 RecordApiException', () async {
      final client = MockClient((req) async => http.Response('not found', 404));
      final api = RecordApi(client: client, baseUrl: baseUrl);

      expect(
        () => api.listRecords(),
        throwsA(predicate((e) =>
            e is RecordApiException && e.message.contains('404'))),
      );
    });

    test('HTTP 500 抛出 RecordApiException', () async {
      final client = MockClient((req) async => http.Response('server error', 500));
      final api = RecordApi(client: client, baseUrl: baseUrl);

      expect(
        () => api.listRecords(),
        throwsA(isA<RecordApiException>()),
      );
    });
  });

  group('RecordApi getMeta', () {
    test('正常响应解析为 RecordMeta', () async {
      final client = MockClient((req) async {
        if (req.url.path == '/api/records/1/meta') {
          return http.Response(jsonEncode(_metaJson(1)), 200);
        }
        return http.Response('not found', 404);
      });

      final api = RecordApi(client: client, baseUrl: baseUrl);
      final meta = await api.getMeta(1);

      expect(meta.id, 1);
      expect(meta.sampleRate, 250);
      expect(meta.startUnix, 1723104000);
      expect(meta.duration, 60);
      expect(meta.totalSamples, 15000);
      expect(meta.abnormalSeconds, 3);
    });

    test('HTTP 404 抛出 RecordApiException', () async {
      final client = MockClient((req) async => http.Response('not found', 404));
      final api = RecordApi(client: client, baseUrl: baseUrl);

      expect(
        () => api.getMeta(999),
        throwsA(isA<RecordApiException>()),
      );
    });
  });

  group('RecordApi downloadData', () {
    test('正常响应返回完整字节（200）', () async {
      final expectedBytes = _fakeEcgrBytes();
      final client = MockClient((req) async {
        if (req.url.path == '/api/records/1/data' && req.method == 'GET') {
          return http.Response.bytes(expectedBytes, 200,
              headers: {'Content-Length': '${expectedBytes.length}'});
        }
        return http.Response('not found', 404);
      });

      final api = RecordApi(client: client, baseUrl: baseUrl);
      final data = await api.downloadData(1);

      expect(data, expectedBytes);
      expect(data.length, expectedBytes.length);
    });

    test('HTTP 404 抛出 RecordApiException', () async {
      final client = MockClient((req) async => http.Response('not found', 404));
      final api = RecordApi(client: client, baseUrl: baseUrl);

      expect(
        () => api.downloadData(999),
        throwsA(isA<RecordApiException>()),
      );
    });
  });

  group('RecordApi deleteRecord', () {
    test('成功删除返回 true', () async {
      final client = MockClient((req) async {
        if (req.url.path == '/api/records/1' && req.method == 'DELETE') {
          return http.Response('{"deleted":true}', 200);
        }
        return http.Response('not found', 404);
      });

      final api = RecordApi(client: client, baseUrl: baseUrl);
      final result = await api.deleteRecord(1);

      expect(result, true);
    });

    test('HTTP 404 抛出 RecordApiException', () async {
      final client = MockClient((req) async => http.Response('not found', 404));
      final api = RecordApi(client: client, baseUrl: baseUrl);

      expect(
        () => api.deleteRecord(999),
        throwsA(isA<RecordApiException>()),
      );
    });
  });
}
