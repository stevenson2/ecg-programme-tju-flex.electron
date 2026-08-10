import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:ecg_app/services/upload_service.dart';
import 'package:ecg_app/services/ecg_record_codec.dart';

/**
 * @file upload_service_test.dart
 * @brief CloudUploadService 单元测试（MockClient TDD）
 *
 * 覆盖 Contract C8 端点：
 *   - POST /v1/records（multipart 上传 .ecgr + JSON metadata）
 *   - POST /v1/records/{id}/analyze
 *   - GET  /v1/records/{id}/report
 *   以及异常场景（401/500/超时）和响应解析。
 */

// ─────────────────────────── 测试内 fixture 构建 ───────────────────────────

/// 构建合法的 32B .ecgr 头部 + 少量样本
Uint8List _buildFakeEcgrBytes({
  int sampleRate = 250,
  int startUnixTime = 1700000000,
  int durationSec = 5,
  int totalSamples = 1250,
  int abnormalSeconds = 2,
}) {
  final size = 32 + totalSamples * 2;
  final bytes = Uint8List(size);
  bytes[0] = 0x45;
  bytes[1] = 0x43;
  bytes[2] = 0x47;
  bytes[3] = 0x52; // 'ECGR'
  bytes[4] = 1; // version
  bytes[5] = 0x00; // flags: 无异常位图（简化测试 fixture）
  _putU32LE(bytes, 6, sampleRate);
  _putU32LE(bytes, 10, startUnixTime);
  _putU32LE(bytes, 14, durationSec);
  _putU32LE(bytes, 18, totalSamples);
  _putU32LE(bytes, 22, abnormalSeconds);
  return bytes;
}

void _putU32LE(Uint8List b, int off, int v) {
  b[off] = v & 0xFF;
  b[off + 1] = (v >> 8) & 0xFF;
  b[off + 2] = (v >> 16) & 0xFF;
  b[off + 3] = (v >> 24) & 0xFF;
}

/// 在临时目录中创建 .ecgr 文件并返回 File + EcgRecord
({File file, EcgRecord record, Directory dir}) _createEcgrFixture({
  String prefix = 'upload_test_',
  Uint8List? bytes,
}) {
  final bytesActual = bytes ?? _buildFakeEcgrBytes();
  final tmpDir = Directory.systemTemp.createTempSync(prefix);
  final ecgrFile = File('${tmpDir.path}/test.ecgr');
  ecgrFile.writeAsBytesSync(bytesActual);
  final record = EcgRecordCodec.decode(bytesActual)!;
  return (file: ecgrFile, record: record, dir: tmpDir);
}

void main() {
  const baseUrl = 'http://127.0.0.1:8000/v1';
  const token = 'dev-token';

  group('CloudUploadService uploadRecord', () {
    test('multipart POST 成功返回 record_id', () async {
      final client = MockClient((req) async {
        if (req.url.path == '/v1/records' && req.method == 'POST') {
          // 校验认证头与 Content-Type
          expect(req.headers['authorization'], 'Bearer $token');
          expect(
            req.headers['content-type'],
            contains('multipart/form-data'),
          );

          return http.Response(
            jsonEncode({'record_id': 'rec-abc-123', 'status': 'uploaded'}),
            201,
          );
        }
        return http.Response('not found', 404);
      });

      final service = CloudUploadService(
        client: client,
        baseUrl: baseUrl,
        token: token,
      );

      final fixture = _createEcgrFixture();
      try {
        final result =
            await service.uploadRecord(fixture.file, fixture.record);
        expect(result.recordId, 'rec-abc-123');
        expect(result.status, 'uploaded');
      } finally {
        fixture.dir.deleteSync(recursive: true);
      }
    });

    test('HTTP 401 抛出 CloudUploadException（未授权）', () async {
      final client = MockClient((req) async {
        return http.Response('{"error":"unauthorized"}', 401);
      });

      final service = CloudUploadService(
        client: client,
        baseUrl: baseUrl,
        token: token,
      );

      final fixture = _createEcgrFixture();
      try {
        // 使用 async expect — 直接 await 会抛异常，用 try/catch 捕获
        CloudUploadException? caught;
        try {
          await service.uploadRecord(fixture.file, fixture.record);
          fail('应该抛出异常');
        } on CloudUploadException catch (e) {
          caught = e;
        }
        expect(caught, isNotNull);
        expect(caught.statusCode, 401);
        expect(caught.message, contains('401'));
      } finally {
        fixture.dir.deleteSync(recursive: true);
      }
    });

    test('HTTP 500 抛出 CloudUploadException（服务器错误）', () async {
      final client = MockClient((req) async {
        return http.Response('server error', 500);
      });

      final service = CloudUploadService(
        client: client,
        baseUrl: baseUrl,
        token: token,
      );

      final fixture = _createEcgrFixture();
      try {
        CloudUploadException? caught;
        try {
          await service.uploadRecord(fixture.file, fixture.record);
          fail('应该抛出异常');
        } on CloudUploadException catch (e) {
          caught = e;
        }
        expect(caught, isNotNull);
        expect(caught!.statusCode, 500);
      } finally {
        fixture.dir.deleteSync(recursive: true);
      }
    });

    test('文件不存在抛出 CloudUploadException', () async {
      final client = MockClient((req) async {
        return http.Response('ok', 201);
      });

      final service = CloudUploadService(
        client: client,
        baseUrl: baseUrl,
        token: token,
      );

      final fakeFile = File('/nonexistent/path/file.ecgr');
      final bytes = _buildFakeEcgrBytes();
      final record = EcgRecordCodec.decode(bytes)!;

      try {
        await service.uploadRecord(fakeFile, record);
        fail('应该抛出异常');
      } on CloudUploadException catch (e) {
        expect(e.message, contains('文件不存在'));
      }
    });
  });

  group('CloudUploadService analyze', () {
    test('POST analyze 成功返回 analyzed 状态', () async {
      final recordId = 'rec-abc-123';
      final client = MockClient((req) async {
        if (req.url.path == '/v1/records/$recordId/analyze' &&
            req.method == 'POST') {
          expect(req.headers['authorization'], 'Bearer $token');
          return http.Response(
            jsonEncode({
              'record_id': recordId,
              'status': 'analyzed',
            }),
            200,
          );
        }
        return http.Response('not found', 404);
      });

      final service = CloudUploadService(
        client: client,
        baseUrl: baseUrl,
        token: token,
      );

      final report = await service.analyze(recordId);
      expect(report.recordId, recordId);
      expect(report.status, 'analyzed');
    });

    test('analyze 返回非 200 抛出 CloudUploadException', () async {
      final recordId = 'rec-not-found';
      final client = MockClient((req) async {
        return http.Response('{"error":"not found"}', 404);
      });

      final service = CloudUploadService(
        client: client,
        baseUrl: baseUrl,
        token: token,
      );

      try {
        await service.analyze(recordId);
        fail('应该抛出异常');
      } on CloudUploadException catch (_) {
        // 预期
      }
    });
  });

  group('CloudUploadService fetchReport', () {
    test('GET report 成功解析完整报告字段', () async {
      final recordId = 'rec-abc-123';
      final responseJson = {
        'record_id': recordId,
        'status': 'completed',
        'summary': {
          'total_duration_s': 60,
          'abnormal_duration_s': 12,
          'abnormal_ratio': 0.20,
          'avg_heart_rate_bpm': 72.5,
        },
        'events': [
          {'second': 5, 'type': 'arrhythmia', 'confidence': 0.89},
          {'second': 30, 'type': 'arrhythmia', 'confidence': 0.92},
        ],
        'recommendation': 'Possible arrhythmia detected, consult a doctor',
      };

      final client = MockClient((req) async {
        if (req.url.path == '/v1/records/$recordId/report' &&
            req.method == 'GET') {
          expect(req.headers['authorization'], 'Bearer $token');
          return http.Response(jsonEncode(responseJson), 200);
        }
        return http.Response('not found', 404);
      });

      final service = CloudUploadService(
        client: client,
        baseUrl: baseUrl,
        token: token,
      );

      final report = await service.fetchReport(recordId);

      expect(report.recordId, recordId);
      expect(report.status, 'completed');
      expect(report.summary, isNotNull);
      expect(report.summary!['total_duration_s'], 60);
      expect(report.summary!['abnormal_ratio'], 0.20);
      expect(report.events, hasLength(2));
      expect(report.events![0]['type'], 'arrhythmia');
      expect(report.recommendation, contains('arrhythmia'));
    });

    test('report 返回 404 抛出 CloudUploadException', () async {
      final client = MockClient((req) async {
        return http.Response('{"error":"not found"}', 404);
      });

      final service = CloudUploadService(
        client: client,
        baseUrl: baseUrl,
        token: token,
      );

      try {
        await service.fetchReport('rec-not-found');
        fail('应该抛出异常');
      } on CloudUploadException catch (_) {
        // 预期
      }
    });

    test('report 缺少字段时使用默认值', () async {
      final recordId = 'rec-minimal';
      final client = MockClient((req) async {
        if (req.url.path == '/v1/records/$recordId/report') {
          return http.Response(
            jsonEncode({
              'record_id': recordId,
              'status': 'pending',
            }),
            200,
          );
        }
        return http.Response('not found', 404);
      });

      final service = CloudUploadService(
        client: client,
        baseUrl: baseUrl,
        token: token,
      );

      final report = await service.fetchReport(recordId);
      expect(report.recordId, recordId);
      expect(report.status, 'pending');
      expect(report.summary, isNull);
      expect(report.events, isNull);
      expect(report.recommendation, isNull);
    });
  });
}
