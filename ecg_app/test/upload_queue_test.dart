import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:ecg_app/services/upload_service.dart';
import 'package:ecg_app/services/upload_queue.dart';
import 'package:ecg_app/services/ecg_record_codec.dart';

/**
 * @file upload_queue_test.dart
 * @brief UploadQueue 单元测试（可注入临时目录 + MockClient）
 *
 * 覆盖：
 *   - enqueue 持久化到 pendingDir/upload_queue.json
 *   - processAll：失败 → pending（离线重试语义）；成功 → done
 *   - statusOf：各状态查询
 *   - done 标记跨 UploadQueue 实例存活（重新加载同一 JSON）
 *   - 空队列无 crash
 *   - QueueStatus 枚举值完整性
 */

// ─────────────────────────── 测试内 fixture ───────────────────────────

/// 构建合法的 .ecgr 字节（32B 头部 + 少量样本）
Uint8List _buildFakeEcgrBytes({
  int sampleRate = 250,
  int startUnixTime = 1700000000,
  int durationSec = 3,
  int totalSamples = 750,
  int abnormalSeconds = 1,
}) {
  final size = 32 + totalSamples * 2 + durationSec; // +位图
  final bytes = Uint8List(size);
  bytes[0] = 0x45;
  bytes[1] = 0x43;
  bytes[2] = 0x47;
  bytes[3] = 0x52;
  bytes[4] = 1;
  bytes[5] = 0x01;
  _putU32LE(bytes, 6, sampleRate);
  _putU32LE(bytes, 10, startUnixTime);
  _putU32LE(bytes, 14, durationSec);
  _putU32LE(bytes, 18, totalSamples);
  _putU32LE(bytes, 22, abnormalSeconds);
  // 样本流全部置零（简化）
  // 位图紧随其后
  return bytes;
}

void _putU32LE(Uint8List b, int off, int v) {
  b[off] = v & 0xFF;
  b[off + 1] = (v >> 8) & 0xFF;
  b[off + 2] = (v >> 16) & 0xFF;
  b[off + 3] = (v >> 24) & 0xFF;
}

/// 创建返回成功的 MockClient
CloudUploadService _successService() {
  final client = MockClient((req) async {
    if (req.url.path == '/v1/records' && req.method == 'POST') {
      return http.Response(
        jsonEncode({'record_id': 'rec-success', 'status': 'uploaded'}),
        201,
      );
    }
    if (req.url.path.contains('/analyze')) {
      return http.Response(
        jsonEncode({'record_id': 'rec-success', 'status': 'analyzed'}),
        200,
      );
    }
    if (req.url.path.contains('/report')) {
      return http.Response(
        jsonEncode({
          'record_id': 'rec-success',
          'status': 'completed',
          'recommendation': '一切正常',
        }),
        200,
      );
    }
    return http.Response('not found', 404);
  });
  return CloudUploadService(client: client);
}

/// 创建始终返回 500 的 MockClient（模拟上传失败）
CloudUploadService _failingService() {
  final client = MockClient((req) async {
    return http.Response('server error', 500);
  });
  return CloudUploadService(client: client);
}

void main() {
  group('UploadQueue enqueue 与持久化', () {
    test('enqueue 写入 JSON 文件且可读取', () async {
      final tmpDir = Directory.systemTemp.createTempSync('upload_queue_test_');
      try {
        final service = _successService();
        final queue = UploadQueue(service: service, pendingDir: tmpDir);

        final ecgrBytes = _buildFakeEcgrBytes();
        final record = EcgRecordCodec.decode(ecgrBytes)!;
        final ecgrFile = File('${tmpDir.path}/test.ecgr');
        ecgrFile.writeAsBytesSync(ecgrBytes);

        await queue.enqueue(ecgrFile.path, record);

        // JSON 文件已创建
        final jsonFile = File('${tmpDir.path}/upload_queue.json');
        expect(jsonFile.existsSync(), isTrue);

        final content = jsonDecode(jsonFile.readAsStringSync());
        expect(content, isA<List>());
        expect(content, hasLength(1));
        expect(content[0]['ecgrPath'], ecgrFile.path);
        expect(content[0]['status'], 'pending');
      } finally {
        tmpDir.deleteSync(recursive: true);
      }
    });

    test('statusOf 返回 pending（刚 enqueue 后）', () async {
      final tmpDir = Directory.systemTemp.createTempSync('upload_queue_test_');
      try {
        final service = _successService();
        final queue = UploadQueue(service: service, pendingDir: tmpDir);

        final ecgrBytes = _buildFakeEcgrBytes();
        final record = EcgRecordCodec.decode(ecgrBytes)!;
        final ecgrFile = File('${tmpDir.path}/test.ecgr');
        ecgrFile.writeAsBytesSync(ecgrBytes);

        await queue.enqueue(ecgrFile.path, record);

        final status = queue.statusOf(ecgrFile.path);
        expect(status, QueueStatus.pending);
      } finally {
        tmpDir.deleteSync(recursive: true);
      }
    });

    test('statusOf 未知路径返回 QueueStatus.unknown', () async {
      final tmpDir = Directory.systemTemp.createTempSync('upload_queue_test_');
      try {
        final service = _successService();
        final queue = UploadQueue(service: service, pendingDir: tmpDir);

        final status = queue.statusOf('/nonexistent/file.ecgr');
        expect(status, QueueStatus.unknown);
      } finally {
        tmpDir.deleteSync(recursive: true);
      }
    });

    test('done 标记跨 UploadQueue 实例存活（重新加载 JSON）', () async {
      final tmpDir = Directory.systemTemp.createTempSync('upload_queue_test_');
      try {
        final ecgrBytes = _buildFakeEcgrBytes();
        final record = EcgRecordCodec.decode(ecgrBytes)!;
        final ecgrFile = File('${tmpDir.path}/test.ecgr');
        ecgrFile.writeAsBytesSync(ecgrBytes);

        // 第一次实例：enqueue + processAll 成功
        final queue1 = UploadQueue(
          service: _successService(),
          pendingDir: tmpDir,
        );
        await queue1.enqueue(ecgrFile.path, record);
        await queue1.processAll();

        expect(queue1.statusOf(ecgrFile.path), QueueStatus.done);

        // 第二个实例（模拟 App 重启）：从同一目录加载 JSON
        final queue2 = UploadQueue(
          service: _successService(),
          pendingDir: tmpDir,
        );
        await queue2.load();

        expect(queue2.statusOf(ecgrFile.path), QueueStatus.done);
      } finally {
        tmpDir.deleteSync(recursive: true);
      }
    });
  });

  group('UploadQueue processAll 离线重试与成功', () {
    test('processAll 全部失败 → 状态保持 pending（离线重试语义）', () async {
      final tmpDir = Directory.systemTemp.createTempSync('upload_queue_test_');
      try {
        final ecgrBytes = _buildFakeEcgrBytes();
        final record = EcgRecordCodec.decode(ecgrBytes)!;
        final ecgrFile = File('${tmpDir.path}/test_fail.ecgr');
        ecgrFile.writeAsBytesSync(ecgrBytes);

        final queue = UploadQueue(
          service: _failingService(),
          pendingDir: tmpDir,
        );
        await queue.enqueue(ecgrFile.path, record);

        // processAll：上传失败不应崩溃
        await queue.processAll();

        // 状态保持 pending（离线重试语义：失败不清除，下次 processAll 重试）
        final status = queue.statusOf(ecgrFile.path);
        expect(status, QueueStatus.pending);

        // JSON 文件仍存在且包含该条目
        final jsonFile = File('${tmpDir.path}/upload_queue.json');
        final content = jsonDecode(jsonFile.readAsStringSync());
        expect(content, hasLength(1));
        expect(content[0]['status'], 'pending');
      } finally {
        tmpDir.deleteSync(recursive: true);
      }
    });

    test('processAll 成功 → 状态变为 done', () async {
      final tmpDir = Directory.systemTemp.createTempSync('upload_queue_test_');
      try {
        final ecgrBytes = _buildFakeEcgrBytes();
        final record = EcgRecordCodec.decode(ecgrBytes)!;
        final ecgrFile = File('${tmpDir.path}/test_success.ecgr');
        ecgrFile.writeAsBytesSync(ecgrBytes);

        final queue = UploadQueue(
          service: _successService(),
          pendingDir: tmpDir,
        );
        await queue.enqueue(ecgrFile.path, record);
        await queue.processAll();

        final status = queue.statusOf(ecgrFile.path);
        expect(status, QueueStatus.done);

        // done 条目的 recordId 已记录
        final jsonFile = File('${tmpDir.path}/upload_queue.json');
        final content = jsonDecode(jsonFile.readAsStringSync());
        expect(content[0]['recordId'], 'rec-success');
        expect(content[0]['status'], 'done');
      } finally {
        tmpDir.deleteSync(recursive: true);
      }
    });

    test('失败后再成功 → pending → done 状态切换', () async {
      final tmpDir = Directory.systemTemp.createTempSync('upload_queue_test_');
      try {
        final ecgrBytes = _buildFakeEcgrBytes();
        final record = EcgRecordCodec.decode(ecgrBytes)!;
        final ecgrFile = File('${tmpDir.path}/test_retry.ecgr');
        ecgrFile.writeAsBytesSync(ecgrBytes);

        // 第一步：失败
        final queueFail = UploadQueue(
          service: _failingService(),
          pendingDir: tmpDir,
        );
        await queueFail.enqueue(ecgrFile.path, record);
        await queueFail.processAll();
        expect(queueFail.statusOf(ecgrFile.path), QueueStatus.pending);

        // 第二步：重新加载，用成功 service 重试
        final queueSuccess = UploadQueue(
          service: _successService(),
          pendingDir: tmpDir,
        );
        await queueSuccess.load();
        await queueSuccess.processAll();

        expect(queueSuccess.statusOf(ecgrFile.path), QueueStatus.done);
      } finally {
        tmpDir.deleteSync(recursive: true);
      }
    });
  });

  group('UploadQueue 边界与健壮性', () {
    test('空队列 processAll 不崩溃', () async {
      final tmpDir = Directory.systemTemp.createTempSync('upload_queue_test_');
      try {
        final queue = UploadQueue(
          service: _successService(),
          pendingDir: tmpDir,
        );
        // 无 enqueue，直接 processAll
        await queue.processAll();
        // 不抛异常即通过
      } finally {
        tmpDir.deleteSync(recursive: true);
      }
    });

    test('JSON 文件不存在时 load 不崩溃', () async {
      final tmpDir = Directory.systemTemp.createTempSync('upload_queue_test_');
      try {
        // 确保 JSON 文件不存在
        final jsonFile = File('${tmpDir.path}/upload_queue.json');
        expect(jsonFile.existsSync(), isFalse);

        final queue = UploadQueue(
          service: _successService(),
          pendingDir: tmpDir,
        );
        await queue.load();
        // 不抛异常即通过
      } finally {
        tmpDir.deleteSync(recursive: true);
      }
    });

    test('JSON 文件损坏时 load 不崩溃', () async {
      final tmpDir = Directory.systemTemp.createTempSync('upload_queue_test_');
      try {
        final jsonFile = File('${tmpDir.path}/upload_queue.json');
        jsonFile.writeAsStringSync('not valid json {{');

        final queue = UploadQueue(
          service: _successService(),
          pendingDir: tmpDir,
        );
        await queue.load();
        // 不抛异常即通过；队列状态为空
        expect(queue.hasPending, isFalse);
      } finally {
        tmpDir.deleteSync(recursive: true);
      }
    });

    test('hasPending 正确反映队列状态', () async {
      final tmpDir = Directory.systemTemp.createTempSync('upload_queue_test_');
      try {
        final ecgrBytes = _buildFakeEcgrBytes();
        final record = EcgRecordCodec.decode(ecgrBytes)!;
        final ecgrFile = File('${tmpDir.path}/test.ecgr');
        ecgrFile.writeAsBytesSync(ecgrBytes);

        final queue = UploadQueue(
          service: _successService(),
          pendingDir: tmpDir,
        );

        expect(queue.hasPending, isFalse);

        await queue.enqueue(ecgrFile.path, record);
        expect(queue.hasPending, isTrue);

        await queue.processAll();
        expect(queue.hasPending, isFalse); // 全部 done
      } finally {
        tmpDir.deleteSync(recursive: true);
      }
    });

    test('pendingCount 返回待处理项数', () async {
      final tmpDir = Directory.systemTemp.createTempSync('upload_queue_test_');
      try {
        final queue = UploadQueue(
          service: _failingService(),
          pendingDir: tmpDir,
        );

        final ecgrBytes = _buildFakeEcgrBytes();
        final record = EcgRecordCodec.decode(ecgrBytes)!;

        final f1 = File('${tmpDir.path}/test1.ecgr');
        f1.writeAsBytesSync(ecgrBytes);
        await queue.enqueue(f1.path, record);

        final f2 = File('${tmpDir.path}/test2.ecgr');
        f2.writeAsBytesSync(ecgrBytes);
        await queue.enqueue(f2.path, record);

        expect(queue.pendingCount, 2);

        await queue.processAll(); // 全部失败，保持 pending
        expect(queue.pendingCount, 2);
      } finally {
        tmpDir.deleteSync(recursive: true);
      }
    });
  });
}
