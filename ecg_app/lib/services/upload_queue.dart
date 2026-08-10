import 'dart:convert';
import 'dart:io';

import 'upload_service.dart';
import 'ecg_record_codec.dart';

/**
 * @file upload_queue.dart
 * @brief 可恢复的上传队列（离线缓存 + 重试）
 *
 * 通过 JSON 文件持久化待上传项，支持：
 *   - enqueue：加入队列并持久化到 pendingDir/upload_queue.json
 *   - processAll：遍历 pending 项逐一上传；失败保持 pending（离线重试语义）
 *   - statusOf：按 ecgrPath 查询上传状态
 *   - load：从 JSON 文件恢复队列状态（用于 App 重启后）
 *
 * 持久化格式：pendingDir/upload_queue.json — JSON 数组，每项：
 *   { "ecgrPath": <string>, "recordId": <string|null>, "status": "pending"|"done"|"failed" }
 *
 * 原子写入：先写 .tmp 再 rename，避免写中断导致 JSON 损坏。
 */

/** 上传队列项状态 */
enum QueueStatus {
  pending,  /**< 等待上传 */
  uploading, /**< 上传中（内存状态，不持久化） */
  done,     /**< 上传成功 */
  failed,   /**< 上传失败（已持久化标记） */
  unknown,  /**< 未在队列中 */
}

/** 内部持久化条目 */
class _QueueEntry {
  final String ecgrPath;
  String? recordId;
  String status; // 'pending' | 'done' | 'failed'

  _QueueEntry({
    required this.ecgrPath,
    this.recordId,
    this.status = 'pending',
  });

  Map<String, dynamic> toJson() => {
        'ecgrPath': ecgrPath,
        'recordId': recordId,
        'status': status,
      };

  factory _QueueEntry.fromJson(Map<String, dynamic> json) => _QueueEntry(
        ecgrPath: (json['ecgrPath'] as String?) ?? '',
        recordId: json['recordId'] as String?,
        status: (json['status'] as String?) ?? 'pending',
      );
}

/**
 * @brief 可恢复上传队列
 *
 * pendingDir 可注入（测试中用临时目录替代应用文档目录）。
 * 队列持久化为 pendingDir/upload_queue.json。
 */
class UploadQueue {
  final CloudUploadService _service;
  final Directory _pendingDir;

  final List<_QueueEntry> _entries = [];
  bool _loaded = false;

  UploadQueue({
    required CloudUploadService service,
    required Directory pendingDir,
  })  : _service = service,
        _pendingDir = pendingDir;

  /** 从 JSON 文件加载队列状态（幂等：多次调用仅首次生效） */
  Future<void> load() async {
    if (_loaded) return;

    final jsonFile = File('${_pendingDir.path}/upload_queue.json');
    if (!await jsonFile.exists()) {
      _loaded = true;
      return;
    }

    try {
      final content = await jsonFile.readAsString();
      final list = jsonDecode(content) as List<dynamic>;
      _entries.clear();
      _entries.addAll(
        list
            .map((e) => _QueueEntry.fromJson(e as Map<String, dynamic>))
            .where((e) => e.ecgrPath.isNotEmpty),
      );
    } catch (_) {
      // JSON 损坏：重置为空队列（不崩溃）
      _entries.clear();
    }

    _loaded = true;
  }

  /**
   * @brief 入队一条待上传记录
   *
   * 立即持久化到 JSON 文件。同一 ecgrPath 重复入队不做去重
   * （允许重试语义）。调用方可通过 statusOf 判断是否已在队列。
   */
  Future<void> enqueue(String ecgrPath, EcgRecord meta) async {
    await load();

    final entry = _QueueEntry(ecgrPath: ecgrPath, status: 'pending');
    _entries.add(entry);
    await _persist();
  }

  /**
   * @brief 处理所有 pending 项：逐一上传
   *
   * 失败项保持 pending 状态（不抛异常，由 processAll 下次调用重试）。
   * 成功项标记为 done 并记录云端 recordId。
   */
  Future<void> processAll() async {
    await load();

    for (final entry in _entries) {
      if (entry.status != 'pending') continue;

      try {
        final file = File(entry.ecgrPath);
        if (!await file.exists()) {
          entry.status = 'failed';
          await _persist();
          continue;
        }

        final bytes = await file.readAsBytes();
        final record = EcgRecordCodec.decode(bytes);
        if (record == null) {
          entry.status = 'failed';
          await _persist();
          continue;
        }

        final result = await _service.uploadRecord(file, record);
        entry.recordId = result.recordId;
        entry.status = 'done';
        await _persist();
      } catch (_) {
        // 上传失败：保持 pending（离线重试语义），不修改状态
        // 仅持久化当前状态（可能已有其他条目状态变更）
        await _persist();
      }
    }
  }

  /**
   * @brief 查询指定 ecgrPath 的上传状态
   *
   * @return QueueStatus 枚举值（未在队列中返回 unknown）
   */
  QueueStatus statusOf(String ecgrPath) {
    final entry = _findEntry(ecgrPath);
    if (entry == null) return QueueStatus.unknown;

    switch (entry.status) {
      case 'pending':
        return QueueStatus.pending;
      case 'done':
        return QueueStatus.done;
      case 'failed':
        return QueueStatus.failed;
      default:
        return QueueStatus.unknown;
    }
  }

  /** 是否有待处理项 */
  bool get hasPending {
    return _entries.any((e) => e.status == 'pending');
  }

  /** 待处理项数量 */
  int get pendingCount {
    return _entries.where((e) => e.status == 'pending').length;
  }

  // ───────────────────── 内部方法 ─────────────────────

  _QueueEntry? _findEntry(String ecgrPath) {
    try {
      return _entries.firstWhere((e) => e.ecgrPath == ecgrPath);
    } catch (_) {
      return null;
    }
  }

  /** 原子写入 JSON 文件（先 .tmp 再 rename） */
  Future<void> _persist() async {
    final jsonFile = File('${_pendingDir.path}/upload_queue.json');
    final tmpFile = File('${_pendingDir.path}/upload_queue.json.tmp');

    final data = _entries.map((e) => e.toJson()).toList();
    final jsonStr = const JsonEncoder.withIndent('  ').convert(data);

    await tmpFile.writeAsString(jsonStr, flush: true);
    // 跨平台原子 rename（同名覆盖）
    if (await jsonFile.exists()) {
      await jsonFile.delete();
    }
    await tmpFile.rename(jsonFile.path);
  }
}
