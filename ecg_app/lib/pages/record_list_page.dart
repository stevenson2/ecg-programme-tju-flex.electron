import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import '../config/app_config.dart';
import '../services/record_api.dart';
import '../services/ecg_record_codec.dart';
import '../services/upload_service.dart';
import '../services/upload_queue.dart';
import 'playback_page.dart';

/**
 * @brief 默认 .ecgr 加载器：读取文件并解码（顶层函数，可独立单测）
 *
 * 返回 null 表示文件不存在或解码失败（不抛异常，文档化约定与
 * EcgRecordCodec.decode 一致）。真实文件 IO 在 widget 测试的
 * FakeAsync 区域无法完成，故页面提供 ecgrLoader 注入点，
 * 本函数用普通 test() 覆盖。
 */
Future<EcgRecord?> loadEcgrFile(String path) async {
  final file = File(path);
  if (!await file.exists()) return null;
  final bytes = await file.readAsBytes();
  return EcgRecordCodec.decode(bytes);
}

/**
 * @file record_list_page.dart
 * @brief 记录列表页面：AP 连接引导 + 记录下载/上传/管理
 *
 * 功能：
 *   1. AP 连接引导横幅（中文，含热点名与密码）
 *   2. 队列状态栏（待上传计数 + 一键处理）
 *   3. 记录列表（ID / 时长 / 大小 / 异常徽章 + 上传状态）
 *   4. 逐条下载（保存至 downloadDir/ecg_records/<id>.ecgr）
 *   5. 本地回放（卡片回放按钮 / 下载成功 SnackBar 动作 → PlaybackPage）
 *   6. 逐条删除（调用 DELETE API 后刷新列表）
 *   7. 逐条上传（云端 upload → analyze → report）
 *   8. 空状态提示
 *
 * 可注入 RecordApi / CloudUploadService / UploadQueue / Directory downloadDir /
 * ecgrLoader（回放加载器，默认 loadEcgrFile 真实文件 IO）以支持测试。
 */
class RecordListPage extends StatefulWidget {
  /** HTTP 客户端（测试中可注入 MockClient 构造的 RecordApi） */
  final RecordApi api;

  /** 下载目标目录（测试中可注入临时目录；null 时使用应用文档目录） */
  final Directory? downloadDir;

  /** 云端上传服务（null 时创建默认实例） */
  final CloudUploadService? uploadService;

  /** 上传队列（null 时从 downloadDir 自动创建） */
  final UploadQueue? uploadQueue;

  /** 本地回放加载器（null 时用 loadEcgrFile 真实文件 IO；测试注入假实现） */
  final Future<EcgRecord?> Function(int id)? ecgrLoader;

  const RecordListPage({
    super.key,
    required this.api,
    this.downloadDir,
    this.uploadService,
    this.uploadQueue,
    this.ecgrLoader,
  });

  @override
  State<RecordListPage> createState() => _RecordListPageState();
}

class _RecordListPageState extends State<RecordListPage> {
  List<RecordInfo>? _records;
  bool _loading = false;
  String? _error;
  final Set<int> _downloadingIds = {};
  final Set<int> _uploadingIds = {};

  late final CloudUploadService _uploadService;
  late final UploadQueue _uploadQueue;
  bool _queueReady = false;
  String _downloadDirPath = '';

  @override
  void initState() {
    super.initState();
    _initQueue();
    _refresh();
  }

  /** 初始化上传队列（注入或创建默认） */
  Future<void> _initQueue() async {
    _uploadService = widget.uploadService ?? CloudUploadService();

    final dir = await _getDownloadDir();
    _downloadDirPath = dir.path;

    if (widget.uploadQueue != null) {
      _uploadQueue = widget.uploadQueue!;
    } else {
      _uploadQueue = UploadQueue(
        service: _uploadService,
        pendingDir: dir,
      );
    }

    await _uploadQueue.load();
    if (mounted) {
      setState(() => _queueReady = true);
      // 自动处理队列中 pending 项
      _processQueue();
    }
  }

  /** 处理上传队列 */
  Future<void> _processQueue() async {
    await _uploadQueue.processAll();
    if (mounted) {
      setState(() {});
    }
  }

  /** 刷新记录列表 */
  Future<void> _refresh() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final records = await widget.api.listRecords();
      if (mounted) {
        setState(() {
          _records = records;
          _loading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = e.toString();
          _loading = false;
        });
      }
    }
  }

  /** 获取下载目录（优先注入的测试目录，否则走 path_provider） */
  Future<Directory> _getDownloadDir() async {
    if (widget.downloadDir != null) {
      return widget.downloadDir!;
    }
    final appDir = await getApplicationDocumentsDirectory();
    final dir = Directory('${appDir.path}/ecg_records');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir;
  }

  /** 获取已下载记录的文件路径（同步，依赖 _downloadDirPath 已初始化） */
  String _getEcgrPathSync(int id) {
    return '$_downloadDirPath/$id.ecgr';
  }

  /** 删除记录 */
  Future<void> _deleteRecord(int id) async {
    try {
      await widget.api.deleteRecord(id);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
              content: Text('已删除'), duration: Duration(seconds: 2)),
        );
        _refresh();
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('删除失败: $e'), backgroundColor: Colors.red),
        );
      }
    }
  }

  /** 下载记录数据并保存到文件 */
  Future<void> _downloadRecord(RecordInfo info) async {
    if (_downloadingIds.contains(info.id)) return;

    setState(() => _downloadingIds.add(info.id));

    try {
      final data = await widget.api.downloadData(info.id);
      final dir = await _getDownloadDir();
      final file = File('${dir.path}/${info.id}.ecgr');
      await file.writeAsBytes(data);

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('下载成功: ${file.path}'),
            duration: const Duration(seconds: 3),
            action: SnackBarAction(label: '回放', onPressed: () => _openPlayback(info)),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('下载失败: $e'), backgroundColor: Colors.red),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _downloadingIds.remove(info.id));
      }
    }
  }

  /** 本地回放：经加载器取 .ecgr 解码结果并跳转回放页 */
  Future<void> _openPlayback(RecordInfo info) async {
    final loader = widget.ecgrLoader ??
        (int id) => loadEcgrFile(_getEcgrPathSync(id));
    final record = await loader(info.id);
    if (!mounted) return;

    if (record == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('无法回放（请先下载记录文件）'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => PlaybackPage(record: record),
      ),
    );
  }

  /** 上传记录到云端 */
  Future<void> _uploadRecord(RecordInfo info) async {
    if (_uploadingIds.contains(info.id)) return;

    final ecgrPath = _getEcgrPathSync(info.id);
    final file = File(ecgrPath);

    if (!await file.exists()) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('请先下载记录文件再上传'),
            backgroundColor: Colors.orange,
          ),
        );
      }
      return;
    }

    setState(() => _uploadingIds.add(info.id));

    try {
      final bytes = await file.readAsBytes();
      final record = EcgRecordCodec.decode(bytes);
      if (record == null) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('文件解码失败，无法上传'),
              backgroundColor: Colors.red,
            ),
          );
        }
        return;
      }

      await _uploadQueue.enqueue(ecgrPath, record);
      await _processQueue();

      if (mounted) {
        final status = _uploadQueue.statusOf(ecgrPath);
        final msg = status == QueueStatus.done ? '上传成功' : '上传失败，已加入队列';
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(msg),
            duration: const Duration(seconds: 2),
            backgroundColor: status == QueueStatus.done ? Colors.green : null,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('上传失败: $e'), backgroundColor: Colors.red),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _uploadingIds.remove(info.id));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D0D1A),
      appBar: AppBar(
        title: const Text('记录管理'),
        backgroundColor: const Color(0xFF1A1A2E),
        elevation: 0,
      ),
      body: Column(
        children: [
          _buildApGuideBanner(),
          if (_queueReady) _buildQueueBar(),
          Expanded(child: _buildBody()),
        ],
      ),
    );
  }

  /** AP 连接引导横幅 */
  Widget _buildApGuideBanner() {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.fromLTRB(8, 8, 8, 4),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
            color: const Color(0xFF00BFFF).withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.wifi, color: Color(0xFF00BFFF), size: 18),
              SizedBox(width: 6),
              Text(
                '连接指南',
                style: TextStyle(
                  color: Color(0xFF00BFFF),
                  fontWeight: FontWeight.w600,
                  fontSize: 14,
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            '请连接手机 WiFi 到热点 ESP32-ECG-XXXX（密码 ${AppConfig.apPassword}），然后返回本页刷新',
            style: TextStyle(color: Colors.white70, fontSize: 13),
          ),
          const SizedBox(height: 8),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: _loading ? null : _refresh,
              icon: const Icon(Icons.refresh, size: 16),
              label: const Text('刷新'),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF00BFFF),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 8),
              ),
            ),
          ),
        ],
      ),
    );
  }

  /** 上传队列状态栏 */
  Widget _buildQueueBar() {
    final pendingCount = _uploadQueue.pendingCount;

    return Container(
      width: double.infinity,
      margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: pendingCount > 0
              ? Colors.orange.withValues(alpha: 0.4)
              : const Color(0xFF00BFFF).withValues(alpha: 0.2),
        ),
      ),
      child: Row(
        children: [
          Icon(
            Icons.cloud_upload,
            color: pendingCount > 0 ? Colors.orange : const Color(0xFF00BFFF),
            size: 18,
          ),
          const SizedBox(width: 6),
          Text(
            pendingCount > 0 ? '上传队列: $pendingCount 条待处理' : '上传队列: 已全部完成',
            style: TextStyle(
              color: pendingCount > 0 ? Colors.orange : Colors.white54,
              fontSize: 13,
            ),
          ),
          const Spacer(),
          if (pendingCount > 0)
            GestureDetector(
              onTap: () {
                _processQueue().then((_) {
                  if (mounted) setState(() {});
                });
              },
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    '立即上传',
                    style: TextStyle(
                      color: Color(0xFF00BFFF),
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  SizedBox(width: 2),
                  Icon(Icons.arrow_forward, color: Color(0xFF00BFFF), size: 14),
                ],
              ),
            ),
        ],
      ),
    );
  }

  /** 主体：加载中 / 错误 / 空列表 / 记录列表 */
  Widget _buildBody() {
    if (_loading && _records == null) {
      return const Center(
        child: CircularProgressIndicator(color: Color(0xFF00BFFF)),
      );
    }

    if (_error != null && _records == null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.error_outline, color: Colors.red, size: 48),
              const SizedBox(height: 8),
              Text(_error!,
                  style: const TextStyle(color: Colors.redAccent),
                  textAlign: TextAlign.center),
              const SizedBox(height: 12),
              ElevatedButton(onPressed: _refresh, child: const Text('重试')),
            ],
          ),
        ),
      );
    }

    if (_records == null || _records!.isEmpty) {
      return const Center(
        child: Text(
          '暂无记录',
          style: TextStyle(color: Colors.white38, fontSize: 16),
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _refresh,
      child: ListView.builder(
        itemCount: _records!.length,
        padding: const EdgeInsets.symmetric(horizontal: 8),
        itemBuilder: (context, index) {
          final info = _records![index];
          return _buildRecordCard(info);
        },
      ),
    );
  }

  /** 单条记录卡片 */
  Widget _buildRecordCard(RecordInfo info) {
    final isDownloading = _downloadingIds.contains(info.id);
    final isUploading = _uploadingIds.contains(info.id);
    final sizeKB = (info.size / 1024).toStringAsFixed(1);

    return Card(
      color: const Color(0xFF1A1A2E),
      margin: const EdgeInsets.only(bottom: 6),
      child: Padding(
        padding: const EdgeInsets.all(10),
        child: Row(
          children: [
            // 左侧信息
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(
                        '#${info.id}',
                        style: const TextStyle(
                          color: Color(0xFF00BFFF),
                          fontWeight: FontWeight.w600,
                          fontSize: 15,
                        ),
                      ),
                      const SizedBox(width: 8),
                      if (info.abnormalSeconds > 0)
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: Colors.red.withValues(alpha: 0.2),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(
                            '${info.abnormalSeconds}',
                            style: const TextStyle(
                                color: Colors.redAccent, fontSize: 11),
                          ),
                        ),
                      const SizedBox(width: 6),
                      // 上传状态标识
                      if (_queueReady) _buildUploadStatus(info.id),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '${info.duration}s  |  ${sizeKB}KB',
                    style:
                        const TextStyle(color: Colors.white54, fontSize: 12),
                  ),
                ],
              ),
            ),
            // 右侧操作按钮
            if (isUploading)
              const SizedBox(
                width: 24,
                height: 24,
                child: CircularProgressIndicator(
                    strokeWidth: 2, color: Color(0xFF00BFFF)),
              )
            else
              _buildUploadButton(info),
            isDownloading
                ? const SizedBox(
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: Color(0xFF00BFFF)),
                  )
                : IconButton(
                    icon: const Icon(Icons.cloud_download, size: 20),
                    color: const Color(0xFF00BFFF),
                    tooltip: '下载',
                    onPressed: () => _downloadRecord(info),
                    constraints: const BoxConstraints(
                        minWidth: 36, minHeight: 36),
                    padding: EdgeInsets.zero,
                  ),
            IconButton(
              icon: const Icon(Icons.play_circle_outline, size: 20),
              color: const Color(0xFF00BFFF),
              tooltip: '本地回放',
              onPressed: () => _openPlayback(info),
              constraints:
                  const BoxConstraints(minWidth: 36, minHeight: 36),
              padding: EdgeInsets.zero,
            ),
            IconButton(
              icon: const Icon(Icons.delete_outline, size: 20),
              color: Colors.redAccent,
              tooltip: '删除',
              onPressed: () => _deleteRecord(info.id),
              constraints:
                  const BoxConstraints(minWidth: 36, minHeight: 36),
              padding: EdgeInsets.zero,
            ),
          ],
        ),
      ),
    );
  }

  /** 上传状态标识（inline 小图标 + 文字） */
  Widget _buildUploadStatus(int recordId) {
    final ecgrPath = _getEcgrPathSync(recordId);
    final status = _uploadQueue.statusOf(ecgrPath);

    switch (status) {
      case QueueStatus.pending:
        return const Icon(Icons.hourglass_empty, size: 14, color: Colors.grey);
      case QueueStatus.done:
        return const Icon(Icons.check_circle, size: 14, color: Colors.green);
      case QueueStatus.failed:
        return const Icon(Icons.error, size: 14, color: Colors.redAccent);
      case QueueStatus.uploading:
        return const SizedBox(
          width: 14,
          height: 14,
          child: CircularProgressIndicator(strokeWidth: 2),
        );
      case QueueStatus.unknown:
        return const SizedBox.shrink(); // 未入队，不显示
    }
  }

  /** 上传按钮（仅在队列就绪后显示） */
  Widget _buildUploadButton(RecordInfo info) {
    if (!_queueReady) {
      return const SizedBox(width: 36, height: 36);
    }

    return IconButton(
      icon: const Icon(Icons.cloud_upload, size: 20),
      color: const Color(0xFF00BFFF),
      tooltip: '上传到云端',
      onPressed: () => _uploadRecord(info),
      constraints: const BoxConstraints(minWidth: 36, minHeight: 36),
      padding: EdgeInsets.zero,
    );
  }

}
