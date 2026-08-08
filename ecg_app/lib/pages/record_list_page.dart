import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import '../services/record_api.dart';

/**
 * @file record_list_page.dart
 * @brief 记录列表页面：AP 连接引导 + 记录下载与管理
 *
 * 功能：
 *   1. AP 连接引导横幅（中文，含热点名与密码）
 *   2. 记录列表（ID / 时长 / 大小 / 异常徽章）
 *   3. 逐条下载（保存至 downloadDir/ecg_records/<id>.ecgr）
 *   4. 逐条删除（调用 DELETE API 后刷新列表）
 *   5. 空状态提示
 *
 * 可注入 RecordApi + Directory downloadDir 以支持测试。
 */

class RecordListPage extends StatefulWidget {
  /** HTTP 客户端（测试中可注入 MockClient 构造的 RecordApi） */
  final RecordApi api;

  /** 下载目标目录（测试中可注入临时目录；null 时使用应用文档目录） */
  final Directory? downloadDir;

  const RecordListPage({super.key, required this.api, this.downloadDir});

  @override
  State<RecordListPage> createState() => _RecordListPageState();
}

class _RecordListPageState extends State<RecordListPage> {
  List<RecordInfo>? _records;
  bool _loading = false;
  String? _error;
  final Set<int> _downloadingIds = {};

  @override
  void initState() {
    super.initState();
    _refresh();
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
          Expanded(child: _buildBody()),
        ],
      ),
    );
  }

  /** AP 连接引导横幅 */
  Widget _buildApGuideBanner() {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.all(8),
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
          const Text(
            '请连接手机 WiFi 到热点 ESP32-ECG-XXXX（密码 12345678），然后返回本页刷新',
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
}
