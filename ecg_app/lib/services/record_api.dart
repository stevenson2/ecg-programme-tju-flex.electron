import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

/**
 * @file record_api.dart
 * @brief ESP32-ECG 记录 HTTP API 客户端（Contract C7）
 *
 * 固件端已实现下列端点，客户端通过 HTTP 与之交互：
 *   GET    /api/records            → 记录列表
 *   GET    /api/records/{id}/meta  → 单条记录元数据
 *   GET    /api/records/{id}/data  → 原始 .ecgr 字节
 *   DELETE /api/records/{id}       → 删除记录
 *
 * 固件服务地址：http://192.168.4.1（ESP32 热点）
 */

/** 记录列表项（来自 /api/records） */
class RecordInfo {
  final int id;
  final int duration;          /**< 录制时长（秒） */
  final int size;              /**< 文件大小（字节） */
  final int abnormalSeconds;   /**< 异常秒数 */
  final String start;          /**< ISO 8601 开始时间 */

  const RecordInfo({
    required this.id,
    required this.duration,
    required this.size,
    required this.abnormalSeconds,
    required this.start,
  });

  factory RecordInfo.fromJson(Map<String, dynamic> json) {
    return RecordInfo(
      id: json['id'] as int,
      duration: json['duration'] as int,
      size: json['size'] as int,
      abnormalSeconds: (json['abnormal_seconds'] as int?) ?? 0,
      start: (json['start'] as String?) ?? '',
    );
  }
}

/** 记录元数据（来自 /api/records/{id}/meta） */
class RecordMeta {
  final int id;
  final int sampleRate;        /**< 采样率（固定 250 Hz） */
  final int startUnix;         /**< Unix 时间戳（秒） */
  final int duration;          /**< 时长（秒） */
  final int totalSamples;      /**< 总采样点数 */
  final int abnormalSeconds;   /**< 异常秒数 */

  const RecordMeta({
    required this.id,
    required this.sampleRate,
    required this.startUnix,
    required this.duration,
    required this.totalSamples,
    required this.abnormalSeconds,
  });

  factory RecordMeta.fromJson(Map<String, dynamic> json) {
    return RecordMeta(
      id: json['id'] as int,
      sampleRate: (json['sample_rate'] as int?) ?? 250,
      startUnix: (json['start_unix'] as int?) ?? 0,
      duration: json['duration'] as int,
      totalSamples: (json['total_samples'] as int?) ?? 0,
      abnormalSeconds: (json['abnormal_seconds'] as int?) ?? 0,
    );
  }
}

/** 记录 API 调用异常 */
class RecordApiException implements Exception {
  final String message;
  final int? statusCode;

  const RecordApiException(this.message, {this.statusCode});

  @override
  String toString() => 'RecordApiException: $message (status=$statusCode)';
}

/**
 * @brief ESP32 记录 HTTP API 客户端
 *
 * 可注入 http.Client（用于测试 MockClient）和 baseUrl。
 * 连接 ESP32 热点后使用默认地址 http://192.168.4.1。
 */
class RecordApi {
  final http.Client _client;
  final String baseUrl;

  static const Duration _defaultTimeout = Duration(seconds: 10);
  static const Duration _downloadTimeout = Duration(seconds: 60);

  RecordApi({
    http.Client? client,
    this.baseUrl = 'http://192.168.4.1',
  }) : _client = client ?? http.Client();

  /**
   * @brief 获取记录列表
   * @return RecordInfo 列表（无记录时返回空列表）
   * @throws RecordApiException 非 200 响应
   */
  Future<List<RecordInfo>> listRecords() async {
    final uri = Uri.parse('$baseUrl/api/records');
    final response = await _client.get(uri).timeout(_defaultTimeout);
    _checkResponse(response);

    final body = jsonDecode(response.body) as Map<String, dynamic>;
    final records = (body['records'] as List<dynamic>?) ?? [];
    return records
        .map((e) => RecordInfo.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /**
   * @brief 获取单条记录元数据
   * @param id 记录 ID
   * @throws RecordApiException 非 200 响应
   */
  Future<RecordMeta> getMeta(int id) async {
    final uri = Uri.parse('$baseUrl/api/records/$id/meta');
    final response = await _client.get(uri).timeout(_defaultTimeout);
    _checkResponse(response);

    return RecordMeta.fromJson(
        jsonDecode(response.body) as Map<String, dynamic>);
  }

  /**
   * @brief 下载记录原始 .ecgr 字节
   *
   * 客户端不发送 Range 请求头，期望服务器 200 返回完整文件。
   * Content-Length 由服务器设置。
   *
   * @param id 记录 ID
   * @return 完整 .ecgr 字节（32B 头部 + int16 样本 + 位图）
   * @throws RecordApiException 非 200 响应
   */
  Future<Uint8List> downloadData(int id) async {
    final uri = Uri.parse('$baseUrl/api/records/$id/data');
    final response = await _client.get(uri).timeout(_downloadTimeout);
    _checkResponse(response);

    return response.bodyBytes;
  }

  /**
   * @brief 删除记录
   * @param id 记录 ID
   * @return true 删除成功
   * @throws RecordApiException 非 200 响应
   */
  Future<bool> deleteRecord(int id) async {
    final uri = Uri.parse('$baseUrl/api/records/$id');
    final response = await _client.delete(uri).timeout(_defaultTimeout);
    _checkResponse(response);

    final body = jsonDecode(response.body) as Map<String, dynamic>;
    return body['deleted'] == true;
  }

  /// 检查 HTTP 响应状态码，非 200 抛出 RecordApiException
  void _checkResponse(http.Response response) {
    if (response.statusCode != 200) {
      throw RecordApiException(
        '${response.statusCode}: ${response.reasonPhrase}',
        statusCode: response.statusCode,
      );
    }
  }

  /// 释放底层 HTTP 客户端资源
  void dispose() {
    _client.close();
  }
}
