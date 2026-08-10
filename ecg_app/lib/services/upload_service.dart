import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import 'ecg_record_codec.dart';

/**
 * @file upload_service.dart
 * @brief 云端记录上传服务客户端（Contract C8）
 *
 * 端点：
 *   POST   /v1/records                  multipart 上传 .ecgr + JSON 元数据
 *   POST   /v1/records/{id}/analyze     触发云端分析
 *   GET    /v1/records/{id}/report      获取分析报告
 *
 * 认证：Bearer token（默认 'dev-token'，configurable const）。
 * 生产环境替换 baseUrl 为实际云服务地址。
 *
 * 元数据结构（multipart "meta" part）：
 *   {
 *     "device_id": "esp32-ecg-app",
 *     "firmware_version": "app-v1.0.0",
 *     "sample_rate": <int>,
 *     "duration_sec": <int>,
 *     "total_samples": <int>,
 *     "abnormal_seconds": <int>,
 *     "abnormal_ratio": <double>,
 *     "start_unix": <int>,
 *     "onboard_ai_summary": {
 *       "model": "exp6-SGD",
 *       "abnormal_seconds": <int>,
 *       "abnormal_ratio": <double>,
 *       "total_duration": <int>
 *     }
 *   }
 */

/** 上传结果（POST /v1/records 201 响应） */
class UploadResult {
  final String recordId;
  final String status;

  const UploadResult({required this.recordId, required this.status});

  factory UploadResult.fromJson(Map<String, dynamic> json) {
    return UploadResult(
      recordId: (json['record_id'] as String?) ?? '',
      status: (json['status'] as String?) ?? 'unknown',
    );
  }
}

/** 分析报告（analyze / report 端点响应） */
class AnalysisReport {
  final String recordId;
  final String status;
  final Map<String, dynamic>? summary;
  final List<Map<String, dynamic>>? events;
  final String? recommendation;

  const AnalysisReport({
    required this.recordId,
    required this.status,
    this.summary,
    this.events,
    this.recommendation,
  });

  factory AnalysisReport.fromJson(Map<String, dynamic> json) {
    return AnalysisReport(
      recordId: (json['record_id'] as String?) ?? '',
      status: (json['status'] as String?) ?? 'unknown',
      summary: json['summary'] as Map<String, dynamic>?,
      events: (json['events'] as List<dynamic>?)
          ?.map((e) => Map<String, dynamic>.from(e as Map))
          .toList(),
      recommendation: json['recommendation'] as String?,
    );
  }
}

/** 云端上传 API 调用异常 */
class CloudUploadException implements Exception {
  final String message;
  final int? statusCode;

  const CloudUploadException(this.message, {this.statusCode});

  @override
  String toString() => 'CloudUploadException: $message (status=$statusCode)';
}

/**
 * @brief 云端记录上传 HTTP 客户端
 *
 * 可注入 http.Client（用于测试 MockClient）、baseUrl 和 token。
 * 默认指向本地 mock 服务器；生产环境需替换 baseUrl 与 token。
 */
class CloudUploadService {
  final http.Client _client;
  final String baseUrl;
  final String token;

  static const String defaultBaseUrl = 'http://127.0.0.1:8000/v1';
  static const String defaultToken = 'dev-token';
  static const Duration _defaultTimeout = Duration(seconds: 30);

  CloudUploadService({
    http.Client? client,
    this.baseUrl = defaultBaseUrl,
    this.token = defaultToken,
  }) : _client = client ?? http.Client();

  /**
   * @brief 上传 .ecgr 记录到云端
   *
   * multipart POST /v1/records：
   *   - part "meta"：JSON 元数据字符串
   *   - part "data"：原始 .ecgr 字节文件
   *
   * @param ecgrFile 本地 .ecgr 文件路径
   * @param metaSource 解码后的 EcgRecord（用于提取元数据字段）
   * @return UploadResult（含 record_id）
   * @throws CloudUploadException 文件不存在 / 非 201 响应
   */
  Future<UploadResult> uploadRecord(File ecgrFile, EcgRecord metaSource) async {
    if (!await ecgrFile.exists()) {
      throw CloudUploadException('文件不存在: ${ecgrFile.path}');
    }

    final metaJson = _buildMetaJson(metaSource);

    final uri = Uri.parse('$baseUrl/records');
    final request = http.MultipartRequest('POST', uri);
    request.headers['Authorization'] = 'Bearer $token';
    request.fields['meta'] = jsonEncode(metaJson);
    request.files.add(
      await http.MultipartFile.fromPath('data', ecgrFile.path),
    );

    final streamedResponse =
        await _client.send(request).timeout(_defaultTimeout);
    final bodyBytes = await streamedResponse.stream.toBytes();
    final response = http.Response.bytes(
      bodyBytes,
      streamedResponse.statusCode,
      headers: streamedResponse.headers,
    );

    _checkCreated(response);

    return UploadResult.fromJson(
        jsonDecode(utf8.decode(bodyBytes)) as Map<String, dynamic>);
  }

  /**
   * @brief 触发云端 AI 分析
   *
   * POST /v1/records/{id}/analyze → 200
   *
   * @param recordId 云端记录 ID
   * @return AnalysisReport（status="analyzed"）
   * @throws CloudUploadException 非 200 响应
   */
  Future<AnalysisReport> analyze(String recordId) async {
    final uri = Uri.parse('$baseUrl/records/$recordId/analyze');
    final response = await _client
        .post(uri, headers: {'Authorization': 'Bearer $token'})
        .timeout(_defaultTimeout);
    _checkOk(response);

    return AnalysisReport.fromJson(
        jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>);
  }

  /**
   * @brief 获取分析报告
   *
   * GET /v1/records/{id}/report → 200
   *
   * @param recordId 云端记录 ID
   * @return AnalysisReport（status="completed" 时含 summary/events/recommendation）
   * @throws CloudUploadException 非 200 响应
   */
  Future<AnalysisReport> fetchReport(String recordId) async {
    final uri = Uri.parse('$baseUrl/records/$recordId/report');
    final response = await _client
        .get(uri, headers: {'Authorization': 'Bearer $token'})
        .timeout(_defaultTimeout);
    _checkOk(response);

    return AnalysisReport.fromJson(
        jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>);
  }

  // ───────────────────── 内部方法 ─────────────────────

  /** 从解码后的 EcgRecord 构建上传元数据 JSON */
  Map<String, dynamic> _buildMetaJson(EcgRecord record) {
    final abnormalRatio = record.durationSec > 0
        ? record.abnormalSeconds / record.durationSec
        : 0.0;

    return {
      'device_id': 'esp32-ecg-app',
      'firmware_version': 'app-v1.0.0',
      'sample_rate': record.sampleRate,
      'duration_sec': record.durationSec,
      'total_samples': record.totalSamples,
      'abnormal_seconds': record.abnormalSeconds,
      'abnormal_ratio': double.parse(abnormalRatio.toStringAsFixed(4)),
      'start_unix': record.startUnixTime,
      'onboard_ai_summary': {
        'model': 'exp6-SGD',
        'abnormal_seconds': record.abnormalSeconds,
        'abnormal_ratio':
            double.parse(abnormalRatio.toStringAsFixed(4)),
        'total_duration': record.durationSec,
      },
    };
  }

  /** 检查 201 Created 响应（上传） */
  void _checkCreated(http.Response response) {
    if (response.statusCode != 201) {
      throw CloudUploadException(
        '上传失败: ${response.statusCode} ${response.reasonPhrase}',
        statusCode: response.statusCode,
      );
    }
  }

  /** 检查 200 OK 响应（analyze / report） */
  void _checkOk(http.Response response) {
    if (response.statusCode != 200) {
      throw CloudUploadException(
        '请求失败: ${response.statusCode} ${response.reasonPhrase}',
        statusCode: response.statusCode,
      );
    }
  }

  /** 释放底层 HTTP 客户端资源 */
  void dispose() {
    _client.close();
  }
}
