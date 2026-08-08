/**
 * @file alarm_event.dart
 * @brief 告警事件模型 — 记录一次告警周期的元数据
 *
 * 字段说明：
 * - triggerTime: 告警触发时刻（上升沿时间戳）
 * - duration:     告警持续时长（触发 → 恢复）
 * - peakConfidence: 周期内 AI 异常置信度峰值
 * - avgBpm:       周期内心率均值（仅 bpm>0 样本纳入）
 * - recoveryMethod: 恢复方式 ('user_confirm' | 'signal_normal')
 *
 * toJson() 输出格式：
 *   triggerTime: ISO8601 字符串
 *   duration:    毫秒整数
 *   其余字段保持原样
 */

enum AlarmState { idle, alarming, arming }

class AlarmEvent {
  final DateTime triggerTime;
  final Duration duration;
  final double peakConfidence;
  final double avgBpm;
  final String recoveryMethod; // 'user_confirm' | 'signal_normal'

  const AlarmEvent({
    required this.triggerTime,
    required this.duration,
    required this.peakConfidence,
    required this.avgBpm,
    required this.recoveryMethod,
  });

  /// 序列化为 JSON（供持久化 / 跨 agent 通信）
  Map<String, dynamic> toJson() {
    return {
      'triggerTime': triggerTime.toUtc().toIso8601String(),
      'duration': duration.inMilliseconds,
      'peakConfidence': peakConfidence,
      'avgBpm': avgBpm,
      'recoveryMethod': recoveryMethod,
    };
  }

  /// 从 JSON 反序列化
  factory AlarmEvent.fromJson(Map<String, dynamic> json) {
    return AlarmEvent(
      triggerTime: DateTime.parse(json['triggerTime'] as String),
      duration: Duration(milliseconds: json['duration'] as int),
      peakConfidence: (json['peakConfidence'] as num).toDouble(),
      avgBpm: (json['avgBpm'] as num).toDouble(),
      recoveryMethod: json['recoveryMethod'] as String,
    );
  }

  @override
  String toString() =>
      'AlarmEvent(triggerTime=$triggerTime, duration=$duration, '
      'peakConfidence=$peakConfidence, avgBpm=$avgBpm, '
      'recoveryMethod=$recoveryMethod)';
}
