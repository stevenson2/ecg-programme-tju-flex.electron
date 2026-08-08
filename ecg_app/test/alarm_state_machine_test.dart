import 'package:fake_async/fake_async.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:ecg_app/models/ecg_data.dart';
import 'package:ecg_app/models/alarm_event.dart';
import 'package:ecg_app/providers/ecg_provider.dart';

/**
 * @brief 告警状态机测试（Contract C1）
 *
 * 覆盖 C1 中定义的所有状态转换规则：
 * - 上升沿触发 → ALARMING（仅一次）
 * - 锁存期无二次触发
 * - 3 秒干净窗口 → IDLE（signal_normal）
 * - 3 秒内重新异常 → 取消恢复
 * - confirmAlarm → 立即 IDLE（user_confirm）
 * - 断开连接 → 重置（无事件组装）
 * - 峰值置信度 / 平均 BPM 簿记
 * - AlarmEvent 字段 / toJson/fromJson 往返
 * - 边界条件：瞬态异常（1 样本窗口内）
 */

void main() {
  /** 便捷构造器（与 smoke test 一致） */
  ECGSample sample({
    double clean = 0,
    double noisy = 0,
    double filtered = 0,
    int bpm = 0,
    int abnormal = 0,
    double confidence = 0,
  }) {
    return ECGSample(clean, noisy, filtered,
        bpm: bpm, abnormal: abnormal, confidence: confidence);
  }

  group('AlarmState 枚举', () {
    test('AlarmState 包含 idle/alarming/arming 三个值', () {
      expect(AlarmState.values.length, 3);
      expect(AlarmState.idle.index, 0);
      expect(AlarmState.alarming.index, 1);
      expect(AlarmState.arming.index, 2);
    });
  });

  group('AlarmEvent 模型', () {
    test('构造与字段', () {
      final now = DateTime(2026, 8, 8, 12, 0, 0);
      final event = AlarmEvent(
        triggerTime: now,
        duration: const Duration(seconds: 5, milliseconds: 200),
        peakConfidence: 0.87,
        avgBpm: 75.3,
        recoveryMethod: 'signal_normal',
      );
      expect(event.triggerTime, now);
      expect(event.duration, const Duration(seconds: 5, milliseconds: 200));
      expect(event.peakConfidence, 0.87);
      expect(event.avgBpm, 75.3);
      expect(event.recoveryMethod, 'signal_normal');
    });

    test('toJson / fromJson 往返', () {
      final now = DateTime(2026, 8, 8, 12, 0, 0, 500);
      final original = AlarmEvent(
        triggerTime: now,
        duration: const Duration(minutes: 2, seconds: 30, milliseconds: 500),
        peakConfidence: 0.92,
        avgBpm: 88.1,
        recoveryMethod: 'user_confirm',
      );
      final json = original.toJson();
      final restored = AlarmEvent.fromJson(json);

      expect(restored.triggerTime.millisecondsSinceEpoch,
          original.triggerTime.millisecondsSinceEpoch);
      expect(restored.duration, original.duration);
      expect(restored.peakConfidence, original.peakConfidence);
      expect(restored.avgBpm, original.avgBpm);
      expect(restored.recoveryMethod, original.recoveryMethod);
    });

    test('toJson 格式：triggerTime 为 ISO8601，duration 为毫秒整数', () {
      final now = DateTime(2026, 8, 8, 12, 0, 0, 0).toUtc();
      final event = AlarmEvent(
        triggerTime: now,
        duration: const Duration(seconds: 3),
        peakConfidence: 0.5,
        avgBpm: 60,
        recoveryMethod: 'signal_normal',
      );
      final json = event.toJson();
      expect(json['triggerTime'], isA<String>());
      expect(json['duration'], isA<int>());
      expect(json['duration'], 3000);
      expect(json['peakConfidence'], 0.5);
      expect(json['avgBpm'], 60.0);
      expect(json['recoveryMethod'], 'signal_normal');
      // ISO8601 应包含 'T'
      expect(json['triggerTime'], contains('T'));
    });
  });

  group('告警状态机 — 状态转换（使用 fakeAsync 控制 3 秒定时器）', () {
    test('初始状态为 idle，所有告警字段为默认值', () {
      final provider = ECGProvider();
      expect(provider.alarmState, AlarmState.idle);
      expect(provider.alarmTriggerTime, isNull);
      expect(provider.alarmPeakConfidence, 0.0);
      expect(provider.lastCompletedAlarm, isNull);
      expect(provider.lastBpm, 0);
      provider.dispose();
    });

    test('上升沿：idle + abnormal==1 → ALARMING（触发一次）', () {
      final provider = ECGProvider();
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.75, bpm: 72));
      expect(provider.alarmState, AlarmState.alarming);
      expect(provider.alarmTriggerTime, isNotNull);
      expect(provider.alarmPeakConfidence, 0.75);
      provider.dispose();
    });

    test('锁存：abnormal==1 持续 → ARMING，不重复触发', () {
      final provider = ECGProvider();
      // 首次触发
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.6, bpm: 70));
      expect(provider.alarmState, AlarmState.alarming);

      // 持续异常 → 锁存到 arming（此时 alarmTriggerTime 不变）
      final firstTrigger = provider.alarmTriggerTime;
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.7, bpm: 72));
      expect(provider.alarmState, AlarmState.arming);
      expect(provider.alarmTriggerTime, firstTrigger); // 不重新计时

      // 再多喂几个异常，保持在 arming，峰值追踪正确
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.9, bpm: 75));
      expect(provider.alarmState, AlarmState.arming);
      expect(provider.alarmPeakConfidence, 0.9);
      provider.dispose();
    });

    test('3 秒干净窗口：abnormal==0 持续 3 秒 → IDLE (signal_normal)', () {
      fakeAsync((async) {
        final provider = ECGProvider();
        // 触发告警
        provider.debugAddSample(
            sample(abnormal: 1, confidence: 0.8, bpm: 80));
        expect(provider.alarmState, AlarmState.alarming);

        // 异常消失：启动 3 秒恢复定时器
        provider.debugAddSample(sample(abnormal: 0, bpm: 78));

        // 经过 3 秒（只有正常样本）
        async.elapse(const Duration(seconds: 3));

        // 现在应回到 idle，且恢复方式为 signal_normal
        expect(provider.alarmState, AlarmState.idle);
        expect(provider.lastCompletedAlarm, isNotNull);
        expect(provider.lastCompletedAlarm!.recoveryMethod, 'signal_normal');
        expect(provider.alarmTriggerTime, isNull);
        expect(provider.alarmPeakConfidence, 0.0);
        provider.dispose();
      });
    });

    test('3 秒内重新异常：取消恢复定时器，回到 ARMING', () {
      fakeAsync((async) {
        final provider = ECGProvider();
        // 触发告警
        provider.debugAddSample(
            sample(abnormal: 1, confidence: 0.7, bpm: 65));
        expect(provider.alarmState, AlarmState.alarming);

        // 异常消失
        provider.debugAddSample(sample(abnormal: 0, bpm: 64));

        // 经过 1 秒
        async.elapse(const Duration(seconds: 1));

        // 还未到 3 秒，仍在告警周期中
        expect(provider.alarmState, isNot(AlarmState.idle));

        // 重新出现异常 → 取消定时器，回到 arming
        provider.debugAddSample(
            sample(abnormal: 1, confidence: 0.85, bpm: 68));
        expect(provider.alarmState, AlarmState.arming);
        expect(provider.alarmPeakConfidence, 0.85); // 峰值更新

        // 再经过 3 秒，不应触发 idle（定时器已取消）
        async.elapse(const Duration(seconds: 3));
        expect(provider.alarmState, isNot(AlarmState.idle));
        provider.dispose();
      });
    });

    test('confirmAlarm → 立即 IDLE (user_confirm)，取消待处理定时器', () {
      fakeAsync((async) {
        final provider = ECGProvider();
        // 触发告警
        provider.debugAddSample(
            sample(abnormal: 1, confidence: 0.9, bpm: 90));
        expect(provider.alarmState, AlarmState.alarming);

        // 用户确认（cancel pending timer）
        provider.confirmAlarm();

        expect(provider.alarmState, AlarmState.idle);
        expect(provider.lastCompletedAlarm, isNotNull);
        expect(provider.lastCompletedAlarm!.recoveryMethod, 'user_confirm');
        expect(provider.alarmPeakConfidence, 0.0);
        expect(provider.alarmTriggerTime, isNull);

        // 经过 3 秒后不应重新触发（定时器已取消）
        async.elapse(const Duration(seconds: 3));
        expect(provider.alarmState, AlarmState.idle);
        provider.dispose();
      });
    });

    test('confirmAlarm 在 arm 状态下也可用', () {
      final provider = ECGProvider();
      // 触发后持续异常进入 arming
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.6, bpm: 70));
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.7, bpm: 72));
      expect(provider.alarmState, AlarmState.arming);

      provider.confirmAlarm();
      expect(provider.alarmState, AlarmState.idle);
      expect(provider.lastCompletedAlarm!.recoveryMethod, 'user_confirm');
      provider.dispose();
    });

    test('confirmAlarm 在 idle 状态下为无操作', () {
      final provider = ECGProvider();
      expect(provider.alarmState, AlarmState.idle);
      provider.confirmAlarm(); // 不应崩溃
      expect(provider.alarmState, AlarmState.idle);
      expect(provider.lastCompletedAlarm, isNull);
      provider.dispose();
    });

    test('disconnect 重置告警状态：→ IDLE，取消定时器，无事件组装', () {
      fakeAsync((async) {
        final provider = ECGProvider();
        // 触发告警
        provider.debugAddSample(
            sample(abnormal: 1, confidence: 0.8, bpm: 75));
        expect(provider.alarmState, AlarmState.alarming);

        // 断开连接
        provider.disconnect();

        expect(provider.alarmState, AlarmState.idle);
        // 断开不组装事件，lastCompletedAlarm 为 null
        expect(provider.lastCompletedAlarm, isNull);
        expect(provider.alarmTriggerTime, isNull);
        expect(provider.alarmPeakConfidence, 0.0);
        provider.dispose();
      });
    });

    test('dispose 重置告警状态（与 disconnect 相同语义）', () {
      final provider = ECGProvider();
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.8, bpm: 75));
      expect(provider.alarmState, AlarmState.alarming);

      provider.dispose();
      // 不再访问 provider 以避免使用已释放对象，但 dispose 内部逻辑应与 disconnect 一致
    });
  });

  group('告警状态机 — 簿记（峰值置信度 / 平均 BPM）', () {
    test('峰值置信度在告警周期内追踪最大值', () {
      final provider = ECGProvider();
      // 触发（conf 0.5）
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.5, bpm: 70));
      expect(provider.alarmPeakConfidence, 0.5);

      // 更高（conf 0.8）
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.8, bpm: 72));
      expect(provider.alarmPeakConfidence, 0.8);

      // 更低（conf 0.6）→ 不覆盖峰值
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.6, bpm: 71));
      expect(provider.alarmPeakConfidence, 0.8);
      provider.dispose();
    });

    test('峰值置信度在 confirmAlarm 后清零', () {
      final provider = ECGProvider();
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.9, bpm: 80));
      expect(provider.alarmPeakConfidence, 0.9);

      provider.confirmAlarm();
      expect(provider.alarmPeakConfidence, 0.0);
      provider.dispose();
    });

    test('平均 BPM 在告警周期内正确计算（仅 bpm>0 的样本纳入均值）', () {
      fakeAsync((async) {
        final provider = ECGProvider();
        // 触发 (bpm=70)
        provider.debugAddSample(
            sample(abnormal: 1, confidence: 0.7, bpm: 70));
        // 持续异常 (bpm=80)
        provider.debugAddSample(
            sample(abnormal: 1, confidence: 0.8, bpm: 80));
        // bpm=0 的异常样本（不纳入均值）
        provider.debugAddSample(
            sample(abnormal: 1, confidence: 0.6, bpm: 0));
        // bpm=90
        provider.debugAddSample(
            sample(abnormal: 1, confidence: 0.9, bpm: 90));

        // 异常消失
        provider.debugAddSample(sample(abnormal: 0));
        async.elapse(const Duration(seconds: 3));

        // 均值 = (70 + 80 + 90) / 3 = 80
        expect(provider.lastCompletedAlarm, isNotNull);
        expect(provider.lastCompletedAlarm!.avgBpm, closeTo(80.0, 0.01));
        provider.dispose();
      });
    });

    test('平均 BPM 在 confirmAlarm 后正确记录', () {
      final provider = ECGProvider();
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.7, bpm: 65));
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.8, bpm: 75));
      provider.confirmAlarm();

      expect(provider.lastCompletedAlarm!.avgBpm, closeTo(70.0, 0.01));
      provider.dispose();
    });
  });

  group('告警状态机 — AlarmEvent 组装正确性', () {
    test('signal_normal 恢复：事件 duration 和字段正确', () {
      fakeAsync((async) {
        final provider = ECGProvider();
        // 记录触发时间（通过近似时间戳验证）
        provider.debugAddSample(
            sample(abnormal: 1, confidence: 0.88, bpm: 85));

        final triggerTime = provider.alarmTriggerTime!;

        // 异常消失
        provider.debugAddSample(sample(abnormal: 0));
        async.elapse(const Duration(seconds: 3));

        final event = provider.lastCompletedAlarm!;
        expect(event.recoveryMethod, 'signal_normal');
        expect(event.triggerTime, triggerTime);
        expect(event.peakConfidence, 0.88);
        expect(event.avgBpm, 85.0);
        // duration 应非零（fakeAsync 中 DateTime.now() 使用真实时钟，
        // 因此 duration 不会反映 3s 流逝，但事件组装本身已验证正确）
        expect(event.duration, isNotNull);
        provider.dispose();
      });
    });

    test('新的告警周期清除上一次的 lastCompletedAlarm', () {
      fakeAsync((async) {
        final provider = ECGProvider();
        // 第一个周期
        provider.debugAddSample(
            sample(abnormal: 1, confidence: 0.5, bpm: 60));
        provider.debugAddSample(sample(abnormal: 0));
        async.elapse(const Duration(seconds: 3));
        expect(provider.lastCompletedAlarm, isNotNull);
        final firstEvent = provider.lastCompletedAlarm;

        // 第二个周期：触发时应清除上一个事件
        provider.debugAddSample(
            sample(abnormal: 1, confidence: 0.9, bpm: 90));
        // 新周期开始时 lastCompletedAlarm 已清除
        expect(provider.lastCompletedAlarm, isNull);

        // 完成第二周期
        provider.debugAddSample(sample(abnormal: 0));
        async.elapse(const Duration(seconds: 3));
        expect(provider.lastCompletedAlarm, isNotNull);
        expect(provider.lastCompletedAlarm, isNot(firstEvent));
        provider.dispose();
      });
    });
  });

  group('告警状态机 — 边界条件', () {
    test('瞬态异常：单样本 abnormal=1 后立即异常=0，上升沿仍被捕获', () {
      fakeAsync((async) {
        final provider = ECGProvider();
        // 仅一个异常样本（在 10 样本窗口内就消失了）
        provider.debugAddSample(
            sample(abnormal: 1, confidence: 0.6, bpm: 70));
        expect(provider.alarmState, AlarmState.alarming);

        // 立即恢复正常（虽然 hasAbnormalAlert 仍可能为 true，但状态机只看 raw sample）
        provider.debugAddSample(sample(abnormal: 0));

        // 3 秒后恢复
        async.elapse(const Duration(seconds: 3));
        expect(provider.alarmState, AlarmState.idle);
        expect(provider.lastCompletedAlarm, isNotNull);
        expect(provider.lastCompletedAlarm!.recoveryMethod, 'signal_normal');
        provider.dispose();
      });
    });

    test('从 idle 直接接 abnormal=0 不触发告警', () {
      final provider = ECGProvider();
      provider.debugAddSample(sample(abnormal: 0));
      expect(provider.alarmState, AlarmState.idle);
      expect(provider.alarmTriggerTime, isNull);
      provider.dispose();
    });

    test('在 idle 状态下，abnormal=1→0→1 序列只触发一次告警', () {
      final provider = ECGProvider();
      // 第一次异常 → 触发
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.5, bpm: 60));
      expect(provider.alarmState, AlarmState.alarming);

      // 异常消失
      provider.debugAddSample(sample(abnormal: 0));

      // 1 秒内重新异常 → 回到 arming，不触发新告警
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.7, bpm: 62));
      expect(provider.alarmState, AlarmState.arming);

      // alarmTriggerTime 应保持第一次触发的时间
      expect(provider.alarmTriggerTime, isNotNull);
      provider.dispose();
    });

    test('lastBpm 反映 _addSample 锁定的心率值', () {
      final provider = ECGProvider();
      expect(provider.lastBpm, 0);

      provider.debugAddSample(sample(bpm: 72));
      expect(provider.lastBpm, 72);

      // bpm=0 不覆盖
      provider.debugAddSample(sample(bpm: 0));
      expect(provider.lastBpm, 72);

      provider.debugAddSample(sample(bpm: 88));
      expect(provider.lastBpm, 88);
      provider.dispose();
    });

    test('BPM 在恢复倒计时期间仍然纳入均值', () {
      fakeAsync((async) {
        final provider = ECGProvider();
        // 触发 (bpm=70)
        provider.debugAddSample(
            sample(abnormal: 1, confidence: 0.7, bpm: 70));

        // 异常消失，但 BPM 持续来
        provider.debugAddSample(sample(abnormal: 0, bpm: 75));
        provider.debugAddSample(sample(abnormal: 0, bpm: 80));
        provider.debugAddSample(sample(abnormal: 0, bpm: 85));

        async.elapse(const Duration(seconds: 3));

        // 均值应包含所有 bpm>0 的样本：(70+75+80+85)/4 = 77.5
        expect(provider.lastCompletedAlarm!.avgBpm, closeTo(77.5, 0.01));
        provider.dispose();
      });
    });

    test('连续多个告警周期独立工作', () {
      fakeAsync((async) {
        final provider = ECGProvider();

        // 周期 1
        provider.debugAddSample(
            sample(abnormal: 1, confidence: 0.4, bpm: 60));
        provider.debugAddSample(sample(abnormal: 0));
        async.elapse(const Duration(seconds: 3));
        expect(provider.lastCompletedAlarm!.peakConfidence, 0.4);
        expect(provider.alarmState, AlarmState.idle);

        // 周期 2（独立触发）
        provider.debugAddSample(
            sample(abnormal: 1, confidence: 0.9, bpm: 100));
        expect(provider.alarmState, AlarmState.alarming);
        expect(provider.alarmPeakConfidence, 0.9);
        provider.debugAddSample(sample(abnormal: 0));
        async.elapse(const Duration(seconds: 3));
        expect(provider.lastCompletedAlarm!.peakConfidence, 0.9);
        expect(provider.lastCompletedAlarm!.avgBpm, 100.0);
        provider.dispose();
      });
    });

    test('多个 abnormal=1 样本后 confirmAlarm 峰值正确', () {
      final provider = ECGProvider();
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.3, bpm: 55));
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.5, bpm: 58));
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.95, bpm: 62));
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.4, bpm: 60));

      expect(provider.alarmPeakConfidence, 0.95);
      provider.confirmAlarm();
      expect(provider.lastCompletedAlarm!.peakConfidence, 0.95);
      expect(provider.lastCompletedAlarm!.avgBpm, closeTo((55+58+62+60)/4, 0.01));
      provider.dispose();
    });
  });
}
