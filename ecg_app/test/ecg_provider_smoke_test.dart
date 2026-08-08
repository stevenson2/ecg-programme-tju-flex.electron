import 'package:flutter_test/flutter_test.dart';

import 'package:ecg_app/models/ecg_data.dart';
import 'package:ecg_app/providers/ecg_provider.dart';

/**
 * @brief ECGProvider 状态管理冒烟测试
 *
 * 通过 @visibleForTesting 的 debugAddSample 直接注入样本，
 * 验证环形缓冲区 / 心率锁定 / 异常置信度锁定 / 防闪烁告警窗口。
 */
void main() {
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

  group('ECGProvider 环形缓冲区与状态', () {
    test('环形缓冲区上限 kBufferSize=1500，溢出丢弃最旧样本', () {
      final provider = ECGProvider();
      for (int i = 0; i < ECGProvider.kBufferSize + 10; i++) {
        provider.debugAddSample(sample(clean: i.toDouble(), bpm: 60));
      }
      expect(provider.bufferSize, ECGProvider.kBufferSize);
      // 最旧的 10 个样本被丢弃
      expect(provider.samples.first.clean, 10.0);
      expect(provider.samples.last.clean,
          (ECGProvider.kBufferSize + 9).toDouble());
      provider.dispose();
    });

    test('lastSample 反映最新样本，空缓冲为 null', () {
      final provider = ECGProvider();
      expect(provider.lastSample, isNull);
      provider.debugAddSample(sample(clean: 0.1));
      provider.debugAddSample(sample(clean: 0.2));
      expect(provider.lastSample!.clean, 0.2);
      provider.dispose();
    });

    test('heartRate 在 bpm>0 时锁定，bpm=0 不覆盖', () {
      final provider = ECGProvider();
      expect(provider.heartRate, 0);
      provider.debugAddSample(sample(bpm: 0));
      expect(provider.heartRate, 0);
      provider.debugAddSample(sample(bpm: 75));
      expect(provider.heartRate, 75.0);
      // bpm=0 的样本不得覆盖已有心率
      provider.debugAddSample(sample(bpm: 0));
      expect(provider.heartRate, 75.0);
      provider.dispose();
    });

    test('abnormalConfidence 在 abnormal==1 时锁定', () {
      final provider = ECGProvider();
      expect(provider.abnormalConfidence, 0.0);
      provider.debugAddSample(sample(abnormal: 0, confidence: 0.9));
      expect(provider.abnormalConfidence, 0.0);
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.87));
      expect(provider.abnormalConfidence, 0.87);
      // 正常样本不覆盖置信度
      provider.debugAddSample(sample(abnormal: 0, confidence: 0.1));
      expect(provider.abnormalConfidence, 0.87);
      provider.dispose();
    });

    test('hasAbnormalAlert 为 kAbnormalWindow=10 防闪烁窗口', () {
      final provider = ECGProvider();
      // 异常样本 + 9 个正常样本：异常仍在 10 样本窗口内 → 告警
      provider.debugAddSample(sample(abnormal: 1, confidence: 0.8));
      for (int i = 0; i < 9; i++) {
        provider.debugAddSample(sample());
      }
      expect(provider.hasAbnormalAlert, isTrue);
      // 再喂 10 个正常样本：异常滑出窗口 → 不再告警
      for (int i = 0; i < 10; i++) {
        provider.debugAddSample(sample());
      }
      expect(provider.hasAbnormalAlert, isFalse);
      provider.dispose();
    });

    test('clear 重置缓冲区与状态', () {
      final provider = ECGProvider();
      provider.debugAddSample(sample(bpm: 70, abnormal: 1, confidence: 0.5));
      provider.clear();
      expect(provider.bufferSize, 0);
      expect(provider.heartRate, 0);
      expect(provider.abnormalConfidence, 0.0);
      expect(provider.hasAbnormalAlert, isFalse);
      provider.dispose();
    });
  });
}
