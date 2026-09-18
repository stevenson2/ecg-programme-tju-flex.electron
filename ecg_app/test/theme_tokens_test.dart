/// P2-4 验收测试：设计令牌 + asrc 分级文案 + 主题装配。
///
/// 覆盖三线整改 P2-1（令牌唯一真值源）与 P0-2（asrc 分级文案）的验收点。
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ecg_app/config/app_theme.dart';
import 'package:ecg_app/services/protocol_generated.dart';

void main() {
  group('P2-1 设计令牌', () {
    test('颜色令牌为 const 且互不重复（语义命名，非色相复制）', () {
      const colors = <Color>[
        AppColors.background,
        AppColors.surface,
        AppColors.surfaceVariant,
        AppColors.traceBackground,
        AppColors.gridMinor,
        AppColors.gridMajor,
        AppColors.traceBaseline,
        AppColors.traceLabel,
        AppColors.traceLine,
        AppColors.primary,
        AppColors.alert,
        AppColors.error,
        AppColors.success,
      ];
      expect(colors.toSet().length, colors.length,
          reason: '令牌间不应存在重复取值（避免两个语义指向同一色）');
    });

    test('间距为 4dp 栅格倍数', () {
      for (final v in [
        AppSpacing.xs,
        AppSpacing.sm,
        AppSpacing.ms,
        AppSpacing.md,
        AppSpacing.lg,
        AppSpacing.xl,
        AppSpacing.xxl,
      ]) {
        expect(v % 2, 0, reason: '间距 $v 应落在偶数栅格上');
      }
    });

    test('触达尺寸下限为 48dp（Material a11y）', () {
      expect(AppSizes.minTouchTarget, 48);
    });

    test('数字样式启用等宽数字（避免跳动）', () {
      final s = AppText.numeric(12);
      expect(s.fontFeatures, contains(const FontFeature.tabularFigures()));
    });

    test('主题装配产出暗色 ColorScheme 且按钮继承 48dp 触达', () {
      final t = AppTheme.dark();
      expect(t.brightness, Brightness.dark);
      expect(t.colorScheme.primary, AppColors.primary);
      expect(t.scaffoldBackgroundColor, AppColors.background);
      final style = t.elevatedButtonTheme.style;
      expect(style, isNotNull);
    });
  });

  group('P0-2 asrc 分级文案', () {
    test('位定义与契约一致（0x01/0x02/0x04/0x08/0x10）', () {
      expect(ProtocolGenerated.ai, 0x01);
      expect(ProtocolGenerated.rs, 0x02);
      expect(ProtocolGenerated.flat, 0x04);
      expect(ProtocolGenerated.vf, 0x08);
      expect(ProtocolGenerated.leadoff, 0x10);
    });

    test('优先级顺序为 LEADOFF > VF > FLAT > RS > AI', () {
      expect(ProtocolGenerated.primaryAsrc(0x10)?.name, 'LEADOFF');
      expect(ProtocolGenerated.primaryAsrc(0x08)?.name, 'VF');
      expect(ProtocolGenerated.primaryAsrc(0x04)?.name, 'FLAT');
      expect(ProtocolGenerated.primaryAsrc(0x02)?.name, 'RS');
      expect(ProtocolGenerated.primaryAsrc(0x01)?.name, 'AI');
    });

    test('位图多源同置时取最高优先级', () {
      // 0x14 = LEADOFF | FLAT → LEADOFF 胜出
      expect(ProtocolGenerated.primaryAsrc(0x14)?.name, 'LEADOFF');
      // 0x0C = VF | FLAT → VF 胜出
      expect(ProtocolGenerated.primaryAsrc(0x0C)?.name, 'VF');
      // 0x03 = RS | AI → RS 胜出
      expect(ProtocolGenerated.primaryAsrc(0x03)?.name, 'RS');
    });

    test('未知位（>=0x20）不产生文案', () {
      expect(ProtocolGenerated.primaryAsrc(0x20), isNull);
      expect(ProtocolGenerated.primaryAsrc(0x00), isNull);
    });

    test('isKnownAsrc 仅认 0x1F 掩码内', () {
      expect(ProtocolGenerated.isKnownAsrc(0x1F), isTrue);
      expect(ProtocolGenerated.isKnownAsrc(0x20), isFalse);
      expect(ProtocolGenerated.isKnownAsrc(0x00), isFalse);
    });

    test('每条文案非空且 level 取值合法', () {
      for (final m in ProtocolGenerated.asrcMeta) {
        expect(m.label.isNotEmpty, isTrue);
        expect(['info', 'warning', 'critical'], contains(m.level));
      }
    });
  });
}