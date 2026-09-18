import 'package:flutter/material.dart';

/// 设计令牌（Design Tokens）—— P2-1 三线整改。
///
/// 本文件是 App 内所有颜色 / 字号 / 间距 / 圆角 / 触达尺寸的**唯一真值源**。
/// 禁止在 widget 中内联 `Color(0xFF...)`、`fontSize:` 魔数、`EdgeInsets` 魔数；
/// 新增视觉常量一律先加到这里，再被引用（R17 契约纪律，见 AGENTS.md）。
///
/// 命名约定：
///   - 颜色按「语义」命名（surface / alert / trace），不按色相命名；
///   - 间距按相对大小命名（xs/sm/md/lg/xl/xxl），不写具体像素；
///   - 波形相关色单独成组，供 ecg_waveform.dart 与 playback_page.dart 共用。

/// 颜色令牌。
class AppColors {
  AppColors._();

  // ---- 层级背景 ----
  /// 页面底色（Scaffold）。
  static const Color background = Color(0xFF0D0D1A);

  /// 卡片 / AppBar / 面板底色。
  static const Color surface = Color(0xFF1A1A2E);

  /// 次级表面：未激活按钮、边框、分隔。
  static const Color surfaceVariant = Color(0xFF2A2A3E);

  // ---- 波形绘制区 ----
  /// 波形区底色（比页面更深，保证对比度）。
  static const Color traceBackground = Color(0xFF0A0A0E);

  /// ECG 细格（1 mm）。
  static const Color gridMinor = Color(0xFF1A1A30);

  /// ECG 粗格（5 mm）—— P2-2 医用网格用，比细格亮。
  static const Color gridMajor = Color(0xFF23233F);

  /// 波形零基线。
  static const Color traceBaseline = Color(0xFF334466);

  /// 坐标轴刻度文字。
  static const Color traceLabel = Color(0xFF445566);

  /// 波形描迹（滤波后信号）。
  static const Color traceLine = Color(0xFF00E5FF);

  // ---- 语义色 ----
  /// 主色：正常态、激活态、主按钮。
  static const Color primary = Color(0xFF00BFFF);

  /// 告警红：波形/光晕/告警态。
  static const Color alert = Color(0xFFFF5252);

  /// 错误红：断开连接等破坏性操作。
  static const Color error = Color(0xFFE53935);

  /// 成功绿：连接正常、AI 正常。
  static const Color success = Color(0xFF4CAF50);

  /// 次级文字（标签、说明）。
  static const Color textSecondary = Color(0xFF9E9E9E);

  /// 禁用态。
  static const Color disabled = Color(0xFF5A5A6E);

  /// 提示/警告文字（状态消息）。
  static const Color warning = Colors.orange;
}

/// 间距令牌（4 dp 基准栅格）。
class AppSpacing {
  AppSpacing._();

  static const double xs = 2;
  static const double sm = 4;
  static const double ms = 6;
  static const double md = 8;
  static const double lg = 12;
  static const double xl = 16;
  static const double xxl = 24;
}

/// 圆角令牌。
class AppRadius {
  AppRadius._();

  static const double sm = 6;
  static const double md = 8;
  static const double lg = 12;

  static const BorderRadius smAll = BorderRadius.all(Radius.circular(sm));
  static const BorderRadius mdAll = BorderRadius.all(Radius.circular(md));
  static const BorderRadius lgAll = BorderRadius.all(Radius.circular(lg));
}

/// 触达尺寸与线宽。
///
/// P2-3 要求所有可点击目标 >= 48 dp（Material 无障碍下限）。
class AppSizes {
  AppSizes._();

  /// 最小触达边长（Material a11y 下限）。
  static const double minTouchTarget = 48;

  /// 波形描迹线宽。
  static const double traceStroke = 2.0;

  /// 波形光晕线宽。
  static const double traceGlowStroke = 4.0;

  /// 网格线宽。
  static const double gridStroke = 1.0;
}

/// 字号令牌（避免散落 fontSize 魔数）。
class AppFontSize {
  AppFontSize._();

  static const double xxs = 9;
  static const double xs = 11;
  static const double sm = 12;
  static const double md = 13;
  static const double lg = 14;
  static const double xl = 15;
  static const double xxl = 16;
  static const double xxxl = 18;
  static const double huge = 20;
  static const double display = 28;
}
/// 字号令牌（对应 Material 3 TextTheme 槽位，避免散落魔法数字）。
class AppText {
  AppText._();

  /// 数字专用字体特性：等宽数字（tabular figures）。
  /// 用于心率 / 置信度 / 计时等会跳动的数字，避免横向抖动。
  static const List<FontFeature> tabular = <FontFeature>[
    FontFeature.tabularFigures(),
  ];

  /// 基础 TextTheme（字形族统一在此声明）。
  static TextTheme textTheme() {
    const base = TextTheme();
    return base.copyWith(
      displayLarge: base.displayLarge?.copyWith(fontSize: 28),
      headlineMedium: base.headlineMedium?.copyWith(fontSize: 20),
      titleLarge: base.titleLarge?.copyWith(fontSize: 18, fontWeight: FontWeight.w500),
      titleMedium: base.titleMedium?.copyWith(fontSize: 16),
      bodyLarge: base.bodyLarge?.copyWith(fontSize: 15),
      bodyMedium: base.bodyMedium?.copyWith(fontSize: 14),
      bodySmall: base.bodySmall?.copyWith(fontSize: 13),
      labelLarge: base.labelLarge?.copyWith(fontSize: 12),
      labelMedium: base.labelMedium?.copyWith(fontSize: 11),
      labelSmall: base.labelSmall?.copyWith(fontSize: 9),
    );
  }

  /// 数字文本样式（等宽数字 + 指定字号）。
  static TextStyle numeric(double size, {Color? color, FontWeight? weight}) {
    return TextStyle(
      fontSize: size,
      color: color,
      fontWeight: weight,
      fontFeatures: tabular,
    );
  }
}

/// 应用主题装配。
class AppTheme {
  AppTheme._();

  /// 深色主题（当前唯一主题；浅色留待后续按需启用）。
  static ThemeData dark() {
    final textTheme = AppText.textTheme();
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: AppColors.background,
      textTheme: textTheme,
      colorScheme: const ColorScheme.dark(
        primary: AppColors.primary,
        surface: AppColors.surface,
        error: AppColors.error,
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: AppColors.surface,
        elevation: 0,
        centerTitle: true,
        titleTextStyle: textTheme.titleLarge,
      ),
      // 触达尺寸兜底：所有按钮最小 48 dp（P2-3）。
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          minimumSize: const Size(0, AppSizes.minTouchTarget),
          padding: const EdgeInsets.symmetric(
            horizontal: AppSpacing.lg,
            vertical: AppSpacing.sm,
          ),
          shape: const RoundedRectangleBorder(borderRadius: AppRadius.smAll),
        ),
      ),
      iconButtonTheme: IconButtonThemeData(
        style: IconButton.styleFrom(
          minimumSize: const Size(
            AppSizes.minTouchTarget,
            AppSizes.minTouchTarget,
          ),
        ),
      ),
    );
  }
}
