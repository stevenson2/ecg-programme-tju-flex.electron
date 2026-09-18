import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import 'config/app_theme.dart';
import 'providers/ecg_provider.dart';
import 'providers/settings_provider.dart';
import 'services/alarm_sound_service.dart';
import 'services/alarm_history_store.dart';
import 'services/record_schedule_service.dart';
import 'models/alarm_event.dart';
import 'widgets/ecg_waveform.dart';
import 'widgets/info_panel.dart';
import 'widgets/alarm_dialog.dart';
import 'widgets/history_sheet.dart';
import 'widgets/settings_sheet.dart';
import 'services/record_api.dart';
import 'services/ble_service.dart';
import 'pages/record_list_page.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // P2-3: 解锁横屏（横屏全幅波形 + 侧栏信息）
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]);
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => ECGProvider()),
        ChangeNotifierProvider(create: (_) => SettingsProvider()),
      ],
      child: const ECGApp(),
    ),
  );
}

class ECGApp extends StatelessWidget {
  const ECGApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '心电监测',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.dark(),
      home: const ECGMonitorScreen(),
    );
  }
}

class ECGMonitorScreen extends StatefulWidget {
  /// 可注入的提示音服务（测试用；null 时创建真实 AlarmSoundService）
  final AlarmSoundService? soundService;

  const ECGMonitorScreen({super.key, this.soundService});

  @override
  State<ECGMonitorScreen> createState() => _ECGMonitorScreenState();
}

class _ECGMonitorScreenState extends State<ECGMonitorScreen> {
  late final SettingsProvider _settings;
  late final ECGProvider _ecgProvider;
  late final AlarmSoundService _soundService;
  late final AlarmHistoryStore _historyStore;
  late final RecordScheduleService _recordScheduleService;
  List<AlarmEvent> _alarmHistory = [];
  AlarmState _prevAlarmState = AlarmState.idle;

  @override
  void initState() {
    super.initState();
    _ecgProvider = context.read<ECGProvider>();
    _settings = context.read<SettingsProvider>();
    _settings.load().then((_) {
      if (mounted) setState(() {});
    });
    _soundService =
        widget.soundService ?? AlarmSoundService(settings: _settings);
    _historyStore = AlarmHistoryStore();
    _historyStore.load().then((events) {
      if (mounted) setState(() => _alarmHistory = events);
    });
    _recordScheduleService = RecordScheduleService(
      settings: _settings,
      sendCommand: (cmd) => _ecgProvider.bleService.sendCommand(cmd),
    );
    _recordScheduleService.start();
    _ecgProvider.addListener(_onEcgChange);
  }

  @override
  void dispose() {
    _recordScheduleService.stop();
    _recordScheduleService.dispose();
    _ecgProvider.removeListener(_onEcgChange);
    _soundService.dispose();
    super.dispose();
  }

  /// 告警管线：监听 ECGProvider 状态变更，控制弹窗 / 提示音 / 历史记录
  void _onEcgChange() {
    final now = _ecgProvider.alarmState;
    final prev = _prevAlarmState;

    /// Rising edge: idle → alarming（每周期仅触发一次）
    if (prev == AlarmState.idle && now == AlarmState.alarming) {
      if (!_settings.dndEnabled) {
        if (mounted) {
          showAlarmDialog(context, _ecgProvider, _settings);
        }
        unawaited(_soundService.startAlarmLoop());
      }
      // DND 抑制弹窗与提示音，历史在 episode 完成时写入
    }
    /// Episode 完成：arming/alarming → idle（signal_normal 或 user_confirm）
    else if (now == AlarmState.idle && prev != AlarmState.idle) {
      unawaited(_soundService.stopAlarmLoop());
      final event = _ecgProvider.lastCompletedAlarm;
      if (event != null) {
        _alarmHistory = [event, ..._alarmHistory];
        unawaited(_historyStore.add(event));
        if (mounted) setState(() {});
      }
    }
    /// 断开重置（lastCompletedAlarm==null 时 idle 不变）—— 不做操作

    _prevAlarmState = now;
  }

  /// 告警总次数（已完成的 + 当前活跃的）
  int get _totalAlarmCount =>
      _alarmHistory.length +
      (_ecgProvider.alarmState != AlarmState.idle ? 1 : 0);

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        appBar: _buildAppBar(context),
        body: LayoutBuilder(
          builder: (context, constraints) {
            // P2-3：横屏（宽度 > 高度）采用全幅波形 + 侧栏信息的双栏布局；
            // 竖屏保持既有单列信息架构。
            final isLandscape = constraints.maxWidth > constraints.maxHeight;
            if (isLandscape) {
              return Row(
                children: [
                  const Expanded(
                    flex: 3,
                    child: Padding(
                      padding: EdgeInsets.all(AppSpacing.md),
                      child: _WaveformArea(),
                    ),
                  ),
                  Expanded(
                    flex: 2,
                    child: SingleChildScrollView(
                      child: Column(
                        children: [
                          Padding(
                            padding: const EdgeInsets.symmetric(
                              horizontal: AppSpacing.md,
                              vertical: AppSpacing.md,
                            ),
                            child: _InfoArea(alarmCount: _totalAlarmCount),
                          ),
                          const _ControlPanel(),
                          const SizedBox(height: AppSpacing.md),
                        ],
                      ),
                    ),
                  ),
                ],
              );
            }
            return Column(
              children: [
                /// 波形显示区
                const Expanded(
                  flex: 4,
                  child: Padding(
                    padding: EdgeInsets.all(AppSpacing.md),
                    child: _WaveformArea(),
                  ),
                ),
                /// 信息面板
                Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: AppSpacing.md,
                  ),
                  child: _InfoArea(alarmCount: _totalAlarmCount),
                ),
                /// 底部控制区
                const _ControlPanel(),
                const SizedBox(height: AppSpacing.md),
              ],
            );
          },
        ),
      ),
    );
  }

  PreferredSizeWidget _buildAppBar(BuildContext context) {
    return AppBar(
      title: const Text(
        'ESP32-ECG 心电监测',
        style: TextStyle(fontSize: AppFontSize.xxxl, fontWeight: FontWeight.w500),
      ),
      centerTitle: true,
      backgroundColor: AppColors.surface,
      elevation: 0,
      actions: [
        /// 记录管理
        IconButton(
          icon: const Icon(Icons.cloud_download),
          tooltip: '记录管理',
          onPressed: () => Navigator.push(
            context,
            MaterialPageRoute(
              builder: (_) => RecordListPage(
                api: RecordApi(),
                // P0-4：连接后向设备发 STATUS，元数据从固件读取（单一真值）。
                deviceMetadataProbe: () => _probeDeviceStatus(),
              ),
            ),
          ),
        ),
        /// 告警设置
        IconButton(
          icon: const Icon(Icons.settings),
          tooltip: '告警设置',
          onPressed: () => showAlarmSettingsSheet(context, _settings),
        ),
        /// 告警历史（含计数徽章）
        IconButton(
          icon: _alarmHistory.isEmpty
              ? const Icon(Icons.history)
              : Badge(
                  label: Text('${_alarmHistory.length}'),
                  child: const Icon(Icons.history),
                ),
          tooltip: '告警历史',
          onPressed: () => showHistorySheet(
            context,
            _alarmHistory,
            () async {
              await _historyStore.clear();
              if (mounted) setState(() => _alarmHistory = []);
            },
          ),
        ),
        /// 关于
        IconButton(
          icon: const Icon(Icons.info_outline),
          onPressed: () => _showAbout(context),
        ),
      ],
    );
  }

  /// P0-4：向设备请求 STATUS 行，供上传元数据读取 firmware_version/model。
  /// 未连接/超时返回 null（页面会用构建期兜底值）。
  Future<String?> _probeDeviceStatus() async {
    final ble = _ecgProvider.bleService;
    if (!ble.isConnected) return null;
    final completer = Completer<String?>();
    void onStatus(String line) {
      if (!completer.isCompleted) completer.complete(line);
    }

    final prev = ble.onStatus;
    ble.onStatus = onStatus;
    try {
      await ble.sendCommand('STATUS');
      return await completer.future
          .timeout(const Duration(seconds: 2), onTimeout: () => null);
    } catch (_) {
      return null;
    } finally {
      ble.onStatus = prev;
    }
  }

  void _showAbout(BuildContext context) {
    showAboutDialog(
      context: context,
      applicationName: 'ESP32-ECG 心电监测',
      applicationVersion: 'v1.0.0',
      children: [
        const Text(
          '实时心电波形监测 App\n\n'
          '连接 ESP32-ECG 设备后，实时显示滤波后的心电波形。\n\n'
          '功能：\n'
          '  速度调节：1s / 2s / 4s / 6s 时间窗口\n'
          '  幅度调节：0.5x / 1x / 2x / 3x 垂直缩放\n\n'
          '波形颜色：青蓝色（滤波后信号）',
        ),
      ],
    );
  }
}

class _WaveformArea extends StatelessWidget {
  const _WaveformArea();

  @override
  Widget build(BuildContext context) {
    return Consumer<ECGProvider>(
      builder: (context, provider, _) {
        return Container(
          decoration: BoxDecoration(
            border: Border.all(color: AppColors.surfaceVariant),
            borderRadius: AppRadius.mdAll,
          ),
          child: ClipRRect(
            borderRadius: AppRadius.smAll,
            child: Stack(
              fit: StackFit.expand,
              children: [
                ECGWaveform(provider: provider),
                // P2-3 三态之「未连接引导态」：无数据时覆盖连接指引，
                // 而非让用户面对一片空白网格。
                if (provider.samples.isEmpty &&
                    !provider.isConnected &&
                    !provider.isScanning)
                  const _ConnectGuide(),
                // P2-3 三态之「加载态」：扫描/连接进行中显示 spinner。
                if (provider.isScanning) const _LoadingOverlay(),
              ],
            ),
          ),
        );
      },
    );
  }
}

/// 未连接引导态：波形区覆盖的连接指引（P2-3）。
class _ConnectGuide extends StatelessWidget {
  const _ConnectGuide();

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(
              Icons.bluetooth_searching,
              size: 40,
              color: AppColors.disabled,
            ),
            const SizedBox(height: AppSpacing.md),
            Text(
              '尚未连接设备',
              style: TextStyle(
                color: AppColors.textSecondary,
                fontSize: AppFontSize.lg,
              ),
            ),
            const SizedBox(height: AppSpacing.sm),
            Text(
              '请确认设备已开机',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: AppColors.traceLabel,
                fontSize: AppFontSize.sm,
                height: 1.6,
              ),
            ),
            Text(
              '然后点击下方「连接」',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: AppColors.traceLabel,
                fontSize: AppFontSize.sm,
                height: 1.6,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _InfoArea extends StatelessWidget {
  final int alarmCount;
  const _InfoArea({required this.alarmCount});

  @override
  Widget build(BuildContext context) {
    return Consumer<ECGProvider>(
      builder: (context, provider, _) {
        return InfoPanel(provider: provider, alarmCount: alarmCount);
      },
    );
  }
}

class _ControlPanel extends StatelessWidget {
  const _ControlPanel();

  @override
  Widget build(BuildContext context) {
    return Consumer<ECGProvider>(
      builder: (context, provider, _) {
        final btnStyle = (bool isActive) => ElevatedButton.styleFrom(
          backgroundColor: isActive ? AppColors.primary : AppColors.surfaceVariant,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(
            horizontal: AppSpacing.lg,
            vertical: AppSpacing.ms,
          ),
          // P2-3: 触达目标 >= 48 dp（修复此前 Size.zero）
          minimumSize: const Size(0, AppSizes.minTouchTarget),
          shape: const RoundedRectangleBorder(borderRadius: AppRadius.smAll),
        );

        return Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
          child: Column(
            children: [
              // 第一行：连接/断开 + 清空
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: provider.isScanning
                          ? null
                          : () => _toggleConnection(context, provider),
                      icon: Icon(
                        provider.isConnected
                            ? Icons.bluetooth_disabled
                            : Icons.bluetooth_searching,
                        size: 18,
                      ),
                      label: Text(
                        provider.isScanning
                            ? '扫描中...'
                            : provider.isConnected
                                ? '断开'
                                : '连接',
                        style: const TextStyle(fontSize: AppFontSize.md),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: provider.isConnected
                            ? Colors.red.withValues(alpha: 0.8)
                            : AppColors.primary,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 10),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  // 状态文字
                  if (provider.statusMessage != '未连接' &&
                      provider.statusMessage != '已连接')
                    Flexible(
                      child: Text(
                        provider.statusMessage,
                        style: const TextStyle(color: Colors.orange, fontSize: AppFontSize.xs),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  const SizedBox(width: 4),
                  IconButton(
                    onPressed: provider.samples.isNotEmpty ? () => provider.clear() : null,
                    icon: const Icon(Icons.clear_all, size: 20),
                    tooltip: '清空',
                    color: Colors.grey,
                    constraints: const BoxConstraints(minWidth: 36),
                    padding: EdgeInsets.zero,
                  ),
                ],
              ),
              const SizedBox(height: 6),
              // 第二行：速度控制
              Row(
                children: [
                  const SizedBox(
                    width: 42,
                    child: Text('速度', style: TextStyle(color: Colors.grey, fontSize: AppFontSize.sm)),
                  ),
                  Expanded(
                    child: Row(
                      children: [1, 2, 4, 6].map((sec) {
                        final isActive = provider.timeWindow == sec;
                        return Expanded(
                          child: Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 2),
                            child: ElevatedButton(
                              onPressed: () => provider.timeWindow = sec,
                              style: btnStyle(isActive),
                              child: Text('${sec}s', style: const TextStyle(fontSize: AppFontSize.sm)),
                            ),
                          ),
                        );
                      }).toList(),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 4),
              // 第三行：幅度控制
              Row(
                children: [
                  const SizedBox(
                    width: 42,
                    child: Text('幅度', style: TextStyle(color: Colors.grey, fontSize: AppFontSize.sm)),
                  ),
                  Expanded(
                    child: Row(
                      children: [0.5, 1.0, 2.0, 3.0].map((scale) {
                        final isActive = (provider.amplitudeScale - scale).abs() < 0.01;
                        return Expanded(
                          child: Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 2),
                            child: ElevatedButton(
                              onPressed: () => provider.amplitudeScale = scale,
                              style: btnStyle(isActive),
                              child: Text('${scale}x', style: const TextStyle(fontSize: AppFontSize.sm)),
                            ),
                          ),
                        );
                      }).toList(),
                    ),
                  ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }

  /// P2-3：连接流程改为「先扫描列设备，再由用户选择」，不再自动抢第一台。
  void _toggleConnection(BuildContext context, ECGProvider provider) async {
    if (provider.isConnected) {
      await provider.disconnect();
      return;
    }

    final candidates = await provider.scanForCandidates();
    if (!context.mounted) return;

    if (candidates.isEmpty) {
      // 错误态：扫描无结果给出明确文案与重试入口（P2-3）。
      showDialog<void>(
        context: context,
        builder: (ctx) => AlertDialog(
          backgroundColor: AppColors.surface,
          title: const Text('未发现设备'),
          content: const Text(
            '请确认：\n'
            '1. 设备已开机（指示灯亮）\n'
            '2. 手机蓝牙已开启\n'
            '3. 距离设备 5 米以内',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('取消'),
            ),
            TextButton(
              onPressed: () {
                Navigator.pop(ctx);
                _toggleConnection(context, provider);
              },
              child: const Text('重试'),
            ),
          ],
        ),
      );
      return;
    }

    // 设备列表选择（含 RSSI），用户显式点选。
    final picked = await showModalBottomSheet<BleCandidate>(
      context: context,
      backgroundColor: AppColors.surface,
      builder: (ctx) => _DevicePickerSheet(candidates: candidates),
    );
    if (picked == null) return; // 用户取消

    await provider.connectToCandidate(picked);
  }
}

/// P2-3：设备选择底部弹窗。
///
/// 分两组显示：
///   - 「已配对」：系统里已配对/已连接的设备（BLE 已配对设备不再广播，
///     只能从这里连；这是连板子的主通道）；
///   - 「扫描到」：本次广播扫描发现的设备（尚未配对的）。
class _DevicePickerSheet extends StatelessWidget {
  final List<BleCandidate> candidates;

  const _DevicePickerSheet({required this.candidates});

  /// 已配对候选（未广播，rssi == 0）。
  List<BleCandidate> get _paired =>
      candidates.where((c) => c.rssi == 0).toList();

  /// 扫描到的候选（有 RSSI）。
  List<BleCandidate> get _scanned =>
      candidates.where((c) => c.rssi != 0).toList();

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Padding(
            padding: const EdgeInsets.all(AppSpacing.lg),
            child: Text(
              '选择设备（共 ${candidates.length} 台）',
              style: TextStyle(
                color: AppColors.textSecondary,
                fontSize: AppFontSize.xxl,
              ),
            ),
          ),
          Flexible(
            child: ListView(
              shrinkWrap: true,
              children: [
                if (_paired.isNotEmpty) ...[
                  _sectionHeader('已配对（推荐）'),
                  ..._paired.map((c) => _tile(context, c, paired: true)),
                ],
                if (_scanned.isNotEmpty) ...[
                  _sectionHeader('扫描到'),
                  ..._scanned.map((c) => _tile(context, c, paired: false)),
                ],
              ],
            ),
          ),
          const SizedBox(height: AppSpacing.md),
        ],
      ),
    );
  }

  Widget _sectionHeader(String text) => Padding(
        padding: const EdgeInsets.fromLTRB(
          AppSpacing.lg,
          AppSpacing.md,
          AppSpacing.lg,
          AppSpacing.sm,
        ),
        child: Text(
          text,
          style: TextStyle(
            color: AppColors.primary,
            fontSize: AppFontSize.sm,
            fontWeight: FontWeight.w500,
          ),
        ),
      );

  Widget _tile(BuildContext context, BleCandidate c, {required bool paired}) {
    return ListTile(
      leading: Icon(
        paired ? Icons.bluetooth_connected : Icons.bluetooth,
        color: AppColors.primary,
      ),
      title: Text(
        c.name,
        style: const TextStyle(color: Colors.white),
      ),
      subtitle: Text(
        c.id,
        style: TextStyle(
          color: AppColors.textSecondary,
          fontSize: AppFontSize.xs,
        ),
      ),
      trailing: Text(
        paired ? '已配对' : c.rssiLabel,
        style: AppText.numeric(
          AppFontSize.sm,
          color: AppColors.traceLabel,
        ),
      ),
      onTap: () => Navigator.pop(context, c),
    );
  }
}

/// P2-3 三态之「加载态」：扫描/连接进行中的覆盖指示。
class _LoadingOverlay extends StatelessWidget {
  const _LoadingOverlay();

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: Container(
        color: AppColors.traceBackground.withValues(alpha: 0.6),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const SizedBox(
                width: 28,
                height: 28,
                child: CircularProgressIndicator(
                  strokeWidth: 2.5,
                  color: AppColors.primary,
                ),
              ),
              const SizedBox(height: AppSpacing.lg),
              Text(
                '正在扫描设备...',
                style: TextStyle(
                  color: AppColors.textSecondary,
                  fontSize: AppFontSize.lg,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}