import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import 'providers/ecg_provider.dart';
import 'providers/settings_provider.dart';
import 'services/alarm_sound_service.dart';
import 'services/alarm_history_store.dart';
import 'models/alarm_event.dart';
import 'widgets/ecg_waveform.dart';
import 'widgets/info_panel.dart';
import 'widgets/alarm_dialog.dart';
import 'widgets/history_sheet.dart';
import 'widgets/settings_sheet.dart';
import 'services/record_api.dart';
import 'pages/record_list_page.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
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
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0D0D1A),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF00BFFF),
        ),
      ),
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
    _ecgProvider.addListener(_onEcgChange);
  }

  @override
  void dispose() {
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
        body: Column(
          children: [
            /// 波形显示区
            const Expanded(
              flex: 4,
              child: Padding(
                padding: EdgeInsets.all(8.0),
                child: _WaveformArea(),
              ),
            ),
            /// 信息面板
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 8.0),
              child: _InfoArea(alarmCount: _totalAlarmCount),
            ),
            /// 底部控制区
            const _ControlPanel(),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
  }

  PreferredSizeWidget _buildAppBar(BuildContext context) {
    return AppBar(
      title: const Text(
        'ESP32-ECG 心电监测',
        style: TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
      ),
      centerTitle: true,
      backgroundColor: const Color(0xFF1A1A2E),
      elevation: 0,
      actions: [
        /// 记录管理
        IconButton(
          icon: const Icon(Icons.cloud_download),
          tooltip: '记录管理',
          onPressed: () => Navigator.push(
            context,
            MaterialPageRoute(
              builder: (_) => RecordListPage(api: RecordApi()),
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
            border: Border.all(color: const Color(0xFF2A2A3E)),
            borderRadius: BorderRadius.circular(8),
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(7),
            child: ECGWaveform(provider: provider),
          ),
        );
      },
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
          backgroundColor: isActive ? const Color(0xFF00BFFF) : const Color(0xFF2A2A3E),
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          minimumSize: Size.zero,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
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
                        style: const TextStyle(fontSize: 13),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: provider.isConnected
                            ? Colors.red.withValues(alpha: 0.8)
                            : const Color(0xFF00BFFF),
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
                        style: const TextStyle(color: Colors.orange, fontSize: 11),
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
                    child: Text('速度', style: TextStyle(color: Colors.grey, fontSize: 12)),
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
                              child: Text('${sec}s', style: const TextStyle(fontSize: 12)),
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
                    child: Text('幅度', style: TextStyle(color: Colors.grey, fontSize: 12)),
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
                              child: Text('${scale}x', style: const TextStyle(fontSize: 12)),
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

  void _toggleConnection(BuildContext context, ECGProvider provider) async {
    if (provider.isConnected) {
      await provider.disconnect();
    } else {
      await provider.connect();
    }
  }
}
