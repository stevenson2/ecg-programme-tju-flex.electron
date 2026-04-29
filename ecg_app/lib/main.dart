import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'providers/ecg_provider.dart';
import 'widgets/ecg_waveform.dart';
import 'widgets/info_panel.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);
  runApp(
    ChangeNotifierProvider(
      create: (_) => ECGProvider(),
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

class ECGMonitorScreen extends StatelessWidget {
  const ECGMonitorScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        appBar: _buildAppBar(context),
        body: const Column(
          children: [
            // 波形显示区
            Expanded(
              flex: 4,
              child: Padding(
                padding: EdgeInsets.all(8.0),
                child: _WaveformArea(),
              ),
            ),
            // 信息面板
            Padding(
              padding: EdgeInsets.symmetric(horizontal: 8.0),
              child: _InfoArea(),
            ),
            // 底部控制区
            _ControlPanel(),
            SizedBox(height: 8),
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
  const _InfoArea();

  @override
  Widget build(BuildContext context) {
    return Consumer<ECGProvider>(
      builder: (context, provider, _) {
        return InfoPanel(provider: provider);
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
