import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'providers/ecg_provider.dart';
import 'widgets/ecg_waveform.dart';
import 'widgets/info_panel.dart';

/**
 * @file main.dart
 * @brief ESP32-ECG 心电监测 App 入口
 *
 * 功能：
 * - 通过 BLE NUS 连接 ESP32-ECG 设备
 * - 实时绘制三通道心电波形（绿/红/蓝）
 * - 显示心率和信号质量
 * - 支持界面缩放操作
 */

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

/**
 * @brief 主监测界面
 */
class ECGMonitorScreen extends StatelessWidget {
  const ECGMonitorScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        appBar: _buildAppBar(context),
        body: const Column(
          children: [
            // 波形显示区（占大部分空间）
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
            // 底部按钮栏
            _ButtonBar(),
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
          '连接 ESP32-ECG 设备后实时显示三通道心电波形。\n\n'
          '🟢 绿色 = 纯净心电信号\n'
          '🔴 红色 = 原始采集信号（含噪声）\n'
          '🔵 蓝色 = 滤波后信号\n\n'
          '黄色竖线 = 200ms 延迟参考线',
        ),
      ],
    );
  }
}

/**
 * @brief 波形显示区域
 */
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

/**
 * @brief 信息面板区域
 */
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

/**
 * @brief 底部控制按钮栏
 */
class _ButtonBar extends StatelessWidget {
  const _ButtonBar();

  @override
  Widget build(BuildContext context) {
    return Consumer<ECGProvider>(
      builder: (context, provider, _) {
        return Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 4),
          child: Row(
            children: [
              // 连接/断开按钮
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: provider.isScanning
                      ? null
                      : () => _toggleConnection(context, provider),
                  icon: Icon(
                    provider.isConnected
                        ? Icons.bluetooth_disabled
                        : Icons.bluetooth_searching,
                    size: 20,
                  ),
                  label: Text(
                    provider.isScanning
                        ? '扫描中...'
                        : provider.isConnected
                            ? '断开'
                            : '连接 ESP32',
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: provider.isConnected
                        ? Colors.red.withOpacity(0.8)
                        : const Color(0xFF00BFFF),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              // 清空按钮
              IconButton(
                onPressed: provider.samples.isNotEmpty ? () => provider.clear() : null,
                icon: const Icon(Icons.clear_all),
                tooltip: '清空波形',
                color: Colors.grey,
              ),
              // 状态信息
              if (provider.statusMessage != '未连接' &&
                  provider.statusMessage != '已连接')
                Expanded(
                  child: Text(
                    provider.statusMessage,
                    style: const TextStyle(color: Colors.orange, fontSize: 12),
                    textAlign: TextAlign.center,
                  ),
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
