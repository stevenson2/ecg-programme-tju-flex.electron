import 'dart:async';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../models/waveform_data_source.dart';
import '../services/ecg_record_codec.dart';
import '../widgets/ecg_waveform.dart';

/**
 * @file playback_page.dart
 * @brief .ecgr 记录本地回放页（静默回放，不触发告警）
 *
 * 回放数据源 PlaybackProvider：轻量 ChangeNotifier，实现 WaveformDataSource，
 * 由 Timer 驱动按记录采样率（250Hz → 4ms/样本）逐样本喂入内部环形缓冲区，
 * 波形经 ECGWaveform 自身 CustomPainter 重绘。
 *
 * 与实时模式的解耦（决策记录）：
 * - 不继承 ECGProvider：其 _addSample 为库私有无法覆写，且会携带 BLE 服务
 *   与告警状态机（回放必须静默，hasAbnormalAlert 恒为 false）；
 * - 波形组件参数类型放宽为 WaveformDataSource（接口位于 ecg_waveform.dart），
 *   ECGProvider 结构上天然满足，无需任何改动；
 * - 页面对外只接收已解码的 EcgRecord，不负责文件读取（由上一级页面接线）。
 *
 * 暗色主题令牌与主界面一致：背景 0xFF0D0D1A / 卡片 0xFF1A1A2E /
 * 主色 0xFF00BFFF / 异常红 0xFFE53935。
 */

/// 回放数据源：按采样率逐样本喂入环形缓冲区，驱动波形重绘
class PlaybackProvider extends ChangeNotifier implements WaveformDataSource {
  static const int kBufferSize = 1500; // 与实时模式一致（6 秒 @250Hz）

  final EcgRecord _record;

  /// 电压环形缓冲区（最新 kBufferSize 点）
  final List<double> _buffer = [];

  /// 下一个待播放样本索引
  int _cursor = 0;

  bool _playing = false;
  Timer? _timer;

  /// 显示窗口（秒）：回放默认 4 秒，便于观察异常段上下文
  final int _timeWindow = 4;

  PlaybackProvider(this._record);

  // ── 回放状态 ──

  bool get isPlaying => _playing;
  int get cursor => _cursor;
  int get totalSamples => _record.totalSamples;
  int get durationSec => _record.durationSec;

  /// 当前播放到的秒
  int get currentSecond =>
      _record.sampleRate > 0 ? _cursor ~/ _record.sampleRate : 0;

  /// 是否已到末尾
  bool get isAtEnd => _cursor >= _record.totalSamples;

  /// 当前秒是否被固件位图标记为异常
  bool get isAbnormalSecond {
    if (!_record.hasBitmap) return false;
    final sec = currentSecond;
    return sec >= 0 &&
        sec < _record.abnormalBySecond.length &&
        _record.abnormalBySecond[sec] == 1;
  }

  // ── WaveformDataSource 实现 ──

  @override
  List<double> get displayData {
    final visible = timeWindow * _record.sampleRate;
    final start = _buffer.length > visible ? _buffer.length - visible : 0;
    return _buffer.sublist(start);
  }

  @override
  double get maxValue {
    final (rawMax, rawMin) = _visibleRange;
    final center = (rawMax + rawMin) / 2;
    final halfRange = ((rawMax - rawMin) / 2).clamp(0.1, 10.0);
    return center + halfRange;
  }

  @override
  double get minValue {
    final (rawMax, rawMin) = _visibleRange;
    final center = (rawMax + rawMin) / 2;
    final halfRange = ((rawMax - rawMin) / 2).clamp(0.1, 10.0);
    return center - halfRange;
  }

  @override
  int get timeWindow => _timeWindow;

  /// 回放为静默审查：永不进入告警态，波形保持正常颜色
  @override
  bool get hasAbnormalAlert => false;

  /// 当前显示窗口原始极值（与 ECGProvider 口径一致）
  (double, double) get _visibleRange {
    if (_buffer.isEmpty) return (1.0, -0.2);
    final visible = timeWindow * _record.sampleRate;
    final start = _buffer.length > visible ? _buffer.length - visible : 0;
    double max = -999, min = 999;
    for (int i = start; i < _buffer.length; i++) {
      final v = _buffer[i];
      if (v > max) max = v;
      if (v < min) min = v;
    }
    return (max == -999 ? 1.0 : max, min == 999 ? -0.2 : min);
  }

  // ── 播放控制 ──

  void play() {
    if (_playing) return;
    if (isAtEnd) {
      _seekCursor(0); // 末尾重播：从头开始
    }
    _playing = true;
    _startTimer();
    notifyListeners();
  }

  void pause() {
    if (!_playing) return;
    _timer?.cancel();
    _timer = null;
    _playing = false;
    notifyListeners();
  }

  void toggle() => _playing ? pause() : play();

  /// 跳转到指定秒（滑块拖动），样本索引 = second × sampleRate
  void seekToSecond(int second) {
    final s = second.clamp(0, _record.durationSec);
    _seekCursor(s * _record.sampleRate);
    notifyListeners();
  }

  void _seekCursor(int target) {
    _cursor = target.clamp(0, _record.totalSamples);
    _rebuildBuffer();
  }

  /// 按记录采样率启动定时器（250Hz → 4ms；其他采样率自适应）
  void _startTimer() {
    final intervalMs = _record.sampleRate > 0
        ? (1000 / _record.sampleRate).round()
        : 4;
    _timer = Timer.periodic(Duration(milliseconds: intervalMs), (_) {
      _feedNext();
    });
  }

  /// 每 tick 推入一个样本；到末尾自动停止（取消定时器）
  void _feedNext() {
    if (_cursor >= _record.totalSamples) {
      _stopAndNotify();
      return;
    }
    _push(_record.samplesV[_cursor]);
    _cursor++;
    if (_cursor >= _record.totalSamples) {
      _stopAndNotify();
    } else {
      notifyListeners(); // 每样本一次通知，波形随播放实时推进
    }
  }

  void _stopAndNotify() {
    _timer?.cancel();
    _timer = null;
    _playing = false;
    notifyListeners();
  }

  void _push(double v) {
    if (_buffer.length >= kBufferSize) {
      _buffer.removeAt(0);
    }
    _buffer.add(v);
  }

  /// seek 后重建缓冲区：回填 cursor 前最多 kBufferSize 个样本
  void _rebuildBuffer() {
    final start = max(0, _cursor - kBufferSize);
    _buffer.clear();
    for (int i = start; i < _cursor; i++) {
      _buffer.add(_record.samplesV[i]);
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    _timer = null;
    super.dispose();
  }
}

/// 记录回放页：接收已解码的 EcgRecord，本地静默回放
class PlaybackPage extends StatefulWidget {
  final EcgRecord record;

  const PlaybackPage({super.key, required this.record});

  @override
  State<PlaybackPage> createState() => _PlaybackPageState();
}

class _PlaybackPageState extends State<PlaybackPage> {
  late final PlaybackProvider _provider;

  @override
  void initState() {
    super.initState();
    _provider = PlaybackProvider(widget.record);
  }

  @override
  void dispose() {
    _provider.dispose(); // 取消播放定时器，测试/退出时无 pending timer
    super.dispose();
  }

  /// 'mm:ss' 时间标签（分钟可超 59，故手动格式化而非 DateFormat）
  static String _fmt(int totalSec) {
    final m = totalSec ~/ 60;
    final s = totalSec % 60;
    return '${m.toString().padLeft(2, '0')}:${s.toString().padLeft(2, '0')}';
  }

  /// 记录 ID：起始时间戳 → 本地日期时间
  static String _fmtId(int unixTime) => DateFormat('yyyy-MM-dd HH:mm')
      .format(DateTime.fromMillisecondsSinceEpoch(unixTime * 1000));

  @override
  Widget build(BuildContext context) {
    final record = widget.record;
    return Scaffold(
      backgroundColor: const Color(0xFF0D0D1A),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1A1A2E),
        elevation: 0,
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              '记录回放',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
            ),
            Text(
              '${_fmtId(record.startUnixTime)} · 时长 ${_fmt(record.durationSec)}',
              style: const TextStyle(fontSize: 12, color: Colors.white70),
            ),
          ],
        ),
      ),
      body: SafeArea(
        child: Column(
          children: [
            /// 波形区（含异常角标）
            Expanded(
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: ListenableBuilder(
                  listenable: _provider,
                  builder: (context, _) {
                    return Stack(
                      children: [
                        Container(
                          decoration: BoxDecoration(
                            border: Border.all(color: const Color(0xFF2A2A3E)),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(7),
                            child: ECGWaveform(provider: _provider),
                          ),
                        ),
                        /// 异常秒红色标识（仅当前秒被位图标记时出现）
                        Positioned(
                          top: 14,
                          right: 14,
                          child: _AbnormalChip(
                            visible: _provider.isAbnormalSecond,
                          ),
                        ),
                      ],
                    );
                  },
                ),
              ),
            ),
            /// 控制行：播放/暂停 + 位置滑块 + 时间标签
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 4.0),
              child: ListenableBuilder(
                listenable: _provider,
                builder: (context, _) {
                  final maxSec = record.durationSec.toDouble();
                  return Row(
                    children: [
                      IconButton(
                        onPressed: _provider.toggle,
                        icon: Icon(
                          _provider.isPlaying ? Icons.pause : Icons.play_arrow,
                        ),
                        tooltip: _provider.isPlaying ? '暂停' : '播放',
                        color: const Color(0xFF00BFFF),
                      ),
                      Expanded(
                        child: Slider(
                          value: _provider.currentSecond.toDouble().clamp(0.0, maxSec),
                          min: 0,
                          max: maxSec,
                          divisions: record.durationSec > 0 ? record.durationSec : null,
                          onChanged: record.durationSec > 0
                              ? (v) => _provider.seekToSecond(v.round())
                              : null,
                        ),
                      ),
                      Text(
                        '${_fmt(_provider.currentSecond)} / ${_fmt(record.durationSec)}',
                        style: const TextStyle(fontSize: 12, color: Colors.white70),
                      ),
                    ],
                  );
                },
              ),
            ),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
  }
}

/// 异常秒角标：红色圆角胶囊，无告警弹窗/声音（静默回放）
class _AbnormalChip extends StatelessWidget {
  final bool visible;

  const _AbnormalChip({required this.visible});

  @override
  Widget build(BuildContext context) {
    if (!visible) return const SizedBox.shrink();
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: const Color(0xFFE53935),
        borderRadius: BorderRadius.circular(12),
      ),
      child: const Text(
        '异常',
        style: TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}
