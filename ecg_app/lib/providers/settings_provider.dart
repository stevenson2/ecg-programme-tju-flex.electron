import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/**
 * @file settings_provider.dart
 * @brief 告警设置状态管理（ChangeNotifier + SharedPreferences 持久化）
 *
 * 管理四项告警设置（Contract C2）：
 * - 免打扰 (dndEnabled)：抑制弹窗 + 提示音（历史仍记录）
 * - 提示音 (soundEnabled)：警报提示音开关
 * - 音量 (soundVolume)：0.0 ~ 1.0
 * - 弹窗自动关闭时长 (autoCloseSeconds)：3 ~ 30 秒
 *
 * 无存储时默认值：dnd=false, sound=true, volume=0.8, autoClose=10。
 * 所有 setter 均先更新内存状态、再持久化到 SharedPreferences，
 * 完成后触发 notifyListeners。
 */
class SettingsProvider extends ChangeNotifier {
  // ── SharedPreferences 键名 ──
  static const String _kDndKey = 'alarm_dnd_enabled';
  static const String _kSoundKey = 'alarm_sound_enabled';
  static const String _kVolumeKey = 'alarm_sound_volume';
  static const String _kAutoCloseKey = 'alarm_auto_close_seconds';

  // ── 默认值与边界 ──
  static const bool kDefaultDnd = false;
  static const bool kDefaultSound = true;
  static const double kDefaultVolume = 0.8;
  static const int kDefaultAutoClose = 10;
  static const double kMinVolume = 0.0;
  static const double kMaxVolume = 1.0;
  static const int kMinAutoClose = 3;
  static const int kMaxAutoClose = 30;

  // ── 内部状态（构造时即默认值，load() 前可直接使用）──
  bool _dndEnabled = kDefaultDnd;
  bool _soundEnabled = kDefaultSound;
  double _soundVolume = kDefaultVolume;
  int _autoCloseSeconds = kDefaultAutoClose;

  // ── Getter ──

  /// 免打扰：抑制弹窗 + 提示音（历史仍记录）
  bool get dndEnabled => _dndEnabled;

  /// 提示音开关
  bool get soundEnabled => _soundEnabled;

  /// 提示音音量 0.0 ~ 1.0
  double get soundVolume => _soundVolume;

  /// 弹窗自动关闭秒数（3 ~ 30）
  int get autoCloseSeconds => _autoCloseSeconds;

  // ── 持久化 ──

  /**
   * 从 SharedPreferences 读取全部设置。
   * 缺失的键回落到默认值，音量/时长越界值收敛到边界。
   */
  Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    _dndEnabled = prefs.getBool(_kDndKey) ?? kDefaultDnd;
    _soundEnabled = prefs.getBool(_kSoundKey) ?? kDefaultSound;
    _soundVolume =
        (prefs.getDouble(_kVolumeKey) ?? kDefaultVolume).clamp(
              kMinVolume,
              kMaxVolume,
            ).toDouble();
    _autoCloseSeconds =
        (prefs.getInt(_kAutoCloseKey) ?? kDefaultAutoClose).clamp(
              kMinAutoClose,
              kMaxAutoClose,
            ).toInt();
    notifyListeners();
  }

  /// 设置免打扰并持久化
  Future<void> setDnd(bool v) async {
    _dndEnabled = v;
    await _persist((prefs) => prefs.setBool(_kDndKey, v));
  }

  /// 设置提示音开关并持久化
  Future<void> setSound(bool v) async {
    _soundEnabled = v;
    await _persist((prefs) => prefs.setBool(_kSoundKey, v));
  }

  /// 设置音量（clamp 0.0 ~ 1.0）并持久化
  Future<void> setVolume(double v) async {
    _soundVolume = v.clamp(kMinVolume, kMaxVolume).toDouble();
    await _persist((prefs) => prefs.setDouble(_kVolumeKey, _soundVolume));
  }

  /// 设置弹窗自动关闭秒数（clamp 3 ~ 30）并持久化
  Future<void> setAutoClose(int s) async {
    _autoCloseSeconds = s.clamp(kMinAutoClose, kMaxAutoClose).toInt();
    await _persist((prefs) => prefs.setInt(_kAutoCloseKey, _autoCloseSeconds));
  }

  /// 持久化并通知监听者
  Future<void> _persist(
      Future<void> Function(SharedPreferences prefs) write) async {
    final prefs = await SharedPreferences.getInstance();
    await write(prefs);
    notifyListeners();
  }
}
