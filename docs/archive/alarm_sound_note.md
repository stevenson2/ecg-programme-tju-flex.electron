> **⚠️ 已归档（2026-08-21）**：报警音效小笔记。非现役文档；现役入口见根目录 README.md 文档导航。

# 告警提示音服务说明（AlarmSoundService）

> 日期：2026-08-08 ｜ 模块：`ecg_app/lib/services/alarm_sound_service.dart`
> 测试：`ecg_app/test/alarm_sound_service_test.dart`（6 用例，TDD：先 RED 后 GREEN）

## 1. 提示音素材

- **文件**：`ecg_app/assets/audio/beep.wav`
- **规格**：44.1 kHz / 16 bit / 单声道 / 时长 0.5 s（44,100 采样点，RIFF 有效）
- **内容**：自合成的 880 Hz 与 1046 Hz 双音（A5/C6 双音交替，短促脉冲式），**无版权**，
  可直接随 App 分发。
- **注册**：已在 `pubspec.yaml` 的 `assets:` 下注册目录 `assets/audio/`。

## 2. 播放链路（audioplayers ^6.7.1）

```
startAlarmLoop()
  └─ _playBeep()
       ├─ player.setVolume(settings.soundVolume)   // 音量取设置，默认 0.8
       ├─ player.play(AssetSource('audio/beep.wav'))
       └─ HapticFeedback.vibrate()                 // 每次实际播报伴随触觉反馈
beep 播放完成
  └─ onPlayerComplete 事件
       └─ 仍处于循环 → Timer(2s) → _playBeep()     // 循环调度，与播放时长解耦
stopAlarmLoop()
  └─ 置 isLooping=false + 取消 Timer + player.stop()
dispose()
  └─ stopAlarmLoop + 取消订阅 + player.dispose()
```

- **循环间隔**：`loopInterval` 构造参数，默认 `Duration(seconds: 2)`，测试可注入。
- **状态暴露**：`isLooping` getter 供 UI/测试判定当前是否处于告警循环。
- **设置联动**（`SettingsProvider`，已实现）：
  - `soundEnabled == false`：`startAlarmLoop` 正常返回但不播放、不振动（免打扰/静音）；
  - `soundVolume`：每次播报前应用，默认 0.8，范围 0.0~1.0。

## 3. 回退路径（播放失败 → 系统提示音）

audioplayers 播放失败（`play()` 抛异常 / `onPlayerComplete` 事件流错误）时：

```
_playFallback() → SystemSound.play(SystemSoundType.alert)
```

保证资源异常时告警仍有可闻提示。实现方式：
- `_playBeep()` 内对 `setVolume`/`play` 用 try/catch 包裹；
- `onPlayerComplete` 订阅带 `onError` 处理器。

## 4. 可测性设计

- 服务构造接受可注入的 `AudioPlayer? player`（测试传入 `FakeAudioPlayer`，
  继承 `AudioPlayer` 并覆写 `play/stop/setVolume/dispose/getCurrentPosition`，
  用"静默 Completer"覆写 `creatingCompleter` 规避测试环境平台通道异常）；
- 循环定时器由 `fakeAsync` 控制虚拟时钟推进，无需等待真实 2 秒；
- `HapticFeedback.vibrate()` 在测试绑定（TestWidgetsFlutterBinding）下为 no-op，
  仅验证不抛错，不断言平台行为。

## 5. 测试用例清单（test/alarm_sound_service_test.dart）

| # | 用例 | 断言要点 |
|---|------|---------|
| (a) | startAlarmLoop 从设置读取音量并播放 | play 1 次、setVolume 以 0.8 调用、isLooping=true |
| (b) | onPlayerComplete 后循环持续 | 完成事件 → 2s 定时器 → 再次播放（play ≥2 次） |
| (c) | stopAlarmLoop 停止且不再重播 | stop 1 次、isLooping=false、之后完成事件不触发重播 |
| (d) | soundEnabled=false 不播放 | play/setVolume 均 0 次 |
| (e) | dispose 清理 | stop 被调用、player.disposed、之后完成事件不触发重播 |
| (f) | 播放失败回退系统提示音 | play 抛异常 → 平台通道捕获 `SystemSound.play` |

## 6. 集成说明（后续任务）

- **未接入** `main.dart`（集成任务另行执行）；服务与 UI 解耦，不持有 BuildContext。
- 集成时建议：告警状态机进入 ALARMING/ARMING 时 `startAlarmLoop()`，
  恢复 IDLE / 用户确认时 `stopAlarmLoop()`，App 退出时 `dispose()`。
