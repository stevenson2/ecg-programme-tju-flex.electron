# R 峰触发固件改动计划 / 已改内容（2026-10-20）

## 已落地代码（未刷机）

1. `components/ecg_ai/include/ecg_ai.h`
   - `ECG_AI_TRIGGER_BLIND` / `ECG_AI_TRIGGER_RPEAK`
   - `cfg.trigger_mode`, `cfg.rpeak_half_win`
   - `ecg_ai_on_rpeak()`
2. `components/ecg_ai/src/ecg_ai.cpp`
   - R 峰预约：`rpeak_fire_at = sample_count + 125`
   - 盲切路径仅在 `trigger_mode==BLIND` 走 stride
   - 不应期 ~600 ms（150 样本 @250 Hz）
3. `main/main.cc`
   - init：`cfg.trigger_mode = ECG_AI_TRIGGER_RPEAK`
   - `hr.beatDetected` → `ecg_ai_on_rpeak()`

备份：同目录 `*.bak_rpeak_20261020`

## 刷机前检查

- [ ] `idf.py build` 通过
- [ ] 静坐：AI_RESULT 节奏应贴合心率（约每拍一次），不是严格 1 Hz
- [ ] 拔电极：无拍时 AI 不刷屏；FLAT/停搏规则仍工作
- [ ] 对比同段 ECGR：盲切 vs R 峰回放 FP 进入次数

## 回滚

`cfg.trigger_mode = ECG_AI_TRIGGER_BLIND` 或恢复 `.bak_rpeak_20261020`。
