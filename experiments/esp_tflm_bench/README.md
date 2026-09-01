# esp_tflm_bench（PC/ESP32 一致性验证）

- `prepare_consistency.py`：从 `mit_deploy_causal_match.npz` 选 200 拍（100 N + 100 A，seed 42），
  用 PC TFLite BUILTIN_REF 跑 exp7c INT8，生成 `consistency_pc.json` 和 `main/test_vectors.h`。
- `analyze_consistency.py`：解析板上 `RESULT,` 日志，生成 `consistency_result.json`。
- 板上可执行：`main/main.cc` 先跑 200 拍一致性，再跑 `ecg_ai` 组件 smoke test，最后 BENCH。
- 参考日志：`C:\esp\esp_tflm_bench\monitor_component2.log`。

## 关键数字（2026-08-24）

- mean |Δp| = 0.000625
- max |Δp| = 0.027344
- |ΔAUC| = 0.00015
- 单次推理平均 49.0ms
