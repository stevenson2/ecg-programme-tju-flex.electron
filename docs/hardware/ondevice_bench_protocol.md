# ESP32-S3 板上基准测试协议

> **版本**: v1.0 | **日期**: 2026-08-03 | **执行人**: 学生（硬件操作） | **编写**: Agent（协议规范）
>
> **前提**: 固件已编译并烧录 (`pio run -t upload`)，AI 推理模块正常工作。
> 本文档提供精确的操作步骤；所有测量值由学生记录，Agent 不接触硬件（AGENTS.md §2）。

---

## 目录

1. [准备](#1-准备)
2. [测量一：推理延迟](#2-测量一推理延迟)
3. [测量二：功耗](#3-测量二功耗)
4. [测量三：INT8 一致性](#4-测量三int8-一致性)
5. [测量四：温升曲线](#5-测量四温升曲线)
6. [数据汇总](#6-数据汇总)
7. [附录：固件修改方案](#7-附录固件修改方案)

---

## 1. 准备

### 1.1 硬件

| 项目 | 要求 |
|------|------|
| 开发板 | ESP32-S3-SUPERMINI (ESP32S3FH4R2) |
| USB 线 | 数据线（连接 PC 串口监视器） |
| 万用表 | 支持 mA/μA 电流测量（或 USB 功率计） |
| 可选: USB 功率计 | 如 UM25C / FNB38 / Power-Z 等，USB-A 公-母直插式 |

### 1.2 软件

- **串口监视器**: PlatformIO `pio device monitor -b 115200` 或 Arduino IDE Serial Monitor
- **日志捕获**: 推荐使用 `pio device monitor -b 115200 > output.csv` 或 PuTTY 日志保存

### 1.3 板上串口命令参考

| 按键 | 功能 |
|------|------|
| `t` | 打印温度状态 (`[温度] 当前: X.X°C ...`) |
| `a` | 打印 AI 统计 (`[AI] 推理统计 \| 总次数: N \| 异常: M \| 平均延迟: X us`) |
| `c` | 打印 CPU 频率 (`[系统] CPU 当前频率: 240 MHz`) |
| `m` | 切换模拟/真实 AFE 模式 |

### 1.4 固件状态确认

**关键问题**: 当前固件 `ai_inference.cpp` 的延迟计时包含了 `memcpy` + `preprocess_samples` + `fill_input_tensor` 在内（`t_start` 在 `run_single_inference` 开头，`t_elapsed` 在 `Invoke()` 之后）。本次基准测试需要 **纯推理延迟**（仅 `g_interpreter->Invoke()` 耗时），而非端到端总延迟。

**方案说明**: 见 §7 附录。学生需按 §7.1 修改固件、§7.2 编译确认后烧录，然后执行 §2 测量。若不修改固件，使用现有总延迟数据也可（历史上限 80-120ms ROADMAP 4.2-4 已验证总延迟，数据解释时将注明含预处理）。

---

## 2. 测量一：推理延迟

### 2.1 目标

| 指标 | 取值 |
|------|------|
| 测量量 | 纯 TFLite `Invoke()` 耗时（微秒） |
| 预期范围 | 80–120 ms/窗口（历史记录，含预处理时为上限） |
| 样本量 | ≥100 个推理窗口 |
| 输出 | mean ± std, P95, P99, min, max |

### 2.2 步骤

1. **固件准备**: 按 §7.1 应用延迟补丁，按 §7.2 编译确认通过，烧录。
2. **启动串口监视器**，捕获日志到文件：
   ```powershell
   # 当前终端 (端口号以实际为准, 如 COM3)
   pio device monitor -b 115200 --port COM3 > ondevice_latency_20260803.csv
   ```
3. **等待预热**: 上电后等待 5 秒，AI 推理稳定运行（观察串口 CSV 数据流）。
4. **持续运行**: 保持设备运行 ≥1 分钟（≥120 个推理窗口；推理间隔 0.5s，100 次 ≈50 秒）。
5. **停止捕获**: `Ctrl+C` 结束串口监视器。

### 2.3 数据格式

延迟补丁后的串口输出每行格式：

```
LAT,<inference_count>,<latency_us>
```

示例：
```
LAT,1,84230
LAT,2,91105
LAT,3,87466
...
```

### 2.4 注意事项

- 初次上电前 2–3 次推理延迟可能偏高（冷启动、cache miss），分析时排除前 5 个样本。
- 如果无法修改固件，使用现有 `s` 命令打印的 `ai_inference_stats` 平均延迟（包含预处理）也可接受，但必须在数据备注中标明 "含预处理"。

---

## 3. 测量二：功耗

### 3.1 目标

| 指标 | 取值 |
|------|------|
| 供电电压 | 3.3V（USB 5V → 板上 LDO）|
| 功耗单位 | mW（电流 mA × 3.3V）|
| 测量场景 | 3 态：idle / AI-inference / BLE-active |
| 输出 | 每态电流 (mA) 均值 ± std |

### 3.2 方案选择

提供两种方案，按现场条件选择其一：

#### 方案 A：USB 功率计（推荐，精度高）

1. 将 USB 功率计串接在 USB 线和开发板 USB-C 口之间。
2. 等待功率计稳定显示（室温条件下）。
3. 记录 3 个场景下的电压 (V) 和电流 (mA)：
   - **Idle 基线**: 模拟模式下，串口 `m` 命令发出 AI 推理暂停 → 等待 5 秒
   - **AI 推理活跃**: AI 推理运行中，BLE 暂停
   - **BLE 活跃**: 打开手机 App 连接 BLE NUS，bleTX 数据传输中
4. 每场景稳定后观察 10 秒，记录波动范围。

#### 方案 B：万用表串联（手动，可接受）

1. 断开 USB 5V 线正极，将万用表（mA 档）串联在 5V 路径中。
2. 按方案 A 场景顺序测量。
3. 每场景记录 3–5 个读数取平均。

### 3.3 数据格式

CSV 模板（手工记录下来后写成文件）:

```csv
# ondevice_power_<date>.csv
mode,voltage_v,current_ma,power_mw,notes
idle_baseline,5.12,42.0,215.0,AI推理暂停 模拟器运行
ai_active,,,,
ble_active,,,,
```

### 3.4 注意事项

- BLE TX 功率设置为 +9 dBm（原始最高功率），是主要热源之一（README 温度诊断表）。
- 如果用方案 B，务必在断电状态下连接万用表，再通电读取。
- 如果现场有示波器，可以在 3.3V 轨接分流电阻测量瞬态电流（高级选项，不做硬性要求）。

---

## 4. 测量三：INT8 一致性

### 4.1 目标

验证板载 TFLite Micro INT8 推理与 PC 端 TFLite INT8 推理在**相同输入**下输出的一致性。

| 指标 | 取值 |
|------|------|
| 测试向量 | 从 `eval_deploy_match.py` INT8 阶段选取 200 个代表拍（100 正常 + 100 异常） |
| PC 端 | TFLite Runtime INT8 Interpreter（Python `tf.lite.Interpreter`） |
| 板载端 | ESP32-S3 TFLite Micro INT8 Interpreter |
| 输出 | AUC 排序保持，mean\|Δp\|，max\|Δp\|，一致率 @θ=0.35 / @θ=0.5 |
| 历史参考 | mean\|Δp\|≈0.25（双重 softmax 压缩），\|ΔAUC\|≤0.025（TUNING_HISTORY §13.4） |

### 4.2 方法

**核心思想**: 从 PC 端选取固定的 N 个代表性测试拍 → 在 PC 和 ESP32 两端使用**完全相同的 INT8 模型文件** (`models/deploy_match/exp5_clean_int8.tflite` 或 `p2a_int8.tflite`) 进行推理 → 比对输出。

#### 步骤 A：PC 端生成测试向量和参考输出

1. 运行以下 Python 脚本（在 `pc_tools/ecg_dl/` 下）：
   ```bash
   cd pc_tools/ecg_dl
   python -c "
   import numpy as np
   import tensorflow as tf
   from pathlib import Path

   # Load D3 deployment match cache
   cache = np.load('models/deploy_match/mit_deploy_match.npz')
   beats_d = cache['beats_deploy']   # shape (N, 250)
   labels = cache['labels']

   # Select 100 normal + 100 abnormal (random seed=42)
   rng = np.random.default_rng(42)
   n_idx = np.where(labels == 0)[0]
   a_idx = np.where(labels == 1)[0]
   sel_n = rng.choice(n_idx, min(100, len(n_idx)), replace=False)
   sel_a = rng.choice(a_idx, min(100, len(a_idx)), replace=False)
   sel_idx = np.sort(np.concatenate([sel_n, sel_a]))
   test_inputs = beats_d[sel_idx]   # (200, 250)
   test_labels = labels[sel_idx]

   # Run PC INT8 interpreter
   tflite_path = 'models/deploy_match/exp5_clean_int8.tflite'
   interp = tf.lite.Interpreter(model_path=str(tflite_path))
   interp.allocate_tensors()
   in_det = interp.get_input_details()[0]
   out_det = interp.get_output_details()[0]
   in_scale = in_det['quantization_parameters']['scales'][0]
   in_zp = in_det['quantization_parameters']['zero_points'][0]
   out_scale = out_det['quantization_parameters']['scales'][0]
   out_zp = out_det['quantization_parameters']['zero_points'][0]

   # Quantize to INT8 (firmware style: trunc + clip)
   x_fp32 = test_inputs.astype(np.float32)
   x_q = np.clip(np.trunc(x_fp32 / in_scale + 0.5) + in_zp, -128, 127).astype(np.int8)

   # Per-beat inference (batch=1 matches firmware)
   pc_probs = np.zeros(len(sel_idx), dtype=np.float32)
   for i in range(len(sel_idx)):
       interp.resize_tensor_input(0, [1, 250, 1], strict=False)
       interp.allocate_tensors()
       interp.set_tensor(in_det['index'], x_q[i:i+1].reshape(1, 250, 1))
       interp.invoke()
       y_i8 = interp.get_tensor(out_det['index'])
       y_fp = (y_i8.astype(np.float32) - out_zp) * out_scale
       # Softmax
       e = np.exp(y_fp - np.max(y_fp))
       pc_probs[i] = e[0, 1] / e[0].sum()

   # Save test vectors + PC reference
   np.savez_compressed('models/deploy_match/int8_test_vectors_200.npz',
                        beats_fp32=test_inputs.astype(np.float32),
                        beats_int8=x_q,
                        labels=test_labels,
                        pc_probs=pc_probs,
                        quant_params={'in_scale': float(in_scale), 'in_zp': int(in_zp),
                                      'out_scale': float(out_scale), 'out_zp': int(out_zp)})
   print(f'Test vectors saved: {len(sel_idx)} beats ({np.sum(test_labels==0)} N, {np.sum(test_labels==1)} A)')
   print(f'PC INT8 probs range: [{pc_probs.min():.4f}, {pc_probs.max():.4f}]')
   "
   ```

2. 生成的文件: `models/deploy_match/int8_test_vectors_200.npz`

#### 步骤 B：导出为 ESP32 可用格式

```bash
cd pc_tools/ecg_dl
python -c "
import numpy as np
data = np.load('models/deploy_match/int8_test_vectors_200.npz')

# Export as C header (each vector as 250 int8)
beats = data['beats_int8']
labels = data['labels']
pc_probs = data['pc_probs']

with open('int8_test_vectors.h', 'w') as f:
    f.write('// INT8 test vectors for ESP32-S3 on-device consistency test\n')
    f.write(f'// Generated: 2026-08-03 | {len(beats)} beats\n')
    f.write(f'// Source: mit_deploy_match.npz D3 deployment chain\n\n')
    f.write(f'#define INT8_TEST_COUNT {len(beats)}\n')
    f.write(f'#define INT8_TEST_WINDOW 250\n\n')
    f.write('const int8_t test_vectors_int8[INT8_TEST_COUNT][INT8_TEST_WINDOW] = {\n')
    for i, beat in enumerate(beats):
        f.write('    {')
        f.write(','.join(str(int(b)) for b in beat))
        f.write('}')
        if i < len(beats) - 1:
            f.write(',\n')
        else:
            f.write('\n')
    f.write('};\n\n')
    f.write(f'const int test_labels[{len(beats)}] = {{')
    f.write(','.join(str(int(l)) for l in labels))
    f.write('};\n\n')
    f.write(f'// PC reference probabilities (Softmax output, per-beat)\n')
    f.write(f'const float pc_reference_probs[{len(beats)}] = {{')
    f.write(','.join(f'{p:.8f}f' for p in pc_probs))
    f.write('};\n')
print(f'Exported int8_test_vectors.h: {len(beats)} vectors')
"
```

**或者**，不导出 .h 文件，使用 `ai_deploy_test.ino` 小脚本（推荐，见下一步）。

**推荐方案：使用固件内联测试模式**

在固件中添加一个特殊的 "INT8 一致性测试模式" 命令（串口 `i` 键），固件收到命令后：

1. 使用硬编码的测试向量（仅需 200×250 = 50,000 字节，可放在 Flash 中通过 PROGMEM 存储）
2. 逐个执行推理
3. 串口输出每拍的 logit 值（反量化后，未做 softmax）

**为了方便学生执行，这里给出一个更简单的 USB 直连方案**：

#### 步骤 B-simple：串口回传法（推荐执行）

1. **PC 端**发送测试向量：通过串口发送测试拍数据（每行一个拍，250 个浮点数），ESP32 收到后触发推理并回传结果。
2. **固件修改**：在 `main.cpp` 添加串口命令 `i`，进入 "INT8 测试模式"：
   - 接收一行 250 个逗号分隔的 float
   - 执行 Z-score 归一化 → INT8 量化 → TFLite Invoke → 反量化 → 输出 logit[0], logit[1]
   - 回传格式: `INT8TEST,<beat_index>,<logit_normal>,<logit_abnormal>`
3. **PC 端**脚本遍历所有 200 个测试拍，逐一发送并收集响应，保存到 CSV。

> **学生注意**: 如果固件修改和串口回传实现困难，可以退一步用 §7.1 的方法，将 INT8 一致性简化为 PC-on-PC 验证（PC TFLite FP32 vs PC TFLite INT8），论文中注明 "TFLite Micro 与 TFLite Runtime INT8 语义等价（相同模型权重 + 相同量化方案），PC-on-PC INT8 一致性已验证（见 TUNING_HISTORY §13.4），板上 INT8 结果因硬件条件限制未包含"。

### 4.3 数据格式

若成功执行板载测量：

```csv
# ondevice_int8_<date>.csv
beat_index,label,pc_prob_abnormal,esp32_prob_abnormal,abs_delta
0,0,0.1523,0.1841,0.0318
1,0,0.0412,0.0556,0.0144
2,1,0.7234,0.6841,0.0393
...
```

### 4.4 预期结果

- `|ΔAUC| ≤ 0.025`（TUNING_HISTORY §13.4 已验证）
- `mean|Δp| ≈ 0.25`（双重 softmax 压缩系统偏差）
- `一致率@0.50: 94-98%`, `一致率@0.35: 91-95%`

---

## 5. 测量四：温升曲线

### 5.1 目标

| 指标 | 取值 |
|------|------|
| 测量时长 | 10 分钟（600 秒）连续 AI 推理 |
| 采样周期 | 1 秒（板载 `thermalUpdate()`） |
| 输出 | T_start, T_max, ΔT, 每分钟平均温度 |
| 传感器 | ESP32-S3 内置温度传感器 (`temperatureRead()`) |

### 5.2 步骤

1. **板子放置在通风良好的桌面**（避免封闭空间导致过热）。
2. **固件准备**: 不需要修改；热敏模块已在 `main.cpp` 初始化并每 250 帧（≈1 秒）更新一次。
3. **启动串口监视器**并记录日志（10 分钟）：
   ```powershell
   pio device monitor -b 115200 --port COM3 > ondevice_thermal_20260803.csv
   ```
4. **启动计时器**（手机秒表即可），开始记录。
5. **在 10 分钟内不做任何操作**：不按键、不拔插 USB、不连接 BLE。
6. 10 分钟后 `Ctrl+C` 停止。

### 5.3 数据处理

- 串口日志中提取所有含 `[温度]` 的行（约 600 行）。
- 解析格式: `[温度] 当前: XX.X°C | 平均: XX.X°C | 范围: XX.X~XX.X°C`
- 提取 `当前温度` 字段即可。

**简便提取命令**（PowerShell）：
```powershell
Select-String -Path "ondevice_thermal_20260803.csv" -Pattern "\[温度\] 当前:" |
  ForEach-Object { $_ -replace '.*当前: (\d+\.\d+)°C.*', '$1' } |
  Out-File "ondevice_thermal_20260803_parsed.csv"
```

### 5.4 数据格式

```csv
# ondevice_thermal_<date>.csv
time_s,temperature_c
0,42.3
1,42.5
2,42.8
...
600,48.2
```

### 5.5 注意事项

- 初始温度应接近室温（25-35°C）。如果初始温度 >40°C，说明板子之前运行过，需冷却 5 分钟再开始。
- 如果温度触发过热保护（>65°C），固件会自动降频至 60MHz，此时推理延迟将显著增加 → 在报告中注明。
- BLE 未连接时理论功耗 ~200-300mW，温升应在 5-10°C 范围。如果 BLE 开启，温升将更高（BLE 是主热源）。

---

## 6. 数据汇总

### 6.1 文件清单

| 测试 | 输出文件 | 行数 (估) |
|------|---------|-----------|
| 延迟 | `ondevice_latency_<date>.csv` | ≥100 行 |
| 功耗 | `ondevice_power_<date>.csv` | 3–5 行 |
| INT8 一致性 | `ondevice_int8_<date>.csv` | 200 行 |
| 温升 | `ondevice_thermal_<date>.csv` | 600 行 |

### 6.2 分析脚本

执行完成后，运行:

```bash
cd pc_tools/ecg_dl
python analyze_ondevice_bench.py --latency ondevice_latency_<date>.csv \
                                 --power ondevice_power_<date>.csv \
                                 --int8 ondevice_int8_<date>.csv \
                                 --thermal ondevice_thermal_<date>.csv \
                                 --output ondevice_bench_summary.json
```

脚本输出: 汇总表 (latency dist, power, INT8 agreement, thermal curve stats) → `ondevice_bench_summary.json`。

### 6.3 期望验收标准

| 测试 | 验收标准 |
|------|---------|
| 延迟 | P95 ≤ 120 ms（含预处理）/ P95 ≤ 80 ms（纯推理） |
| 功耗 | AI 推理增量 ≤ 100 mW vs idle |
| INT8 一致性 | \|ΔAUC\| ≤ 0.025, mean\|Δp\| ~0.25, 一致率@0.5 ≥ 94% |
| 温升 | T_max ≤ 65°C（不超过热保护阈值），ΔT ≤ 15°C |

---

## 7. 附录：固件修改方案

### 7.1 延迟测量补丁（PROPOSE — 学生执行）

**文件**: `src/ai_inference/ai_inference.cpp`

**修改位置**: `run_single_inference()` 函数，第 121–141 行。

**当前代码**（第 121–141 行）：
```cpp
static ai_result_t run_single_inference(const float* samples, uint32_t sample_index) {
    ai_result_t result = {0};
    uint32_t t_start = micros();

    float local_buf[AI_WINDOW_SIZE];
    memcpy(local_buf, samples, sizeof(float) * AI_WINDOW_SIZE);
    preprocess_samples(local_buf, AI_WINDOW_SIZE);
    fill_input_tensor(local_buf);

    TfLiteStatus status = g_interpreter->Invoke();
    uint32_t t_elapsed = micros() - t_start;
    // ...
```

**修改为**（添加纯推理计时 + 串口打印）：

```cpp
static ai_result_t run_single_inference(const float* samples, uint32_t sample_index) {
    ai_result_t result = {0};

    float local_buf[AI_WINDOW_SIZE];
    memcpy(local_buf, samples, sizeof(float) * AI_WINDOW_SIZE);
    preprocess_samples(local_buf, AI_WINDOW_SIZE);
    fill_input_tensor(local_buf);

    // === 纯推理延迟计时 (仅 Invoke) ===
    uint32_t t_invoke_start = micros();
    TfLiteStatus status = g_interpreter->Invoke();
    uint32_t t_invoke_elapsed = micros() - t_invoke_start;
    // =================================

    if (status != kTfLiteOk) {
        result.confidence = 0.0f;
        result.is_abnormal = 0;
        result.sample_index = sample_index;
        result.latency_us = t_invoke_elapsed;
        return result;
    }

    float abnormal_conf = parse_output_confidence();

    // === 基准测试: 打印每拍纯推理延迟 (启用 AI_PROFILE_LATENCY 时生效) ===
#ifdef AI_PROFILE_LATENCY
    static uint32_t bench_count = 0;
    bench_count++;
    Serial.print("LAT,");
    Serial.print(bench_count);
    Serial.print(",");
    Serial.println(t_invoke_elapsed);
#endif
    // =====================================================================

    // ...
```

**同时修改 `tflite_settings.h`**，取消注释 `AI_PROFILE_LATENCY`：

```cpp
// 文件: include/ai_inference/tflite_settings.h
// 修改第 59 行附近:

// #define AI_PROFILE_LATENCY          /* 取消注释以记录每次推理耗时 */
// ↓ 改为:
#define AI_PROFILE_LATENCY          /* 基准测试: 每拍打印纯推理延迟 */
```

**注意**: 基准测试完成后，学生应**重新注释 `#define AI_PROFILE_LATENCY`** 并重新烧录，避免正常运行时串口被延迟日志污染。

### 7.2 编译验证

```bash
# 在项目根目录执行
pio run
```

期望输出: `SUCCESS`（无编译错误）。

### 7.3 INT8 测试模式补丁（可选，高级）

如果执行 §4.2 的 "串口回传法"，需要在 `main.cpp` 中添加 `i` 命令的处理逻辑。因复杂度较高（需解析串口浮点数据、执行归一化+量化+推理+反量化并回传），此处省略详细代码；学生可参照现有 `s`/`a`/`t` 命令模式实现，或使用 `INT8_TEST_MODE` 预编译宏控制。

### 7.4 双模型相关的注意事项

当前 ROADMAP 4.2 计划部署双模型（P2A + exp5）。若此时已实施双模型，基准测试需额外注明：
- **功耗**: 报告单模型 vs 双模型功耗增量
- **延迟**: 分别测量两个模型的推理时间（预期每个模型 ~80-120ms，双模型串行总延迟 <250ms）
- **温升**: 双模型时温升预期更高（两个 interpreter 的 tensor arena 共 64KB SRAM）

---

## 变更记录

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-08-03 | v1.0 | 初始版本，4 项测量协议 + 固件修改方案 + CSV 模板 |
