> **⚠️ 已归档（2026-08-21）**：SUPERMINI 时代 Flash 预算分析（现行板 N16R8 无容量问题；OR 方案已否决）。非现役文档；现役入口见根目录 README.md 文档导航。

# 双专家 Flash 预算可行性报告（下一步待办 #4, 2026-08-06）

> **⚠️ 存档说明（2026-08-21）**：本报告为 SUPERMINI（4MB Flash）时代的分析存档。
> **现行板卡为 N16R8（16MB Flash / 8MB PSRAM），双模型仅占 ~2%，本文的分区压缩
> 方案已无需执行**；且"双专家 OR"方案本身已被 TH §8.8 否决，现行部署为分模型 +
> 前置关卡。保留本文件仅作历史分析参考。

> **任务来源**：双专家 OR 部署受 Flash 阻塞（原下一步待办 #4），评估
> 压缩/分区方案可行性。
> **验收**：可行性报告（本文件）。
> **结论先行**：4MB SUPERMINI 板上双模型仅超 82.6KB（5.7%），**无需压缩、无需换板**——
> 调整分区表去掉 OTA 双槽即可放下（零硬件成本）；换 N16R8（16MB）为长期无压力路线。

---

## 1. 现状盘点（实测，2026-08-06 `pio run` 后）

| 项 | 值 | 来源 |
|----|----|----|
| 板型 | SUPERMINI（4MB Flash，`adafruit_qtpy_esp32s3_n4r2`） | `platformio.ini` |
| 分区表 | **OTA 双槽**：nvs 20KB + otadata 8KB + **ota_0 1.375MB** + **ota_1 1.375MB** + uf2 256KB + ffat 960KB | `.pio/build/esp32-s3-supermini/partitions.bin` 解析 |
| app 分区容量 | 1,441,792 B（1.375MB，ota_0） | pio run 输出 |
| 当前固件占用 | **1,357,001 B（94.1%）** | pio run 输出 |
| 其中 AI 模型 | 167,376 B（163.5KB，`ecg_model_data.h`） | TFLite 实测 |
| 非模型部分 | 1,189,625 B（~1.13MB：固件+库+其余模块） | 差减 |
| 双模型需求 | 1,189,625 + 2×167,376 = **1,524,377 B** | 计算 |
| **缺口** | **−82,585 B（−80.7KB，超分区 5.7%）** | 计算 |

注：P2A / exp5 / exp6-SGD / KD(a070_t1) 均为 ResNet-L 架构（80K 参数），INT8 TFLite
尺寸相同（163.5KB，实测 `ecg_model_{p2a,exp5,exp6_sgd}_int8.tflite`）；KD 学生模型
同为 ResNet-Large（`train_kd.py`），尺寸相当。→ 双专家组合任何一对，需求相同。

## 2. 方案对比

### 方案 A：分区调整（去 OTA 双槽）—— ✅ 推荐（短期，零成本）

- 做法：自定义 `partitions.csv`，删除 ota_1（1.375MB），app 分区从 0x160000 扩至
  **0x2BF000（2,879,488 B ≈ 2.75MB）**；nvs/otadata/uf2/ffat 布局保留
  （uf2 256KB 为 QTPY 板烧录引导必需，不动）。
- 效果：双模型 1,524,377 B 仅占 app 的 **52.9%**，余 1.35MB（甚至三模型也可放）。
- 代价：失去**无线 OTA**（当前开发用 USB 上传，`board_build.usb_cdc_on_boot=1`，
  无 OTA 需求；后续量产如需 OTA 再评估 8MB+ 板）。
- 风险：低。分区表改动需重新烧录 bootloader 布局（`pio run -t erase` + 全量烧录），
  属开发环境一次操作；固件代码零改动。

### 方案 B：换 ESP32-S3-WROOM-1-N16R8（16MB Flash / 8MB PSRAM）—— ✅ 推荐（长期）

- 效果：app 分区可 ≥8MB，双模型占 <2%，且为未来"云端多类诊断/长程节律"等新模型
  留足空间；PSRAM 8MB 顺带解除 Tensor Arena / 分时加载的内存顾虑。
- 代价：硬件更换（模块约 ¥20–30，`docs/FINAL_RESULTS.md` §表4 已注明该路线）。
- 兼容性：ESP32-S3 同芯片同 SDK，固件零修改（仅板级配置）。

### 方案 C：模型压缩（ResNet-M/S 替换专家）—— ❌ 不推荐

- ResNet-M（55K 参数）INT8 估算 ~112KB → L+M 组合 279KB → 需求 1,475,321 B，
  **仍超 33.5KB**；ResNet-S（25K）~51KB → L+S 组合 218KB → 勉强放下（余 34KB，
  无安全余量），但 S 在 PTB 域性能预期显著掉点（未实测，需重训+重评估）。
- 收益（省 82KB）远低于方案 A 的零成本 1.35MB，且引入性能风险。**否决**。

### 方案 D：分时加载（模型存 ffat，运行时切换）—— ❌ 不推荐（近期）

- 4MB 板 ffat 仅 960KB，存双模型（327KB）可行，但 TFLite Micro 静态 interpreter
  / 固定 arena 需重构为动态模型切换，复杂度高、调试成本大，收益被方案 A/B 覆盖。

## 3. 推荐路线

1. **近期（当前板）**：方案 A —— 新增 `partitions/esp32s3_4m_noota.csv` +
   `platformio.ini` 配置项，`pio run` 验证双模型头文件编译（固件代码不变）。
2. **中期**：方案 B（N16R8 到货后换板，双模型 + 未来模型无容量障碍）。
3. 方案 C/D 仅在新需求（云端模型下放）出现且 16MB 板不可用时才重新评估。

## 4. 验证留痕

- 本报告基于 2026-08-06 `pio run`（SUCCESS, RAM 32.1% / Flash 94.1%）实测分区
  （`partitions.bin` 解析：ota_0/ota_1 各 0x160000）。
- 分区调整落地后验收：`pio run` 通过 + `esptool.py image_info` 显示 app 分区
  0x2BF000 + 双 `ecg_model_*_int8.tflite` 转 C 头文件编译链接成功。

## 5. 落地状态（2026-08-06 实施, TH §二十七）

- **分区表已生效**：`partitions/esp32s3_4m_noota.csv`（去 ota_1, ota_0 1408K→2816K,
  纯 ASCII 注释——platformio 在 Windows 用 GBK 读 CSV, UTF-8 中文注释会
  UnicodeDecodeError）+ `platformio.ini` 挂载 → `pio run` SUCCESS:
  **Flash 94.1% → 47.1%**（1,356,993 / 2,883,584 B）。
- **双模型链接实测**：`ecg_model_p2a_data.h`（P2A INT8 167,376 B）经独立编译单元
  + extern 引用 + 真实读取 probe 强制链接 → **Flash 52.9%**（1,524,437 B,
  与理论 1,524,377 一致, +60B 为 probe 代码）→ 余 1.36MB（三模型亦可行）。
  注意坑：`__attribute__((used))` 的 static volatile 指针仍被 `--gc-sections`
  裁剪, 必须真实引用（printf 读取数组首字节）。
- **probe 已清理**：p2a_probe.cpp / ecg_model_p2a_data.h / extern 引用全部移除,
  最终 `pio run` 回到单模型 47.1%。（P2A 头文件可随时用
  `export.py tflite_to_c_array` 重新生成。）
- **烧录注意**：分区布局变更属一次性烧录动作（bootloader 区不变, 仅 partition
  table + app 区偏移调整）——按"不烧录"原则留待硬件阶段执行；`pio run -t erase`
  非必需（分区表偏移兼容旧 app 位置, 直接烧录即可）。
