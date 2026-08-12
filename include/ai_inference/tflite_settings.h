/**
 * @file tflite_settings.h
 * @brief TFLite Micro 运行时配置
 *
 * 适配 ESP32-S3 (240MHz, 2MB PSRAM)
 * 
 * 内存预算:
 *   - Tensor Arena: 32KB (模型输入 1KB + 中间特征图 ~25KB)
 *   - 模型权重: ~5KB (INT8, 嵌入 Flash)
 *   - 总计: ~37KB (从 SRAM 分配)
 */

#ifndef TFLITE_SETTINGS_H
#define TFLITE_SETTINGS_H

/* ======================== 模型参数 ======================== */
#define MODEL_INPUT_SIZE    250    /* 输入样本数 */
#define MODEL_INPUT_CHANNELS 1     /* 单通道 ECG */
#define MODEL_OUTPUT_CLASSES 2     /* 二分类: Normal / Abnormal */
#define MODEL_QUANTIZED      1     /* 1 = INT8, 0 = FP32 */

/* ======================== TFLite Micro Arena ======================== */
/* 
 * Tensor Arena 大小:
 *   - 输入张量:  250 * 1 * 1 byte (INT8) = 250 B
 *   - 中间特征:  ~25KB (Conv1D + MaxPool 中间结果)
 *   - 输出张量:  2 * 1 byte (INT8) = 2 B
 *   - TFLite 元数据: ~4KB
 *   - 预留余量: 2.75KB
 *   - 总计: 32 * 1024 = 32768 B
 * ⚠️ 2026-08-08 N16R8 板上实测: 32KB 不够, AllocateTensors 失败
 *    (TFLite Micro 需额外算子临时缓冲, 如权重重排副本) → 扩至 64KB
 *    (SRAM 余量: 编译 92KB/320KB, 64KB arena 后仍充足; 双模型需 2x64KB)
 */
#define TENSOR_ARENA_SIZE    (64 * 1024)   /* 64KB */

/* ======================== 推理配置 ======================== */
/* 2026-08-10 (TH §40): 真实 ECG 验证发现模型分布偏移 (正常 QRS 形态与 MIT-BIH
 * 训练分布差异, 误报置信度中位 0.93) → 调高阈值 + 多拍确认抑制误报。
 * 论文最优操作点 beat θ=0.35 / patient θ=0.5; 监护场景取更保守的 0.60。
 * 代价: 对真异常敏感性下降 (多拍确认 5 拍 = 5s 确认延迟), 模型微调后可回退。 */
#define INFERENCE_THRESHOLD  0.60f   /* 异常判定阈值 (0.35→0.60, TH §40) */
#define MULTI_BEAT_CONFIRM   5       /* 多拍确认: 连续N拍异常才报警 (2→5, TH §40) */
#define INFERENCE_ENABLED    1       /* 默认启用 */

/* 输入抽取 (方案A, 修复 4.4-4 蹊跷点6: 训练窗口1.0s vs 部署0.5s 不匹配)
 *   - 抽取前: caller 按 500Hz 推送 (每 2ms 1 样本, main.cpp 已做)
 *   - 抽取后: AI 环形缓冲有效采样率 250Hz (仅保留偶数序号样本)
 *   - 窗口    AI_WINDOW_SIZE=250 样本 = 1.0s   (与训练一致)
 *   - 步进    AI_STRIDE=250       样本 = 1.0s   (⚠️ 2026-08-08: 原 125/0.5s,
 *             板上实测单次推理 ~910ms > 500ms 触发间隔 → 任务占满 Core 0,
 *             IDLE0 饿死触发 Task WDT 崩溃重启; 改 1s 间隔 + 推理后让出 CPU)
 *   - 首次推理: 上电后约 1.0s
 *   - 多拍确认: MULTI_BEAT_CONFIRM=2 拍异常时, 时间跨度 >= 2.0s
 *   - 反混叠: LP40 (cutoff 40Hz << Nyquist 125Hz) 已由 filter.cpp 提供
 *   - 关闭: 改为 1 即回到 500Hz 路径 (窗口退化为 0.5s, 恢复旧行为)
 */
#define AI_INPUT_DECIMATION  2   /* 输入抽取: 500Hz->250Hz, 窗口恢复1.0s (方案A, 修复4.4-4蹊跷点6) */

/* 推理步进: 250 个抽取样本 = 1.0s (见 AI_INPUT_DECIMATION 注释, 2026-08-08 从 125 调大) */
#define AI_STRIDE           250

/* 群延迟补偿 (T1-3 P0, 2026-08-06): 因果部署链 (梳状+HP0.5+LP40+2:1抽取) 群延迟
 * ~6 样本 @250Hz (24ms), R 峰在推理窗口内滞后 6 样本 (错位 ~0.035 AUC, corr 0.44)。
 * 触发时刻后移 6 样本 (idx%AI_STRIDE==AI_TRIGGER_OFFSET), 等效评估侧 δ=+6 窗口
 * 重提取语义, 令 R 峰回到训练窗口位置 (125), 零成本抵消群延迟 (FINAL_RESULTS 表5)。
 * 约束: 0 < AI_TRIGGER_OFFSET < AI_STRIDE。 */
#define AI_TRIGGER_OFFSET    6

/* ======================== 性能配置 ======================== */
#define AI_CORE_ID          0       /* 推理任务绑定核心 (Core 0) */
#define AI_STACK_SIZE       16384   /* 任务栈 (16KB; 8KB 曾致 Invoke 栈溢出疑 WDT 卡死, 2026-08-08) */
#define AI_TASK_PRIO        1       /* 优先级 (低于主循环的2) */
#define AI_QUEUE_LENGTH     8       /* 结果队列深度 */

/* ======================== 调试选项 ======================== */
// #define AI_DEBUG_OUTPUT            /* 取消注释以启用串口调试输出 */
// #define AI_PROFILE_LATENCY          /* 基准测试用: 每拍打印 LAT,<count>,<us> (见 docs/hardware/ondevice_bench_protocol.md §7.1) */

#endif /* TFLITE_SETTINGS_H */
