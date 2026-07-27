/**
 * @file ecg_model_data.h
 * @brief TFLite 模型权重 (C 数组)
 *
 * 该文件由 pc_tools/ecg_dl/export.py --pipeline 自动生成
 *
 * 生成命令:
 *   cd pc_tools/ecg_dl
 *   python export.py --pipeline
 *
 * 然后将生成的 models/ecg_model_data.h 复制到此处
 */

#ifndef ECG_MODEL_DATA_H
#define ECG_MODEL_DATA_H

/* ======================== 占位模型 ======================== */
/*
 * 运行 export.py --pipeline 生成实际的模型数据,
 * 然后替换以下占位数组。
 *
 * 最小模型示例: 空模型 (2 字节), 确保编译通过
 * 实际模型: ~5KB INT8 量化 1D-CNN
 */

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 占位 TFLite FlatBuffer 模型数据
 * 该数据仅用于编译验证, 不产生有意义的推理结果
 *
 * 替换方法:
 *   1. 运行: python pc_tools/ecg_dl/export.py --pipeline
 *   2. 复制: pc_tools/ecg_dl/models/ecg_model_data.h -> include/ai_inference/
 */
extern const uint8_t ecg_model_data[];
extern const unsigned int ecg_model_data_len;

#ifdef __cplusplus
}
#endif

#endif /* ECG_MODEL_DATA_H */
