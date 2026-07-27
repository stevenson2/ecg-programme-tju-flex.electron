/**
 * @file ecg_model_data.cpp
 * @brief 模型权重占位定义
 *
 * 该文件提供默认的空模型数据，使项目能够编译通过。
 * 运行 export.py --pipeline 后替换此文件内容。
 */

#include <cstdint>

/* 最小 TFLite FlatBuffer: 仅包含版本号, 无算子 */
static const uint8_t _placeholder_model[] = {
    0x18, 0x00, 0x00, 0x00,  /* 版本号 */
};

extern "C" {
    const uint8_t ecg_model_data[]    = { 0x18, 0x00, 0x00, 0x00 };
    const unsigned int ecg_model_data_len = 4;
}
