#ifndef VF_DETECT_H
#define VF_DETECT_H

/**
 * @file vf_detect.h
 * @brief 模块2: VF/VT 检测器 (T4-9, 秒级危急报警)
 *
 * 设计 (consumer_ecg_architecture_plan.md 模块2):
 *   - 5s 窗 (1250 样本 @250Hz) DSP 特征 + 逻辑回归分数 (PC 训练, VFDB/CUDB 验证)
 *   - 特征: rms / 幅度中位 / VF滤波比(4-10Hz) / VF带ZCR / 峰谷率 / 主频近似
 *   - 连续 2 窗确认 (R1 多次确认范式, 压误报; 时延 ≤7.5s)
 *   - PC 验证 (eval_vf_detect.py): VFDB 留出 Se 0.957 / MIT-BIH 对照 Sp 0.824
 */

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===== 参数 (PC 校准, eval_vf_detect.py model_params) ===== */
#define VF_WIN_SAMPLES   1250      /* 5s @250Hz */
#define VF_FEATURES      6
/* 标准化 (训练域) */
#define VF_MEAN_0  0.4251f
#define VF_MEAN_1  0.2436f
#define VF_MEAN_2  0.2601f
#define VF_MEAN_3  0.0409f
#define VF_MEAN_4  43.31f
#define VF_MEAN_5  3.2761f
#define VF_STD_0   0.4191f
#define VF_STD_1   0.3631f
#define VF_STD_2   0.1320f
#define VF_STD_3   0.0045f
#define VF_STD_4   9.8119f
#define VF_STD_5   1.0245f
/* 逻辑回归系数 */
#define VF_COEF_0  1.2346f
#define VF_COEF_1  3.0073f
#define VF_COEF_2 -0.5847f
#define VF_COEF_3  1.5251f
#define VF_COEF_4 -0.8605f
#define VF_COEF_5  1.1886f
#define VF_INTERCEPT -0.0307f
#define VF_THETA  0.12f      /* 决策阈值 (PC 校准 θ=0.12) */
#define VF_CONFIRM_WINDOWS 2 /* 连续 2 窗确认 */

typedef struct {
    bool     vfAlarm;       /**< VF/VT 报警 (连续 2 窗确认后) */
    bool     windowSuspect; /**< 当前窗疑似 (未确认) */
    float    score;         /**< 逻辑回归分数 (0-1) */
    float    lastRms;       /**< 最近窗 RMS (mV) */
    uint32_t confirmedAtMs; /**< 确认时刻 (ms) */
} VF_Result;

/** @brief 初始化 */
void vfInit(void);

/**
 * @brief 每帧处理 (采样率 250Hz, 每样本调用一次)
 * @param sample 滤波后样本 (与 AI 推理同输入)
 * @return 检测结果 (窗未满时 vfAlarm=false)
 */
VF_Result vfProcess(float sample);

/** @brief 复位 (信号源切换时调用) */
void vfReset(void);

#ifdef __cplusplus
}
#endif

#endif /* VF_DETECT_H */
