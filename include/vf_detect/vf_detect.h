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

/* ===== 参数 (PC 校准 v2, eval_vf_detect_v2.py 2026-08-16) =====
 * v1 系数在 mV 域标定但固件喂 V 域 (AFE 正常窦律 11 次 VF 误报, 实测);
 * v2: ①输入由 main.cpp 显式换算 mV (AFE/SIM ×0.763, REPLAY ×0.001)
 *      ②特征逐位复刻 PC (4 节全精度 SOS forward-backward + ZCR 主频, 弃旧 2 节近似) */
#define VF_WIN_SAMPLES   1250      /* 5s @250Hz */
#define VF_FEATURES      6
/* 标准化 (训练域) */
#define VF_MEAN_0  0.42513171332031363f
#define VF_MEAN_1  0.24363859527220647f
#define VF_MEAN_2  0.25254772741265197f
#define VF_MEAN_3  0.04137449856733724f
#define VF_MEAN_4  43.309992836676216f
#define VF_MEAN_5  4.105229226361016f
#define VF_STD_0   0.4191231586420349f
#define VF_STD_1   0.3631392072566181f
#define VF_STD_2   0.12891100945922151f
#define VF_STD_3   0.004458203265709307f
#define VF_STD_4   9.81194160995198f
#define VF_STD_5   1.7036455584085064f
/* 逻辑回归系数 */
#define VF_COEF_0  0.6866138077335683f
#define VF_COEF_1  4.134092762658282f
#define VF_COEF_2  -0.8109653996849735f
#define VF_COEF_3  1.2396861763592981f
#define VF_COEF_4  -1.2619362566532957f
#define VF_COEF_5  1.3262778596404012f
#define VF_INTERCEPT 0.24291412543887136f
#define VF_THETA  0.15f      /* 决策阈值 (PC 校准 θ=0.15) */
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
