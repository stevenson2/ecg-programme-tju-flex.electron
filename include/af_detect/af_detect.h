#ifndef AF_DETECT_H
#define AF_DETECT_H

/**
 * @file af_detect.h
 * @brief 模块3: AF RR 不规则度检测 (T4-8 + 待办#5 10s 快检)
 *
 * 设计 (consumer_ecg_architecture_plan.md 模块3 + R4 范式):
 *   - 10s RR 窗 (默认, "一键测房颤"快检模式; PTB-XL 记录级验证 AUC 0.9717)
 *     30s 窗行业标准模式可编译期切回 (platformio.ini build_flags 覆盖 AF_WIN_S 等)
 *   - 特征: 变异系数 CV = SDNN/mean, Shannon 熵 (16 bins, 0.3-1.5s)
 *   - 三档输出: 0=正常 / 1=AF 疑似 / 2=无法判定 (RR 数不足或质量差)
 * 阈值 (10s 快检, PTB-XL 校准, FINAL_RESULTS 表6 补充行):
 *   CV>0.08, 熵>1.2 (10s 窗 RR 少、直方图稀疏、熵系统性偏低 → 30s 阈值 1.9 失效,
 *   见 eval_rhythm_af_ptbxl.py 全量验证); 最少 RR=6 (10s @ 40bpm 下限)
 */

#include <stdint.h>
#include <stdbool.h>
#include "heartrate/heartrate.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===== 参数 (默认 10s 快检; 30s 模式: 30.0f/20/0.10f/1.9f) ===== */
#ifndef AF_WIN_S
#define AF_WIN_S          10.0f   /* 评估窗 (秒) */
#endif
#ifndef AF_MIN_RR
#define AF_MIN_RR         6       /* 窗内最少 RR 数 (不足 → 无法判定) */
#endif
#ifndef AF_CV_THR
#define AF_CV_THR         0.08f   /* CV 阈值 (10s 快检校准) */
#endif
#ifndef AF_ENT_THR
#define AF_ENT_THR        1.2f    /* Shannon 熵阈值 (10s 快检校准, 16 bins) */
#endif
#ifndef AF_SQI_GATE
#define AF_SQI_GATE       0.5f    /* 窗内平均 SQI 门控 (低质量 → 无法判定) */
#endif
#ifndef AF_RR_BUF
#define AF_RR_BUF         120     /* RR 缓冲 (10s 窗 ~13拍 / 30s 窗 ~37拍, 裕量) */
#endif

typedef struct {
    uint8_t  label;       /**< 0=正常 1=AF疑似 2=无法判定 */
    float    score;       /**< 组合分数 (CV/0.2 与 熵/4.5 加权) */
    float    cv;          /**< 窗内变异系数 */
    float    entropy;     /**< 窗内 Shannon 熵 */
    uint8_t  nRr;         /**< 窗内 RR 数 */
    bool     windowReady; /**< 窗是否已满 (未满 → label=2) */
} AF_Result;

/** @brief 初始化 */
void afInit(void);

/**
 * @brief 每帧处理 (hrProcess 后调用; 窗未满时 label=2 无法判定)
 */
AF_Result afProcess(const HR_Result *hr);

/** @brief 复位 (信号源切换时调用) */
void afReset(void);

#ifdef __cplusplus
}
#endif

#endif /* AF_DETECT_H */
