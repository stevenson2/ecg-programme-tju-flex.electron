/**
 * @file ecg_ai_dual.h
 * @brief v2 双模型 AND 门控（M4 门控固件集成规范落地, 2026-09-07）
 *
 * beat 模型（int8, 250 点单拍, 逐拍 z-score 输入）+ record 模型（fp16 权重,
 * 1250 点 = 5×250 逐拍 z-score 窗按时间序连接）：
 *   p_b[i] = beat 模型对拍 i 的异常概率
 *   p_r[i] = record 模型对"以 i 结尾的 record_window_beats 拍窗"的异常概率
 *   hbw[i] = mean(p_b 最近 hbw_n 拍)
 *   pos[i] = (p_r[i] >= threshold_r) AND (hbw[i] >= threshold_b)
 *   cand[i] = (pos 最近 confirm_n 拍计数) >= confirm_k
 *   alarm  = cand 为真且距上次报警 >= cooldown_beats
 *
 * 输入约定：与统一基准评测的部署链形态逐位同构（逐拍 z-score 单位尺度，
 * 与 runs/r9 系评测窗、M4 e2e 回放一致）。
 * 门控配置默认 v1-min（最保守）：k=1, tau_r=0.5, tau_b=0.15, cooldown=20。
 */
#ifndef ECG_AI_DUAL_H
#define ECG_AI_DUAL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float threshold_r;        /* tau_r (v1-min: 0.5 / v2-spec: 0.7)   */
    float threshold_b;        /* tau_b (v1-min: 0.15 / v2-spec: 0.10) */
    int confirm_mode;         /* 0 = k-of-n, 1 = 1-of-n               */
    int confirm_n;            /* k-of-n 窗口（默认 5）                */
    int confirm_k;            /* K-of-N 的 K（v1-min: 1 / v2-spec: 2）*/
    int cooldown_beats;       /* 默认 20                              */
    int hbw_n;                /* hbw 平均拍数（默认 5）               */
    int record_window_beats;  /* record 输入拍数（5s 学生 = 5）       */
    int arena_size;           /* record arena 字节（建议 1MB, PSRAM） */
    bool use_psram;
} ecg_ai_dual_config_t;

typedef struct {
    float p_b;                /* beat 概率            */
    float p_r;                /* record 概率          */
    float hbw;                /* 拍概率滑窗均值       */
    uint8_t have_r;           /* record 窗已凑满      */
    uint8_t gate_pos;         /* AND 门控原始判定     */
    uint8_t confirmed;        /* k-of-n + 冷却后报警  */
    uint32_t sample_index;    /* 拍序号（喂入计数）   */
    uint32_t latency_us;      /* 双模型总 Invoke 耗时 */
} ecg_ai_dual_result_t;

void ecg_ai_dual_config_default(ecg_ai_dual_config_t *cfg);

bool ecg_ai_dual_init(const uint8_t *beat_data, size_t beat_len,
                      const uint8_t *record_data, size_t record_len,
                      const ecg_ai_dual_config_t *cfg);
void ecg_ai_dual_reset(void);
bool ecg_ai_dual_is_initialized(void);
void ecg_ai_dual_gate_reset(void);

/* 流式集成入口：喂入一拍"因果 HP 后、已逐拍 z-score"的 250 点窗。 */
ecg_ai_dual_result_t ecg_ai_dual_feed_beat(const float *window_250);

/* bench 直测入口（不改门控状态） */
bool ecg_ai_dual_bench_beat(const float *window_250, float *p_b);
bool ecg_ai_dual_bench_record(const float *window_1250, float *p_r);

uint32_t ecg_ai_dual_total_inferences(void);
uint32_t ecg_ai_dual_total_confirmed(void);

#ifdef __cplusplus
}
#endif

#endif /* ECG_AI_DUAL_H */
