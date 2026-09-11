/*
 * ecg_bench.cpp - ESP32-S3 串口 bench 模式（ECG_AI_BENCH 构建专用）
 *
 * PC 通过 USB-CDC 二进制协议直驱双模型 + AND 门控，用于"模拟 ECG 误报/召回"
 * 上板验收（M4 规范验收协议的固件侧）：
 *   'B' + 250×f32 LE   -> bench beat 推理（不改门控）  -> "B <p_b>\n"
 *   'R' + 1250×f32 LE  -> bench record 推理（不改门控）-> "R <p_r>\n"
 *   'G' + 250×f32 LE   -> 门控流式喂一拍               -> "G <p_b> <p_r> <hbw> <have_r> <pos> <conf> <lat_us>\n"
 *   'X'                -> 门控状态复位                 -> "X ok\n"
 *   'P' + 4×f32        -> set threshold_r              -> "P ok\n"
 *   'Q' + 4×f32        -> set threshold_b              -> "Q ok\n"
 *   'W' + 4×u32 LE     -> confirm_mode,confirm_k,cooldown_beats,hbw_n -> "W ok\n"
 *   'V'                -> "V bench1 dual=<0/1> inferences=<n> confirmed=<n>\n"
 *
 * 喂入窗口 = 部署链形态（因果 HP 后逐拍 z-score, 250 点/拍）。
 * bench 模式独占 Serial；生产构建（无 ECG_AI_BENCH）不含本文件逻辑。
 */
#include "ai_inference/ecg_bench.h"
#include "ai_inference/ecg_ai_dual.h"
#include "ai_inference/ecg_model_data.h"
#include "ai_inference/ecg_record5s_data.h"

#include <Arduino.h>
#include <string.h>

namespace {

constexpr int kBenchWin = 250;
constexpr int kBenchRec = 1250;

struct BenchState {
    bool active = false;         /* 'S' 激活后独占 Serial */
    bool dual_ready = false;
    float fbuf[kBenchRec] = {0};
    int fneed = 0;
    int fgot = 0;
    char cmd = 0;
    ecg_ai_dual_config_t cfg{};
};

BenchState g_bench;

void bench_printf_line(const char *tag, float a, float b, float c,
                       int d, int e, int f, uint32_t g) {
    /* %d 与整数; 浮点 4 位小数 */
    char line[160];
    snprintf(line, sizeof(line),
             "%s %.4f %.4f %.4f %d %d %d %u\n",
             tag, a, b, c, d, e, f, (unsigned)g);
    Serial.print(line);
}

void bench_send(const char *s) { Serial.print(s); }

void bench_handle_cmd(char cmd) {
    if (cmd == 'S') {              /* activate */
        g_bench.active = true;
        if (!g_bench.dual_ready) {
            ecg_ai_dual_config_default(&g_bench.cfg);
            g_bench.dual_ready = ecg_ai_dual_init(
                ecg_model_data, ecg_model_data_len,
                ecg_record5s_data, ecg_record5s_data_len, &g_bench.cfg);
        }
        bench_send(g_bench.dual_ready ? "S ok dual=1\n" : "S fail dual=0\n");
        return;
    }
    if (!g_bench.active) return;   /* 其余命令需先激活 */

    if (cmd == 'V') {
        char line[96];
        snprintf(line, sizeof(line), "V bench1 dual=%d inferences=%u confirmed=%u\n",
                 (int)ecg_ai_dual_is_initialized(),
                 (unsigned)ecg_ai_dual_total_inferences(),
                 (unsigned)ecg_ai_dual_total_confirmed());
        Serial.print(line);
        g_bench.cmd = 0;
        return;
    }
    if (cmd == 'X') {
        ecg_ai_dual_gate_reset();
        bench_send("X ok\n");
        g_bench.cmd = 0;
        return;
    }
    if (cmd == 'B') { g_bench.fneed = kBenchWin; g_bench.fgot = 0; g_bench.cmd = 'B'; return; }
    if (cmd == 'R') { g_bench.fneed = kBenchRec; g_bench.fgot = 0; g_bench.cmd = 'R'; return; }
    if (cmd == 'G') { g_bench.fneed = kBenchWin; g_bench.fgot = 0; g_bench.cmd = 'G'; return; }
    if (cmd == 'P') { g_bench.fneed = 1; g_bench.fgot = 0; g_bench.cmd = 'P'; return; }
    if (cmd == 'Q') { g_bench.fneed = 1; g_bench.fgot = 0; g_bench.cmd = 'Q'; return; }
    if (cmd == 'W') { g_bench.fneed = 4; g_bench.fgot = 0; g_bench.cmd = 'W'; return; }
    g_bench.cmd = 0;               /* 未知命令, 丢弃 */
}

void bench_finish_payload() {
    if (g_bench.cmd == 'B') {
        float p_b = 0.0f;
        if (ecg_ai_dual_bench_beat(g_bench.fbuf, &p_b)) {
            char line[64];
            snprintf(line, sizeof(line), "B %.4f\n", p_b);
            Serial.print(line);
        } else {
            bench_send("B err\n");
        }
    } else if (g_bench.cmd == 'R') {
        float p_r = 0.0f;
        if (ecg_ai_dual_bench_record(g_bench.fbuf, &p_r)) {
            char line[64];
            snprintf(line, sizeof(line), "R %.4f\n", p_r);
            Serial.print(line);
        } else {
            bench_send("R err\n");
        }
    } else if (g_bench.cmd == 'G') {
        ecg_ai_dual_result_t r = ecg_ai_dual_feed_beat(g_bench.fbuf);
        bench_printf_line("G", r.p_b, r.p_r, r.hbw,
                          (int)r.have_r, (int)r.gate_pos, (int)r.confirmed,
                          r.latency_us);
    } else if (g_bench.cmd == 'P') {
        g_bench.cfg.threshold_r = g_bench.fbuf[0];
        bench_send("P ok\n");
    } else if (g_bench.cmd == 'Q') {
        g_bench.cfg.threshold_b = g_bench.fbuf[0];
        bench_send("Q ok\n");
    } else if (g_bench.cmd == 'W') {
        g_bench.cfg.confirm_mode = (int)g_bench.fbuf[0];
        g_bench.cfg.confirm_k = (int)g_bench.fbuf[1];
        g_bench.cfg.cooldown_beats = (int)g_bench.fbuf[2];
        g_bench.cfg.hbw_n = (int)g_bench.fbuf[3];
        bench_send("W ok\n");
    }
    g_bench.cmd = 0;
}

} /* namespace */

void ecg_bench_init(void) {
    /* 懒初始化: 双模型在首次 'S' 激活时构建（避免与生产 AI 抢 PSRAM） */
}

bool ecg_bench_active(void) {
    return g_bench.active;
}

void ecg_bench_on_byte(uint8_t b) {
    if (!g_bench.active) {
        bench_handle_cmd((char)b);
        return;
    }
    if (g_bench.cmd == 0) {
        bench_handle_cmd((char)b);
        return;
    }
    /* 二进制 payload: 小端 f32 逐字节收集 */
    if (g_bench.fgot < g_bench.fneed * (int)sizeof(float)) {
        ((uint8_t *)g_bench.fbuf)[g_bench.fgot++] = b;
        if (g_bench.fgot >= g_bench.fneed * (int)sizeof(float)) {
            bench_finish_payload();
        }
    }
}
