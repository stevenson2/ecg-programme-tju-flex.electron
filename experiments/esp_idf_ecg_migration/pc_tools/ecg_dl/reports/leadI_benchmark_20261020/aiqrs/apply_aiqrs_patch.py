import re, sys
R = "/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/experiments/esp_idf_ecg_migration"
def rd(p): return open(f"{R}/{p}", encoding="utf-8").read()
def wr(p, s): open(f"{R}/{p}", "w", encoding="utf-8", newline="").write(s)
def rep(s, old, new, p):
    if new in s: return s
    if old not in s: sys.exit(f"anchor missing in {p}: {old[:60]!r}")
    return s.replace(old, new, 1)
nl = lambda s: "\r\n" if "\r\n" in s else "\n"

p = "components/ecg_core/CMakeLists.txt"; s = rd(p); e = nl(s)
s = rep(s, '"src/heartrate.cpp"', '"src/heartrate.cpp"' + e + '        "src/ai_qrs.c"', p); wr(p, s)

p = "components/ecg_ai/include/ecg_ai.h"; s = rd(p); e = nl(s)
s = rep(s, "    int rpeak_half_win;       /* @250Hz samples after R before fire; default 125 */",
        "    int rpeak_half_win;       /* @250Hz samples after R before fire; default 125 */" + e +
        "    int rpeak_min_rr;         /* @250Hz AI 防重触发; 0=默认150(600ms); 宽松 QRS 触发用 50(200ms) */", p)
s = rep(s, "void ecg_ai_on_rpeak(void);",
        "void ecg_ai_on_rpeak(void);" + e +
        "/* 宽松 QRS 触发 (ai_qrs): R 位于当前样本之前 lag_250 个 250Hz 样本; 支持多拍排队 */" + e +
        "void ecg_ai_on_rpeak_at(uint32_t lag_250);", p)
wr(p, s)

p = "components/ecg_ai/src/ecg_ai.cpp"; s = rd(p); e = nl(s)
s = rep(s, "    uint32_t rpeak_min_rr = 150;     /* ~600ms @250Hz antiretrig */",
        "    uint32_t rpeak_min_rr = 150;     /* ~600ms @250Hz antiretrig (cfg.rpeak_min_rr 覆盖) */" + e +
        "    /* 待发射 R 中心窗 FIFO (2026-10-20): 宽松触发下相邻 R 可 < 半窗, 单槽会被覆盖丢拍 */" + e +
        "    static constexpr int kPendN = 4;" + e +
        "    uint32_t pend[kPendN] = {0};" + e +
        "    int pend_n = 0;" + e +
        "    uint32_t pend_dropped = 0;", p)
s = rep(s, "    cfg->rpeak_half_win = 125;  /* R at center of 250-pt window */",
        "    cfg->rpeak_half_win = 125;  /* R at center of 250-pt window */" + e +
        "    cfg->rpeak_min_rr = 0;      /* 0 → 150 (legacy 600 ms) */", p)
s = rep(s, "    g_ai.last_rpeak_sc = 0;",
        "    g_ai.last_rpeak_sc = 0;" + e +
        "    g_ai.pend_n = 0;" + e +
        "    g_ai.rpeak_min_rr = (g_ai.cfg.rpeak_min_rr > 0) ? (uint32_t)g_ai.cfg.rpeak_min_rr : 150u;", p)
old = """        if (g_ai.rpeak_fire_at != 0 && g_ai.sample_count >= g_ai.rpeak_fire_at
            && g_ai.sample_count >= kDefaultWindow) {
            g_ai.rpeak_fire_at = 0;
            maybe_fire_window();
        }"""
old = old.replace("\n", e)
new = ("        if (g_ai.pend_n > 0 && g_ai.sample_count >= g_ai.pend[0]) {" + e +
       "            for (int i = 1; i < g_ai.pend_n; i++) g_ai.pend[i - 1] = g_ai.pend[i];" + e +
       "            g_ai.pend_n--;" + e +
       "            if (g_ai.sample_count >= kDefaultWindow) maybe_fire_window();" + e +
       "        }")
s = rep(s, old, new, p)
old = """    g_ai.last_rpeak_sc = g_ai.sample_count;
    g_ai.rpeak_fire_at = g_ai.sample_count + half;
}""".replace("\n", e)
new = ("    g_ai.last_rpeak_sc = g_ai.sample_count;" + e +
       "    g_ai.rpeak_fire_at = g_ai.sample_count + half;" + e +
       "    if (g_ai.pend_n < AiState::kPendN) g_ai.pend[g_ai.pend_n++] = g_ai.rpeak_fire_at;" + e +
       "    else g_ai.pend_dropped++;" + e +
       "}" + e + e +
       "void ecg_ai_on_rpeak_at(uint32_t lag_250) {" + e +
       "    if (!g_ai.initialized) return;" + e +
       "    if (g_ai.cfg.trigger_mode != ECG_AI_TRIGGER_RPEAK) return;" + e +
       "    uint32_t half = (g_ai.cfg.rpeak_half_win > 0) ? (uint32_t)g_ai.cfg.rpeak_half_win : 125u;" + e +
       "    if (lag_250 > half - 5u) lag_250 = half - 5u;   /* 窗口须仍在 250 点环形缓冲内 */" + e +
       "    if (lag_250 > g_ai.sample_count) return;" + e +
       "    uint32_t r_sc = g_ai.sample_count - lag_250;" + e +
       "    if (g_ai.last_rpeak_sc != 0 && (r_sc - g_ai.last_rpeak_sc) < g_ai.rpeak_min_rr) return;" + e +
       "    g_ai.last_rpeak_sc = r_sc;" + e +
       "    if (g_ai.pend_n < AiState::kPendN) g_ai.pend[g_ai.pend_n++] = r_sc + half;" + e +
       "    else g_ai.pend_dropped++;" + e +
       "}")
s = rep(s, old, new, p)
wr(p, s)

p = "main/main.cc"; s = rd(p); e = nl(s)
s = rep(s, '#include "heartrate/heartrate.h"',
        '#include "heartrate/heartrate.h"' + e + '#include "ai_qrs/ai_qrs.h"' + e + e +
        "/* 2026-10-20: AI 通道独立宽松 QRS 触发 (不做早搏拒绝)。1=新 (默认), 0=恢复旧行为" + e +
        " * (hr.beatDetected → ecg_ai_on_rpeak, 600 ms 防重触发)。心率仍由 hrProcess 严格检测器负责。 */" + e +
        "#ifndef ECG_AI_PERMISSIVE_QRS" + e + "#define ECG_AI_PERMISSIVE_QRS 1" + e + "#endif", p)
s = rep(s, "    cfg.rpeak_half_win = 125;",
        "    cfg.rpeak_half_win = 125;" + e +
        "#if ECG_AI_PERMISSIVE_QRS" + e + "    cfg.rpeak_min_rr = 50;   /* 200 ms */" + e + "#endif", p)
s = rep(s, "    hrInit();" + e + "    rsInit();", "    hrInit();" + e + "    aiqrsInit(NULL);" + e + "    rsInit();", p)
s = rep(s, "        HR_Result hr = hrProcess(filteredSample);",
        "        HR_Result hr = hrProcess(filteredSample);" + e +
        "#if ECG_AI_PERMISSIVE_QRS" + e +
        "        AIQRS_Result aq = aiqrsProcess(filteredSample);" + e +
        "        if (aq.fired) ecg_ai_on_rpeak_at((uint32_t)aq.r_lag / 2u);" + e +
        "#endif", p)
s = rep(s, "            ecg_ai_on_rpeak(); /* schedule R-centered AI window */",
        "#if !ECG_AI_PERMISSIVE_QRS" + e +
        "            ecg_ai_on_rpeak(); /* legacy: schedule R-centered AI window */" + e +
        "#endif", p)
wr(p, s)
# also reset aiqrs wherever hrFullReset() is called in main.cc
s = rd(p)
cnt = s.count("hrFullReset();")
s2 = re.sub(r"hrFullReset\(\);(?!\s*aiqrsInit)", "hrFullReset(); aiqrsInit(NULL);", s)
wr(p, s2)
print("patched; hrFullReset sites:", cnt)
