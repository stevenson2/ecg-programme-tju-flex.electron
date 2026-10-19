/**
 * Brady/tachy must use the newest 30 s of RR intervals.
 *
 * Fixture: 20 beats at RR 2.0 s (30 bpm) then 100 beats at RR 0.3 s (200 bpm).
 * That fills the 120-beat buffer exactly. The oldest 30 s is bradycardia;
 * the newest 30 s is tachycardia.
 *
 *   g++ -std=c++17 \
 *     -I experiments/esp_idf_ecg_migration/components/ecg_core/include \
 *     test/rhythm_safety_window_test.cpp \
 *     experiments/esp_idf_ecg_migration/components/ecg_core/src/rhythm_safety.cpp \
 *     -o /tmp/rs_test && /tmp/rs_test
 */
#include "rhythm_safety/rhythm_safety.h"

#include <cstdio>
#include <cstring>

static int g_fail = 0;

static HR_Result beat(float rr) {
    HR_Result hr;
    std::memset(&hr, 0, sizeof(hr));
    hr.beatDetected = true;
    hr.rrInterval = rr;
    hr.sqi = 1.0f;
    hr.bpm = (rr > 0.0f) ? (uint8_t)(60.0f / rr) : 0;
    return hr;
}

static void expect_true(bool cond, const char *msg) {
    if (!cond) {
        std::printf("FAIL: %s\n", msg);
        g_fail++;
    }
}

int main(void) {
    rsInit();

    RS_Result early{};
    for (int i = 0; i < 20; i++) {
        HR_Result hr = beat(2.0f);
        early = rsProcess(&hr);
    }
    expect_true(early.bradycardia, "opening 30 s at 30 bpm must be bradycardia");
    expect_true(!early.tachycardia, "opening window must not be tachycardia");

    RS_Result last{};
    for (int i = 0; i < 100; i++) {
        HR_Result hr = beat(0.3f);
        last = rsProcess(&hr);
    }
    std::printf("final hrWindow=%.2f brady=%d tachy=%d\n",
                last.hrWindowBpm, (int)last.bradycardia, (int)last.tachycardia);
    expect_true(last.tachycardia, "newest 30 s at 200 bpm must be tachycardia");
    expect_true(!last.bradycardia, "newest 30 s must not stay bradycardia");
    expect_true(last.hrWindowBpm > 180.0f, "window HR must be the fast half");

    rsReset();
    HR_Result pause = beat(4.0f);
    RS_Result asy = rsProcess(&pause);
    expect_true(asy.asystole, "RR >= 4 s is asystole");
    expect_true(!asy.bradycardia && !asy.tachycardia,
                "asystole return does not also set a rate alarm");

    if (g_fail == 0) {
        std::printf("PASS rhythm window\n");
        return 0;
    }
    std::printf("%d failure(s)\n", g_fail);
    return 1;
}
