/* Host replay of the firmware VF path.
 *
 * cmd 1: one raw 1250-sample window -> features + logistic score
 *        (compute_features / score_window, no front-end filter)
 * cmd 2: 500 Hz stream through the shipped replay chain:
 *        x - 1.65, double 10-tap comb (main.cc), applyFilter (HP 0.5 + LP 40),
 *        hrProcess every sample, vfProcess(filtered) on even samples
 *        (replay mode: no ×0.763). Returns per-window score/rms/suspect/alarm
 *        and QRS sample indices.
 * cmd 3: 250 Hz stream into vfProcess only (calibration-domain input, no filter).
 *
 * Worker creation is stubbed to fail, so vfProcess uses its synchronous branch.
 * main.cc calls filterInit(), not filterWarmup(); this host does the same.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <vector>
#include "filter/filter.h"
#include "heartrate/heartrate.h"
#include "vf_detect/vf_detect.h"

int64_t g_host_time_us = 1000000;

extern "C" void vf_host_score(const float* src, float* feat, float* score, float* rms, int* suspect);

static bool read_full(void* dst, size_t n) {
    uint8_t* p = static_cast<uint8_t*>(dst);
    size_t got = 0;
    while (got < n) {
        size_t r = fread(p + got, 1, n - got, stdin);
        if (r == 0) return false;
        got += r;
    }
    return true;
}

static bool write_full(const void* src, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(src);
    size_t put = 0;
    while (put < n) {
        size_t w = fwrite(p + put, 1, n - put, stdout);
        if (w == 0) return false;
        put += w;
    }
    fflush(stdout);
    return true;
}

/* Copied from main.cc (COMB_TAPS = 10). Two cascaded moving averages.
 * At 500 Hz the zeros sit on 50 Hz and 100 Hz. */
static const int COMB_TAPS = 10;
static float s_combBuf1[10];
static int s_combIdx1 = 0;
static float s_combSum1 = 0.0f;
static float s_combBuf2[10];
static int s_combIdx2 = 0;
static float s_combSum2 = 0.0f;

static void combReset(void) {
    memset(s_combBuf1, 0, sizeof(s_combBuf1));
    memset(s_combBuf2, 0, sizeof(s_combBuf2));
    s_combIdx1 = s_combIdx2 = 0;
    s_combSum1 = s_combSum2 = 0.0f;
}

static float applyCombFilter(float x) {
    s_combSum1 -= s_combBuf1[s_combIdx1];
    s_combBuf1[s_combIdx1] = x;
    s_combSum1 += x;
    s_combIdx1 = (s_combIdx1 + 1) % COMB_TAPS;
    float y1 = s_combSum1 / static_cast<float>(COMB_TAPS);

    s_combSum2 -= s_combBuf2[s_combIdx2];
    s_combBuf2[s_combIdx2] = y1;
    s_combSum2 += y1;
    s_combIdx2 = (s_combIdx2 + 1) % COMB_TAPS;
    return s_combSum2 / static_cast<float>(COMB_TAPS);
}

static int cmd_score_window(void) {
    float x[1250];
    if (!read_full(x, sizeof(x))) return 2;
    float feat[6], score = 0, rms = 0;
    int suspect = 0;
    vf_host_score(x, feat, &score, &rms, &suspect);
    float out[9];
    out[0] = score;
    out[1] = rms;
    out[2] = static_cast<float>(suspect);
    for (int i = 0; i < 6; i++) out[3 + i] = feat[i];
    return write_full(out, sizeof(out)) ? 0 : 2;
}

static int cmd_vf_stream(void) {
    int32_t n = 0;
    if (!read_full(&n, sizeof(n))) return 2;
    if (n < 0 || n > 8000000) return 3;
    std::vector<float> x(static_cast<size_t>(n));
    if (n && !read_full(x.data(), static_cast<size_t>(n) * sizeof(float))) return 2;
    vfInit();
    std::vector<float> rows;
    rows.reserve(static_cast<size_t>(n / 1250) * 4);
    for (int32_t i = 0; i < n; i++) {
        VF_Result vf = vfProcess(x[static_cast<size_t>(i)]);
        if ((i % 1250) == 1249) {
            rows.push_back(vf.score);
            rows.push_back(vf.lastRms);
            rows.push_back(vf.windowSuspect ? 1.0f : 0.0f);
            rows.push_back(vf.vfAlarm ? 1.0f : 0.0f);
        }
    }
    int32_t nwin = static_cast<int32_t>(rows.size() / 4);
    if (!write_full(&nwin, sizeof(nwin))) return 2;
    if (nwin && !write_full(rows.data(), rows.size() * sizeof(float))) return 2;
    return 0;
}

static int cmd_frontend(void) {
    int32_t n = 0;
    if (!read_full(&n, sizeof(n))) return 2;
    if (n < 0 || n > 8000000) return 3;
    std::vector<float> x(static_cast<size_t>(n));
    if (n && !read_full(x.data(), static_cast<size_t>(n) * sizeof(float))) return 2;

    const float DC_OFFSET_REMOVE = 1.65f;
    combReset();
    filterInit();
    hrInit();
    vfInit();
    g_host_time_us = 1000000;

    std::vector<float> rows;
    std::vector<int32_t> beats;
    rows.reserve(static_cast<size_t>(n / 2500) * 4);
    beats.reserve(static_cast<size_t>(n / 400));
    int vf_i = 0;
    const bool dbg = std::getenv("VF_HOST_DEBUG") != nullptr;
    float sec_max = 0.0f;
    int sec_beats = 0;
    for (int32_t i = 0; i < n; i++) {
        float y = applyCombFilter(x[static_cast<size_t>(i)] - DC_OFFSET_REMOVE);
        float filtered = applyFilter(y);
        HR_Result hr = hrProcess(filtered);
        if (dbg) {
            sec_max = std::max(sec_max, std::fabs(filtered));
            if (hr.beatDetected) sec_beats++;
            if ((i % 500) == 499) {
                fprintf(stderr, "t=%d max|f|=%.5f beats_this_s=%d sqi=%.3f total=%zu\n",
                        i / 500, sec_max, sec_beats, hr.sqi, beats.size() + (hr.beatDetected ? 1 : 0));
                sec_max = 0.0f;
                sec_beats = 0;
            }
        }
        if (hr.beatDetected) beats.push_back(i);
        if ((i % 2) == 0) {
            VF_Result vf = vfProcess(filtered);
            vf_i++;
            if ((vf_i % 1250) == 0) {
                rows.push_back(vf.score);
                rows.push_back(vf.lastRms);
                rows.push_back(vf.windowSuspect ? 1.0f : 0.0f);
                rows.push_back(vf.vfAlarm ? 1.0f : 0.0f);
            }
        }
        g_host_time_us += 2000;
    }
    int32_t nwin = static_cast<int32_t>(rows.size() / 4);
    int32_t nb = static_cast<int32_t>(beats.size());
    if (!write_full(&nwin, sizeof(nwin))) return 2;
    if (nwin && !write_full(rows.data(), rows.size() * sizeof(float))) return 2;
    if (!write_full(&nb, sizeof(nb))) return 2;
    if (nb && !write_full(beats.data(), beats.size() * sizeof(int32_t))) return 2;
    return 0;
}

int main(void) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    for (;;) {
        int32_t cmd = 0;
        if (!read_full(&cmd, sizeof(cmd))) return 0;
        int rc = 0;
        if (cmd == 0) return 0;
        else if (cmd == 1) rc = cmd_score_window();
        else if (cmd == 2) rc = cmd_frontend();
        else if (cmd == 3) rc = cmd_vf_stream();
        else return 4;
        if (rc != 0) return rc;
    }
}
