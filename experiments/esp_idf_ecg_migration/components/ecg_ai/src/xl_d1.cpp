/* D1: model-only consistency + timing on 512 baked dev40 vectors (gen_d1.py). */
#include "ecg_ai.h"

#ifdef ECG_AI_MODEL_XLSMALL
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_heap_caps.h"
#include "xl_pipeline.h"
#include "xl_model.h"
#include "xl_d1_vectors.h"

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static void softmax_model(const int8_t *q, int m, float *p) {
    float lg[4], mx = -1e30f, s = 0.0f;
    for (int k = 0; k < 4; k++) {
        lg[k] = ((float)q[k] - (float)XL_OUT_ZP[m]) * XL_OUT_SCALE[m];
        if (lg[k] > mx) mx = lg[k];
    }
    for (int k = 0; k < 4; k++) { lg[k] = expf(lg[k] - mx); s += lg[k]; }
    for (int k = 0; k < 4; k++) p[k] = lg[k] / s;
}

extern "C" void ecg_ai_xl_d1(void) {
    if (!ecg_ai_is_initialized()) { printf("D1 fail ai_not_initialised\n"); return; }
    static uint32_t us[2][XL_D1_N];
    uint32_t same_bytes = 0, total_bytes = 0, max_lsb = 0;
    uint32_t cls_agree[2] = {0, 0}, ens_agree = 0;
    float max_pdiff[2] = {0, 0}, max_pdiff_ens = 0;
    for (int i = 0; i < XL_D1_N; i++) {
        float pb[2][4], pr[2][4];
        for (int m = 0; m < 2; m++) {
            int8_t out[4];
            int t = xl_model_infer(m, XL_D1_WIN + i * XL_WIN, XL_D1_FEAT + i * 4, out);
            if (t < 0) { printf("D1 fail infer %d/%d\n", i, m); return; }
            us[m][i] = (uint32_t)t;
            const int8_t *ref = XL_D1_REF + (i * 2 + m) * 4;
            for (int k = 0; k < 4; k++) {
                uint32_t d = (uint32_t)abs((int)out[k] - (int)ref[k]);
                total_bytes++;
                if (d == 0) same_bytes++;
                if (d > max_lsb) max_lsb = d;
            }
            softmax_model(out, m, pb[m]);
            softmax_model(ref, m, pr[m]);
            int ab = 0, ar = 0;
            for (int k = 0; k < 4; k++) {
                float df = fabsf(pb[m][k] - pr[m][k]);
                if (df > max_pdiff[m]) max_pdiff[m] = df;
                if (pb[m][k] > pb[m][ab]) ab = k;
                if (pr[m][k] > pr[m][ar]) ar = k;
            }
            if (ab == ar) cls_agree[m]++;
            printf("D1,%d,%d,%d,%d,%d,%d,%lu\n", i, m, out[0], out[1], out[2], out[3], (unsigned long)us[m][i]);
        }
        int ab = 0, ar = 0;
        for (int k = 0; k < 4; k++) {
            float eb = 0.5f * (pb[0][k] + pb[1][k]), er = 0.5f * (pr[0][k] + pr[1][k]);
            float eb_a = 0.5f * (pb[0][ab] + pb[1][ab]), er_a = 0.5f * (pr[0][ar] + pr[1][ar]);
            if (fabsf(eb - er) > max_pdiff_ens) max_pdiff_ens = fabsf(eb - er);
            if (eb > eb_a) ab = k;
            if (er > er_a) ar = k;
        }
        if (ab == ar) ens_agree++;
    }
    uint32_t med[2], p95[2];
    for (int m = 0; m < 2; m++) {
        static uint32_t tmp[XL_D1_N];
        memcpy(tmp, us[m], sizeof(tmp));
        qsort(tmp, XL_D1_N, sizeof(uint32_t), cmp_u32);
        med[m] = (tmp[XL_D1_N / 2 - 1] + tmp[XL_D1_N / 2]) / 2;
        p95[m] = tmp[(XL_D1_N * 95) / 100];
    }
    printf("D1_SUMMARY n=%d same_bytes=%lu/%lu max_lsb=%lu maxpdiff_m0=%.5f maxpdiff_m1=%.5f maxpdiff_ens=%.5f "
           "cls_agree_m0=%lu cls_agree_m1=%lu cls_agree_ens=%lu us_med=%lu,%lu us_p95=%lu,%lu arena_used=%u,%u arena=%u,%u "
           "free_internal=%u free_psram=%u min_internal=%u\n",
           XL_D1_N, (unsigned long)same_bytes, (unsigned long)total_bytes, (unsigned long)max_lsb,
           (double)max_pdiff[0], (double)max_pdiff[1], (double)max_pdiff_ens,
           (unsigned long)cls_agree[0], (unsigned long)cls_agree[1], (unsigned long)ens_agree,
           (unsigned long)med[0], (unsigned long)med[1], (unsigned long)p95[0], (unsigned long)p95[1],
           (unsigned)xl_model_arena_used(0), (unsigned)xl_model_arena_used(1),
           (unsigned)xl_model_arena_size(0), (unsigned)xl_model_arena_size(1),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL), (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL));
}
#endif
