/* Host replica of the firmware AI path (main.cc loop order):
 *   comb(10-tap x2) -> applyFilter (HP0.5+LP40 @500) -> aiqrsProcess  (trigger)
 *                   -> applyFilterAI (HP0.05+LP40 @500) -> keep even samples
 *                      -> causal HP0.5 @250 -> 250-sample ring -> fire at R+125
 * Coefficients are copied from components/ecg_core/src/filter.cpp and
 * components/ecg_ai/src/ecg_ai.cpp; ai_qrs.c is compiled unmodified. */
#include <stdint.h>
#include <string.h>
#include "ai_qrs/ai_qrs.h"

#define COMB_TAPS 10
static double bq(double x, double b0, double b1, double b2, double a1, double a2,
                 double *w1, double *w2) {
    double w = x - a1 * (*w1) - a2 * (*w2);
    double y = b0 * w + b1 * (*w1) + b2 * (*w2);
    *w2 = *w1; *w1 = w; return y;
}

int run_chain(const float *x500, int n500, float *stream250, int32_t *fire_sc,
              int32_t *r_sc_out, int max_fires, int min_rr_250) {
    float cb1[COMB_TAPS] = {0}, cb2[COMB_TAPS] = {0};
    float cs1 = 0.f, cs2 = 0.f; int ci1 = 0, ci2 = 0;
    double hp_w1 = 0, hp_w2 = 0, lp_w1 = 0, lp_w2 = 0;
    double ahp_w1 = 0, ahp_w2 = 0, alp_w1 = 0, alp_w2 = 0;
    double aih_w1 = 0, aih_w2 = 0;
    uint32_t decim = 0, sc = 0, last_r = 0;
    uint32_t pend[4]; int pend_n = 0; int nf = 0;
    const uint32_t half = 125;
    aiqrsInit(NULL);
    for (int i = 0; i < n500; i++) {
        float x = x500[i];
        cs1 -= cb1[ci1]; cb1[ci1] = x; cs1 += x; ci1 = (ci1 + 1) % COMB_TAPS;
        float y1 = cs1 / (float)COMB_TAPS;
        cs2 -= cb2[ci2]; cb2[ci2] = y1; cs2 += y1; ci2 = (ci2 + 1) % COMB_TAPS;
        float comb = cs2 / (float)COMB_TAPS;
        /* display/HR chain -> trigger */
        float f = (float)bq(comb, 0.9955669720176472, -1.9911339440352944, 0.9955669720176472,
                            -1.9911142922016536, 0.9911535958689355, &hp_w1, &hp_w2);
        f = (float)bq(f, 0.046131802093312926, 0.09226360418662585, 0.046131802093312926,
                      -1.3072850288493234, 0.4918122372225752, &lp_w1, &lp_w2);
        /* AI chain */
        float a = (float)bq(comb, 0.9995558103876084, -1.9991116207752169, 0.9995558103876084,
                            -1.9991114234707954, 0.9991118180796384, &ahp_w1, &ahp_w2);
        a = (float)bq(a, 0.046131802093312926, 0.09226360418662585, 0.046131802093312926,
                      -1.3072850288493234, 0.4918122372225752, &alp_w1, &alp_w2);
        /* ecg_ai_feed_sample */
        if ((decim++ & 1) == 0) {
            float v = (float)bq(a, 0.99115359510166301, -1.982307190203326, 0.99115359510166301,
                                -1.9822289297925284, 0.98238545061412508, &aih_w1, &aih_w2);
            stream250[sc] = v;
            sc++;
            if (pend_n > 0 && sc >= pend[0]) {
                uint32_t due = pend[0];
                for (int k = 1; k < pend_n; k++) pend[k - 1] = pend[k];
                pend_n--;
                if (sc >= 250 && nf < max_fires) {
                    fire_sc[nf] = (int32_t)sc;
                    r_sc_out[nf] = (int32_t)(due - half);
                    nf++;
                }
            }
        }
        /* aiqrsProcess + ecg_ai_on_rpeak_at(r_lag/2) */
        AIQRS_Result r = aiqrsProcess(f);
        if (r.fired) {
            uint32_t lag = (uint32_t)r.r_lag / 2u;
            if (lag > half - 5u) lag = half - 5u;
            if (lag <= sc) {
                uint32_t rs = sc - lag;
                if (!(last_r != 0 && (rs - last_r) < (uint32_t)min_rr_250)) {
                    last_r = rs;
                    if (pend_n < 4) pend[pend_n++] = rs + half;
                }
            }
        }
    }
    return nf;
}
