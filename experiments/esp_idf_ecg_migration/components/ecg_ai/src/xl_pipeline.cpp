#include "xl_pipeline.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "xl_params.h"   /* generated: scales, zero points, XL_MP_OFFSET */

namespace {

constexpr double kStdEps = 1e-6;

struct State {
    XlHooks hooks{};
    /* previous-RR history: rr[k] = anchor[k+1]-anchor[k]; chronological, last 9 kept */
    double   rr_hist[9] = {0};
    int      rr_n = 0;
    double   std_hist[16] = {0};
    int      std_n = 0;
    bool     have_prev = false;
    uint32_t prev_index = 0;
    /* the single pending beat (waiting for its next trigger) */
    bool     pending = false;
    XlBeat   beat{};
    double   cur_rr = 0.0;
    double   med = 0.0;
    bool     med_ok = false;
    double   f1 = 0.0, f4 = 0.0;
};
State g;

double median_of(const double *v, int n) {
    double tmp[16];
    for (int i = 0; i < n; i++) tmp[i] = v[i];
    for (int i = 1; i < n; i++) {           /* insertion sort, n <= 16 */
        double x = tmp[i]; int j = i - 1;
        while (j >= 0 && tmp[j] > x) { tmp[j + 1] = tmp[j]; j--; }
        tmp[j + 1] = x;
    }
    if (n & 1) return tmp[n / 2];
    return 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
}

double log_ratio(double num, double den) {
    if (!(num > 0.0) || !(den > 0.0) || !isfinite(num) || !isfinite(den)) return 0.0;
    double v = log(num / den);
    if (v < -3.0) v = -3.0;
    if (v > 3.0) v = 3.0;
    return v;
}

int8_t quant_feat(float f) {
    int32_t q = xl_round_half_even(f / XL_FEAT_SCALE) + XL_FEAT_ZP;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    return (int8_t)q;
}

void resolve(bool has_next, uint32_t next_index) {
    if (!g.pending) return;
    double f2 = 0.0, f3 = 0.0;
    g.beat.has_next = 0;
    if (has_next) {
        double nxt = (double)(next_index - g.beat.sample_index);
        if (nxt > 0.0 && nxt <= (double)XL_TIMEOUT_SAMPLES) {
            g.beat.has_next = 1;
            f3 = log_ratio(nxt, g.cur_rr);              /* 0 when there is no previous RR */
            if (g.med_ok) f2 = log_ratio(nxt, g.med);
        }
    }
    float ff[4] = {(float)g.f1, (float)f2, (float)f3, (float)g.f4};
    for (int i = 0; i < 4; i++) { g.beat.feat_f[i] = ff[i]; g.beat.feat[i] = quant_feat(ff[i]); }
    g.pending = false;
    if (g.hooks.beat_ready) g.hooks.beat_ready(&g.beat, g.hooks.ctx);
}

}  // namespace

uint32_t xl_crc32(const uint8_t *data, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) {
        c ^= data[i];
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return ~c;
}

int32_t xl_round_half_even(float x) {
    float fl = floorf(x);
    float frac = x - fl;
    int32_t base = (int32_t)fl;
    if (frac > 0.5f) return base + 1;
    if (frac < 0.5f) return base;
    return (base & 1) == 0 ? base : base + 1;
}

void xl_set_hooks(const XlHooks *hooks) { g.hooks = *hooks; }

void xl_pipeline_reset(void) {
    XlHooks h = g.hooks;
    g = State();
    g.hooks = h;
}

void xl_push_window(const float *raw, uint32_t sample_index) {
    /* A trigger arriving while a beat is pending resolves it (distance <= 500 keeps f2/f3). */
    if (g.pending) resolve(true, sample_index);

    /* z-score in float32, exactly as ecg_ai.cpp zscore() / capture.cpp zscore(). */
    float buf[XL_WIN];
    memcpy(buf, raw, sizeof(buf));
    float sum = 0.0f;
    for (int i = 0; i < XL_WIN; i++) sum += buf[i];
    float mean = sum / (float)XL_WIN;
    float var = 0.0f;
    for (int i = 0; i < XL_WIN; i++) { float d = buf[i] - mean; var += d * d; }
    float stdf = sqrtf(var / (float)XL_WIN);
    float safe = (stdf < 1e-6f) ? 1.0f : stdf;
    XlBeat &b = g.beat;
    memset(&b, 0, sizeof(b));
    for (int i = 0; i < XL_WIN; i++) {
        float z = (buf[i] - mean) / safe;
        int32_t q = xl_round_half_even(z / XL_IN_SCALE) + XL_IN_ZP;   /* int8 rails == Z_LO / Z_HI clip */
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        b.win[i] = (int8_t)q;
    }
    b.sample_index = sample_index;
    b.crc32 = xl_crc32((const uint8_t *)b.win, XL_WIN);

    /* f1: log(current previous-RR / median of up to 8 earlier previous-RRs); needs >= 3 earlier. */
    g.cur_rr = 0.0; g.med_ok = false; g.med = 0.0; g.f1 = 0.0;
    if (g.have_prev) {
        g.cur_rr = (double)(sample_index - g.prev_index);
        if (g.rr_n >= 3 && g.cur_rr > 0.0) {
            int n = g.rr_n > 8 ? 8 : g.rr_n;
            g.med = median_of(g.rr_hist + (g.rr_n - n), n);
            g.med_ok = true;
            g.f1 = log_ratio(g.cur_rr, g.med);
        }
        if (g.rr_n < 9) g.rr_hist[g.rr_n++] = g.cur_rr;
        else { memmove(g.rr_hist, g.rr_hist + 1, 8 * sizeof(double)); g.rr_hist[8] = g.cur_rr; }
    }
    /* f4: log(std / median of up to 16 earlier window stds); needs >= 4 earlier; both >= 1e-6. */
    g.f4 = 0.0;
    double cur_std = (double)stdf;
    if (g.std_n >= 4 && cur_std >= kStdEps) {
        double m = median_of(g.std_hist, g.std_n);
        if (m >= kStdEps) g.f4 = log_ratio(cur_std, m);
    }
    if (g.std_n < 16) g.std_hist[g.std_n++] = cur_std;
    else { memmove(g.std_hist, g.std_hist + 1, 15 * sizeof(double)); g.std_hist[15] = cur_std; }

    g.have_prev = true;
    g.prev_index = sample_index;
    g.pending = true;
}

void xl_tick(uint32_t sample_count) {
    if (g.pending && (sample_count - g.beat.sample_index) > (uint32_t)XL_TIMEOUT_SAMPLES) resolve(false, 0);
}

void xl_flush(void) { if (g.pending) resolve(false, 0); }

int xl_run_beat(const XlBeat *beat, XlResult *out) {
    memset(out, 0, sizeof(*out));
    float acc[4] = {0, 0, 0, 0};
    for (int m = 0; m < XL_N_MODELS; m++) {
        int us = g.hooks.infer ? g.hooks.infer(m, beat->win, beat->feat, out->logits[m], g.hooks.ctx) : -1;
        if (us < 0) return -1;
        out->us[m] = (uint32_t)us;
        float lg[4], mx = -1e30f, s = 0.0f;
        for (int k = 0; k < 4; k++) {
            lg[k] = ((float)out->logits[m][k] - (float)XL_OUT_ZP[m]) * XL_OUT_SCALE[m];
            if (lg[k] > mx) mx = lg[k];
        }
        for (int k = 0; k < 4; k++) { lg[k] = expf(lg[k] - mx); s += lg[k]; }
        for (int k = 0; k < 4; k++) acc[k] += lg[k] / s;
    }
    int best = 0; double bv = -1e300;
    for (int k = 0; k < 4; k++) {
        out->probs[k] = acc[k] / (float)XL_N_MODELS;
        float p = out->probs[k] < 1e-12f ? 1e-12f : out->probs[k];
        double e = (double)logf(p) + XL_MP_OFFSET[k];
        if (e > bv) { bv = e; best = k; }
    }
    out->verdict = best;
    return 0;
}

int xl_format_line(const XlBeat *b, const XlResult *r, char *buf, size_t cap) {
    static const char kV[4] = {'N', 'S', 'V', 'X'};
    return snprintf(buf, cap, "XL,%lu,%08lx,%d,%d,%d,%d,%.5f,%.5f,%.5f,%.5f,%c,%lu,%lu",
                    (unsigned long)b->sample_index, (unsigned long)b->crc32,
                    b->feat[0], b->feat[1], b->feat[2], b->feat[3],
                    (double)r->probs[0], (double)r->probs[1], (double)r->probs[2], (double)r->probs[3],
                    kV[r->verdict], (unsigned long)r->us[0], (unsigned long)r->us[1]);
}
