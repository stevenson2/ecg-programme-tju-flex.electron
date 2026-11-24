#ifndef XL_PIPELINE_H
#define XL_PIPELINE_H
/**
 * @file xl_pipeline.h
 * @brief XL_small deployment contract (CONTRACT.md): window quantisation, the four
 *        scalar features, the 2 s next-beat wait, ensemble averaging and the m-point
 *        decision.  Pure C++ (no ESP-IDF / TFLM dependency) so the very same file is
 *        compiled for the host simulation (sim/) and for the board.
 */
#include <stdint.h>
#include <stddef.h>

#define XL_WIN 250
#define XL_TIMEOUT_SAMPLES 500          /* 2 s at 250 Hz; a next trigger exactly 500 away still counts */
#define XL_N_MODELS 2

typedef struct {
    uint32_t sample_index;              /* 250 Hz sample count at window end (1-based), as AiState::sample_count */
    uint32_t crc32;                     /* zlib CRC-32 of the 250 int8 window bytes */
    int8_t   win[XL_WIN];               /* quantised z-score window (INPUT_SCALE / INPUT_ZP) */
    int8_t   feat[4];                   /* quantised scalars (model feat_input scale / zp) */
    float    feat_f[4];                 /* float32 scalars before quantisation (diagnostics) */
    uint8_t  has_next;                  /* 1: a next trigger within 500 samples was seen */
} XlBeat;

typedef struct {
    float    probs[4];                  /* ensemble mean of the two softmax outputs: N,S,V,X */
    int      verdict;                   /* 0=N 1=S 2=V 3=X (m-point) */
    uint32_t us[XL_N_MODELS];           /* Invoke time of each model */
    int8_t   logits[XL_N_MODELS][4];    /* raw int8 outputs */
} XlResult;

/* Hooks supplied by the integration layer. */
typedef struct {
    /* a beat has been resolved (next RR known or timed out); consumer runs the models */
    void (*beat_ready)(const XlBeat *beat, void *ctx);
    /* run model m (0/1) on (win,feat); write 4 int8 logits; return microseconds; <0 on failure */
    int  (*infer)(int m, const int8_t *win, const int8_t *feat, int8_t out[4], void *ctx);
    void *ctx;
} XlHooks;

/* Quantisation constants (from the int8 TFLite files, CONTRACT.md section 4); see xl_params.h. */
extern const float   XL_IN_SCALE;       /* ecg_input scale  = 0.047701675444841385 */
extern const int32_t XL_IN_ZP;          /* ecg_input zero point = -7 */
extern const float   XL_FEAT_SCALE;     /* feat_input scale = 0.0235294122248888 */
extern const int32_t XL_FEAT_ZP;        /* feat_input zero point = -1 */
extern const float   XL_OUT_SCALE[XL_N_MODELS];
extern const int32_t XL_OUT_ZP[XL_N_MODELS];
extern const double  XL_MP_OFFSET[4];   /* m-point offsets, fitted on dev40 only */

void     xl_pipeline_reset(void);                       /* segment start: forgets all RR / std history */
/* A trigger fired: `raw` is the 250-sample stream window (HP0.5 @250 Hz output, before z-score). */
void     xl_push_window(const float *raw, uint32_t sample_index);
/* Called once per 250 Hz stream sample after any push; resolves a pending beat on timeout. */
void     xl_tick(uint32_t sample_count);
/* End of stream: the last beat has no next trigger. */
void     xl_flush(void);
void     xl_set_hooks(const XlHooks *hooks);

uint32_t xl_crc32(const uint8_t *data, size_t n);
int32_t  xl_round_half_even(float x);
/* Consumer helper: runs both models via hooks->infer, averages the softmax, applies the m-point. */
int      xl_run_beat(const XlBeat *beat, XlResult *out);
/* Format "XL,<idx>,<crc>,f0,f1,f2,f3,pN,pS,pV,pX,<verdict>,us0,us1"; returns length. */
int      xl_format_line(const XlBeat *beat, const XlResult *res, char *buf, size_t cap);

#endif
