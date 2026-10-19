/**
 * @file ecg_ai_round.h
 * @brief Half-to-even rounding matching numpy.round, including negatives.
 *
 * (int)(x + 0.5f) truncates toward zero, so every negative sample is +1 LSB
 * versus np.round. Z-score makes about half of each window negative.
 */
#ifndef ECG_AI_ROUND_H
#define ECG_AI_ROUND_H

#include <stdint.h>
#include <math.h>

static inline int32_t ecg_ai_round_half_even(float x) {
    float fl = floorf(x);
    float frac = x - fl;
    int32_t base = (int32_t)fl;
    if (frac > 0.5f) return base + 1;
    if (frac < 0.5f) return base;
    if ((base & 1) == 0) return base;
    return base + 1;
}

#endif /* ECG_AI_ROUND_H */
