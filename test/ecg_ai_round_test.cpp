/**
 * INT8 fill rounding must match numpy.round (half to even), including negatives.
 *
 *   g++ -std=c++17 \
 *     -I experiments/esp_idf_ecg_migration/components/ecg_ai/include \
 *     test/ecg_ai_round_test.cpp -o /tmp/ai_round_test && /tmp/ai_round_test
 */
#include "ecg_ai_round.h"

#include <cstdio>

static int g_fail = 0;

static void check(float x, int32_t expect) {
    int32_t got = ecg_ai_round_half_even(x);
    if (got != expect) {
        std::printf("FAIL: round(%g) = %d, expected %d\n",
                    (double)x, (int)got, (int)expect);
        g_fail++;
    }
}

int main(void) {
    /* Values checked against numpy.round on this tree. */
    check(-12.0f, -12);
    check(-12.3f, -12);
    check(-12.5f, -12);
    check(-12.7f, -13);
    check(-1.5f, -2);
    check(-0.5f, 0);
    check(0.0f, 0);
    check(0.5f, 0);
    check(1.5f, 2);
    check(2.5f, 2);
    check(12.5f, 12);
    check(13.5f, 14);

    /* Truncating (int)(x + 0.5) maps every negative non-integer one LSB high. */
    float neg = -12.7f;
    int trunc = (int)(neg + 0.5f);
    if (trunc == ecg_ai_round_half_even(neg)) {
        std::printf("FAIL: negative rounding still matches truncating cast (%d)\n", trunc);
        g_fail++;
    }

    if (g_fail == 0) {
        std::printf("PASS ai round\n");
        return 0;
    }
    std::printf("%d failure(s)\n", g_fail);
    return 1;
}
