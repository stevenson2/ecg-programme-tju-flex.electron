/*
 * SPDX-FileCopyrightText: 2020-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <esp_nn.h>
#include "test_utils.h"


void esp_nn_fully_connected_s8_test()
{
    uint32_t total_c = 0, total_opt = 0;
    /* prepare data */
    uint16_t row_len = 256 + 8 + 7; /* odd len to test unaligned+left-over */
    const int32_t max_out_ch = 16;
    const int32_t max_row_len = 271;
    uint16_t out_channels = 3;

    /* Use heap-allocated aligned buffers (matches TFLite real-world usage) */
    int8_t *input_orig = malloc(max_row_len + 16);
    int8_t *filter_orig = malloc(max_row_len * max_out_ch + 16);
    int8_t *out_c_orig = malloc(max_out_ch + 16);
    int8_t *out_opt_orig = malloc(max_out_ch + 16);
    if (!input_orig || !filter_orig || !out_c_orig || !out_opt_orig) {
        printf(ANSI_COLOR_RED"%s allocations failed\n"ANSI_COLOR_RESET, __FUNCTION__);
        goto fc_s8_cleanup;
    }
    int8_t *input = (int8_t *)(((uint32_t)input_orig + 15) & ~15);
    int8_t *filter_data = (int8_t *)(((uint32_t)filter_orig + 15) & ~15);
    int8_t *output_c = (int8_t *)(((uint32_t)out_c_orig + 15) & ~15);
    int8_t *output_opt = (int8_t *)(((uint32_t)out_opt_orig + 15) & ~15);
    int32_t activation_min = -128;
    int32_t activation_max = 127;
    int32_t input_offset = 5; /* default non-zero, swept per iteration below */
    int32_t filter_offset = 0;
    int32_t out_shift = -10;
    int32_t out_offset = 5;
    int32_t out_mult = 0x59e492c4;
    /* sweep input_offset over realistic TFLite range incl. boundaries */
    const int32_t input_offsets[] = {0, 5, -1, 1, 127, -128, 110};
    printf("\n######## Running %s ##########\n", __FUNCTION__);
    for (int itr = 0; itr < 15; itr++) {
        input_offset = input_offsets[itr % (int)(sizeof(input_offsets) / sizeof(input_offsets[0]))];
        /* TFLite weights are symmetric (filter_offset == 0) almost always;
         * a couple of iterations exercise the non-zero filter_offset SIMD path. */
        filter_offset = (itr == 9 || itr == 12) ? 5 : 0;
        out_mult = INT32_MAX / row_len + rand() % INT16_MAX;
        switch (itr) {
        case 0:
            out_shift = -10;
            break;
        case 1:
            out_shift = SHIFT_MIN;
            break;
        case 2:
            out_shift = SHIFT_MAX;
            break;
        case 3:
            out_shift = 0;
            break;
        case 4:
            row_len = 1;
            out_channels = 16;
            out_shift = -10 + rand() % 5;
            break;
        case 5:
            row_len = 16;
            out_channels = 8;
            out_shift = -10 + rand() % 5;
            break;
        case 6:
            row_len = 8;
            out_channels = 8;
            out_shift = -10 + rand() % 5;
            break;
        case 7:
            row_len = 8;
            out_channels = 15;
            out_shift = -10 + rand() % 5;
            break;
        case 8:
            row_len = 8;
            out_channels = 1;
            out_shift = -10 + rand() % 5;
            break;
        default:
            row_len = rand() % 7 + 1;
            out_channels = 8;
            out_shift = -10 + rand() % 5;
            break;
        }
        if (itr == 0) {
            out_shift = SHIFT_MAX;
        }
        /* Generate input and filter data */
        for (int i = 0; i < row_len; ++i) {
            input[i] = rand() % 256 - 128;
        }
        for (int i = 0; i < row_len * out_channels; ++i) {
            filter_data[i] = rand() % 256 - 128;
        }

        /* enable profiler */
        profile_c_start();

        /* C function */
        esp_nn_fully_connected_s8_ansi(input, input_offset, row_len, filter_data, filter_offset,
                                    NULL, output_c, out_channels, out_offset, out_shift, out_mult,
                                    activation_min, activation_max);

        total_c = profile_c_end();
        profile_opt_start();

        /* Optimized function */
        esp_nn_fully_connected_s8(input, input_offset, row_len, filter_data, filter_offset,
                                NULL, output_opt, out_channels, out_offset, out_shift, out_mult,
                                activation_min, activation_max);

        /* disable profiler */
        total_opt = profile_opt_end();

        bool ret = CHECK_EQUAL(output_c, output_opt, out_channels);
        if (ret == false) {
            printf(ANSI_COLOR_RED"[%3d] failed\n"ANSI_COLOR_RESET, itr);
#if 0
            printf("Output: \n");
            PRINT_ARRAY_HEX(output_opt, out_channels, 1);
            printf("Expected: \n");
            PRINT_ARRAY_HEX(output_c, out_channels, 1);
            printf("Input:\n");
            PRINT_ARRAY_HEX(input, row_len, 1);
            printf("Filter data:\n");
            PRINT_ARRAY_HEX(filter_data, row_len, out_channels);
            printf("Out shift: %d\n", out_shift);
            printf("Out mult: %x\n", out_mult);
#endif
            goto fc_s8_cleanup;
        }
        printf(ANSI_COLOR_GREEN"[%3d] passed [row_len %"PRIu16", out_ch %"PRIu16", in_off %4"PRId32"]"ANSI_COLOR_RESET,
               itr, row_len, out_channels, input_offset);
        printf("\tcycles: c %8"PRIu32", opt %8"PRIu32"\n", total_c, total_opt);
    }

fc_s8_cleanup:
    if (input_orig) {
        free(input_orig);
    }
    if (filter_orig) {
        free(filter_orig);
    }
    if (out_c_orig) {
        free(out_c_orig);
    }
    if (out_opt_orig) {
        free(out_opt_orig);
    }
}

void esp_nn_fully_connected_per_ch_s8_test()
{
    uint32_t total_c = 0, total_opt = 0;
    /* prepare data */
    uint16_t row_len = 256 + 8 + 7; /* odd len to test unaligned+left-over */
    const int32_t max_out_ch = 16;
    const int32_t max_row_len = 271;
    uint16_t out_channels = 3;

    /* Use heap-allocated aligned buffers (matches TFLite real-world usage) */
    int8_t *input_orig = malloc(max_row_len + 16);
    int8_t *filter_orig = malloc(max_row_len * max_out_ch + 16);
    int8_t *out_c_orig = malloc(max_out_ch + 16);
    int8_t *out_opt_orig = malloc(max_out_ch + 16);
    if (!input_orig || !filter_orig || !out_c_orig || !out_opt_orig) {
        printf(ANSI_COLOR_RED"%s allocations failed\n"ANSI_COLOR_RESET, __FUNCTION__);
        goto fc_per_ch_s8_buffers_cleanup;
    }
    int8_t *input = (int8_t *)(((uint32_t)input_orig + 15) & ~15);
    int8_t *filter_data = (int8_t *)(((uint32_t)filter_orig + 15) & ~15);
    int8_t *output_c = (int8_t *)(((uint32_t)out_c_orig + 15) & ~15);
    int8_t *output_opt = (int8_t *)(((uint32_t)out_opt_orig + 15) & ~15);
    int32_t activation_min = -128;
    int32_t activation_max = 127;
    int32_t input_offset = 5; /* default non-zero, swept per iteration below */
    int32_t filter_offset = 0;
    int32_t out_offset = 7;

    int32_t* out_mult = NULL;
    int32_t* out_shift = NULL;

    const int32_t input_offsets[] = {0, 5, -1, 1, 127, -128, 110};
    printf("\n######## Running %s ##########\n", __FUNCTION__);
    for (int itr = 0;  itr < 15; itr++) {
        input_offset = input_offsets[itr % (int)(sizeof(input_offsets) / sizeof(input_offsets[0]))];
        filter_offset = (itr == 9 || itr == 12) ? 5 : 0;
        int32_t out_shift_val = 0;
        switch (itr) {
        case 0:
            out_shift_val = -10;
            break;
        case 1:
            out_shift_val = SHIFT_MIN;
            break;
        case 2:
            out_shift_val = SHIFT_MAX;
            break;
        case 3:
            out_shift_val = 0;
            break;
        case 4:
            row_len = 1;
            out_channels = 16;
            break;
        case 5:
            row_len = 16;
            out_channels = 8;
            break;
        case 6:
            row_len = 8;
            out_channels = 8;
            break;
        case 7:
            row_len = 8;
            out_channels = 15;
            break;
        case 8:
            row_len = 8;
            out_channels = 1;
            break;
        default:
            row_len = rand() % 7 + 1;
            out_channels = 8;
            break;
        }

        out_mult = ESP_NN_TEST_ALLOC(out_channels * sizeof(int32_t));
        out_shift = ESP_NN_TEST_ALLOC(out_channels * sizeof(int32_t));

        if (out_shift == NULL || out_mult == NULL) {
            printf(ANSI_COLOR_RED"out_shift/out_mult allocations failed\n"ANSI_COLOR_RESET);
            goto fully_connected_per_ch_cleanup;
        }

        for (int i = 0; i < out_channels; i++) {
            out_mult[i] = INT32_MAX / row_len + rand() % INT16_MAX;
            if (i < 4) {
                out_shift[i] = out_shift_val;
            } else {
                out_shift[i] = -10 + rand() % 5;
            }
        }

        /* Generate input and filter data */
        for (int i = 0; i < row_len; ++i) {
            input[i] = rand() % 256 - 128;
        }
        for (int i = 0; i < row_len * out_channels; ++i) {
            filter_data[i] = rand() % 256 - 128;
        }
        
        /* enable profiler */
        profile_c_start();

        /* C function */
        esp_nn_fully_connected_per_ch_s8_ansi(input, input_offset, row_len, filter_data, filter_offset,
                                    NULL, output_c, out_channels, out_offset, out_shift, out_mult,
                                    activation_min, activation_max);

        total_c = profile_c_end();
        profile_opt_start();

        /* Optimized function */
        esp_nn_fully_connected_per_ch_s8(input, input_offset, row_len, filter_data, filter_offset,
                                NULL, output_opt, out_channels, out_offset, out_shift, out_mult,
                                activation_min, activation_max);

        /* disable profiler */
        total_opt = profile_opt_end();

        bool ret = CHECK_EQUAL(output_c, output_opt, out_channels);
        if (ret == false) {
            printf(ANSI_COLOR_RED"[%3d] failed\n"ANSI_COLOR_RESET, itr);
            goto fully_connected_per_ch_cleanup;
        }
        printf(ANSI_COLOR_GREEN"[%3d] passed [row_len %"PRIu16", out_ch %"PRIu16", in_off %4"PRId32"]"ANSI_COLOR_RESET,
               itr, row_len, out_channels, input_offset);
        printf("\tcycles: c %8"PRIu32", opt %8"PRIu32"\n", total_c, total_opt);

    fully_connected_per_ch_cleanup:
        if (out_shift) {
            free(out_shift);
        }
        if (out_mult) {
            free(out_mult);
        }
    }

fc_per_ch_s8_buffers_cleanup:
    if (input_orig) {
        free(input_orig);
    }
    if (filter_orig) {
        free(filter_orig);
    }
    if (out_c_orig) {
        free(out_c_orig);
    }
    if (out_opt_orig) {
        free(out_opt_orig);
    }
}

/**
 * Regression sweep for esp-nn#34: per-channel/per-tensor FC reported to give
 * memory-layout-dependent wrong results for a narrow output width (out_ch = 2)
 * at a large row_len.
 *
 * The other FC tests always 16-byte align `input` and `filter_data`, so two
 * things are never exercised for a TFLite-shaped geometry:
 *   a) the `(uintptr_t)input_data & 15` fast-path predicate in the S3
 *      dispatcher, i.e. the fallback into the s16 assembly, and
 *   b) narrow out_ch at a row_len that is a clean multiple of 16 (every filter
 *      row aligned, so the unaligned-filter QUP path is not entered either).
 *
 * This sweeps input and filter misalignment independently, and out_ch across
 * multiples and non-multiples of 8, so an alignment-dependent failure and an
 * out_ch-granularity failure can be told apart. Output buffers are guarded to
 * catch out-of-bounds writes past a 2-byte output.
 */

#define FC_ALIGN_ROW_LEN    1280    /* esp-nn#34 geometry: 1280 -> 2 */
#define FC_ALIGN_MAX_OUT_CH 16
#define FC_ALIGN_GUARD      32
#define FC_ALIGN_TRIALS     8

typedef struct {
    int mismatch_pt;    /* per-tensor kernel vs its ansi reference */
    int mismatch_pc;    /* per-channel kernel vs its ansi reference */
    int guard_pt;       /* guard bytes clobbered by per-tensor kernel */
    int guard_pc;       /* guard bytes clobbered by per-channel kernel */
} fc_align_result_t;

/* Runs FC_ALIGN_TRIALS randomized trials for one (row_len, out_ch, alignment)
 * point and accumulates mismatches/guard corruption into `res`. */
static void fc_align_run_point(int8_t *input, int8_t *filter_data,
                               int8_t *out_region, int8_t *out_ref,
                               uint16_t row_len, uint16_t out_channels,
                               int32_t *out_mult, int32_t *out_shift,
                               fc_align_result_t *res)
{
    /* Quantization params taken from the esp-nn#34 report. */
    const int32_t input_offset = -3;
    const int32_t filter_offset = 0;    /* TFLite int8 weights are symmetric */
    const int32_t out_offset = -128;
    const int32_t activation_min = -128;
    const int32_t activation_max = 127;
    const int32_t mult = 1355715584;
    const int32_t shift = -6;

    int8_t *out = out_region + FC_ALIGN_GUARD;
    /* TFLite FULLY_CONNECTED always carries a bias; the older FC tests only
     * cover the bias == NULL case, so exercise a real one here. */
    int32_t bias[FC_ALIGN_MAX_OUT_CH];

    for (int c = 0; c < out_channels; c++) {
        out_mult[c] = mult - c * 12345;
        out_shift[c] = shift;
    }

    for (int t = 0; t < FC_ALIGN_TRIALS; t++) {
        for (int i = 0; i < row_len; ++i) {
            input[i] = rand() % 256 - 128;
        }
        for (int i = 0; i < row_len * out_channels; ++i) {
            filter_data[i] = rand() % 256 - 128;
        }
        for (int c = 0; c < out_channels; c++) {
            bias[c] = (rand() % 256 - 128) * 64;
        }

        /* ---- per-tensor kernel ---- */
        memset(out_region, 0xA5, FC_ALIGN_GUARD * 2 + out_channels);
        esp_nn_fully_connected_s8(input, input_offset, row_len, filter_data,
                                  filter_offset, bias, out, out_channels,
                                  out_offset, shift, mult,
                                  activation_min, activation_max);
        esp_nn_fully_connected_s8_ansi(input, input_offset, row_len, filter_data,
                                       filter_offset, bias, out_ref, out_channels,
                                       out_offset, shift, mult,
                                       activation_min, activation_max);
        if (!CHECK_EQUAL(out, out_ref, out_channels)) {
            res->mismatch_pt++;
        }
        for (int i = 0; i < FC_ALIGN_GUARD; i++) {
            if (out_region[i] != (int8_t) 0xA5 ||
                out_region[FC_ALIGN_GUARD + out_channels + i] != (int8_t) 0xA5) {
                res->guard_pt++;
                break;
            }
        }

        /* ---- per-channel kernel ---- */
        memset(out_region, 0xA5, FC_ALIGN_GUARD * 2 + out_channels);
        esp_nn_fully_connected_per_ch_s8(input, input_offset, row_len, filter_data,
                                         filter_offset, bias, out, out_channels,
                                         out_offset, out_shift, out_mult,
                                         activation_min, activation_max);
        esp_nn_fully_connected_per_ch_s8_ansi(input, input_offset, row_len,
                                              filter_data, filter_offset, bias,
                                              out_ref, out_channels, out_offset,
                                              out_shift, out_mult,
                                              activation_min, activation_max);
        if (!CHECK_EQUAL(out, out_ref, out_channels)) {
            res->mismatch_pc++;
        }
        for (int i = 0; i < FC_ALIGN_GUARD; i++) {
            if (out_region[i] != (int8_t) 0xA5 ||
                out_region[FC_ALIGN_GUARD + out_channels + i] != (int8_t) 0xA5) {
                res->guard_pc++;
                break;
            }
        }
    }
}

/* `what` may be NULL when the caller has already printed its own label. */
static void fc_align_report(const char *what, int value,
                            const fc_align_result_t *res)
{
    bool ok = (res->mismatch_pt == 0 && res->mismatch_pc == 0 &&
               res->guard_pt == 0 && res->guard_pc == 0);
    printf("%s", ok ? ANSI_COLOR_GREEN : ANSI_COLOR_RED);
    if (what) {
        printf("[%-14s %2d] ", what, value);
    }
    printf("per-tensor %d/%d, per-channel %d/%d, guard pt %d pc %d%s\n",
           res->mismatch_pt, FC_ALIGN_TRIALS, res->mismatch_pc, FC_ALIGN_TRIALS,
           res->guard_pt, res->guard_pc, ANSI_COLOR_RESET);
}

void esp_nn_fully_connected_align_s8_test()
{
    const size_t filter_bytes = (size_t) FC_ALIGN_ROW_LEN * FC_ALIGN_MAX_OUT_CH;
    int8_t *input_orig = ESP_NN_TEST_ALLOC(FC_ALIGN_ROW_LEN + 32);
    int8_t *filter_orig = ESP_NN_TEST_ALLOC(filter_bytes + 32);
    int8_t *out_region = ESP_NN_TEST_ALLOC(FC_ALIGN_GUARD * 2 + FC_ALIGN_MAX_OUT_CH);
    int8_t *out_ref = ESP_NN_TEST_ALLOC(FC_ALIGN_MAX_OUT_CH);
    int32_t *out_mult = ESP_NN_TEST_ALLOC(FC_ALIGN_MAX_OUT_CH * sizeof(int32_t));
    int32_t *out_shift = ESP_NN_TEST_ALLOC(FC_ALIGN_MAX_OUT_CH * sizeof(int32_t));

    if (!input_orig || !filter_orig || !out_region || !out_ref ||
        !out_mult || !out_shift) {
        printf(ANSI_COLOR_RED"%s allocations failed\n"ANSI_COLOR_RESET, __FUNCTION__);
        goto fc_align_cleanup;
    }

    int8_t *input_base = (int8_t *)(((uintptr_t) input_orig + 15) & ~(uintptr_t) 15);
    int8_t *filter_base = (int8_t *)(((uintptr_t) filter_orig + 15) & ~(uintptr_t) 15);

    printf("\n######## Running %s ##########\n", __FUNCTION__);
    printf("geometry: row_len %d, filter_offset 0, input_offset -3 (esp-nn#34)\n",
           FC_ALIGN_ROW_LEN);

    /* Sweep A: input misalignment 0..15, filter 16-byte aligned, out_ch = 2.
     * Exercises the `input_data & 15` dispatcher predicate. */
    printf("-- sweep A: input misalignment, out_ch 2, filter aligned --\n");
    for (int off = 0; off < 16; off++) {
        fc_align_result_t res = {0};
        fc_align_run_point(input_base + off, filter_base, out_region, out_ref,
                           FC_ALIGN_ROW_LEN, 2, out_mult, out_shift, &res);
        fc_align_report("input off", off, &res);
    }

    /* Sweep B: filter misalignment 0..15, input 16-byte aligned, out_ch = 2. */
    printf("-- sweep B: filter misalignment, out_ch 2, input aligned --\n");
    for (int off = 0; off < 16; off++) {
        fc_align_result_t res = {0};
        fc_align_run_point(input_base, filter_base + off, out_region, out_ref,
                           FC_ALIGN_ROW_LEN, 2, out_mult, out_shift, &res);
        fc_align_report("filter off", off, &res);
    }

    /* Sweep C: out_ch across multiples and non-multiples of 8, at both an
     * aligned and a misaligned input, to separate an out_ch-granularity
     * failure from an alignment failure. */
    const uint16_t out_chs[] = {1, 2, 3, 7, 8, 15, 16};
    for (int a = 0; a < 2; a++) {
        int off = a ? 4 : 0;
        printf("-- sweep C: out_ch sweep, input off %d --\n", off);
        for (int i = 0; i < (int)(sizeof(out_chs) / sizeof(out_chs[0])); i++) {
            fc_align_result_t res = {0};
            fc_align_run_point(input_base + off, filter_base, out_region, out_ref,
                               FC_ALIGN_ROW_LEN, out_chs[i], out_mult, out_shift,
                               &res);
            fc_align_report("out_ch", out_chs[i], &res);
        }
    }

    /* Sweep D0: row_len below one SIMD width, where the header now promises
     * correctness too. Misaligned small rows go to ansi, 8-aligned ones reach
     * the s16 assembly's scalar tail — both need covering. */
    const uint16_t small_rows[] = {1, 5, 8, 15, 16, 17};
    printf("-- sweep D0: small row_len, out_ch 2, input off 0/1/8 --\n");
    for (int i = 0; i < (int)(sizeof(small_rows) / sizeof(small_rows[0])); i++) {
        for (int j = 0; j < 3; j++) {
            const int off = (int[]){0, 1, 8}[j];
            fc_align_result_t res = {0};
            fc_align_run_point(input_base + off, filter_base, out_region, out_ref,
                               small_rows[i], 2, out_mult, out_shift, &res);
            printf("  row_len %2u / in off %d: ", small_rows[i], off);
            fc_align_report(NULL, 0, &res);
        }
    }

    /* Sweep D: row_len not a multiple of 16, so filter rows are unaligned too
     * and a misaligned input cannot be rescued by swapping the operands. */
    const int align_offs[] = {0, 1, 4, 8};
    printf("-- sweep D: row_len 1271 (odd), out_ch 2, input misalignment --\n");
    for (int i = 0; i < (int)(sizeof(align_offs) / sizeof(align_offs[0])); i++) {
        fc_align_result_t res = {0};
        fc_align_run_point(input_base + align_offs[i], filter_base, out_region,
                           out_ref, 1271, 2, out_mult, out_shift, &res);
        fc_align_report("input off", align_offs[i], &res);
    }

    /* Sweep E: input and filter both misaligned. */
    printf("-- sweep E: input + filter misalignment, out_ch 2 --\n");
    for (int i = 0; i < (int)(sizeof(align_offs) / sizeof(align_offs[0])); i++) {
        for (int j = 0; j < (int)(sizeof(align_offs) / sizeof(align_offs[0])); j++) {
            fc_align_result_t res = {0};
            fc_align_run_point(input_base + align_offs[i],
                               filter_base + align_offs[j], out_region, out_ref,
                               FC_ALIGN_ROW_LEN, 2, out_mult, out_shift, &res);
            printf("  in off %d / filt off %d: ", align_offs[i], align_offs[j]);
            fc_align_report(NULL, 0, &res);
        }
    }

fc_align_cleanup:
    if (input_orig) {
        free(input_orig);
    }
    if (filter_orig) {
        free(filter_orig);
    }
    if (out_region) {
        free(out_region);
    }
    if (out_ref) {
        free(out_ref);
    }
    if (out_mult) {
        free(out_mult);
    }
    if (out_shift) {
        free(out_shift);
    }
}
