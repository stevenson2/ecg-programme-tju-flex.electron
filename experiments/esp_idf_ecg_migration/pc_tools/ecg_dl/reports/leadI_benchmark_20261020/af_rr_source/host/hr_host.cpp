// Host harness: firmware applyFilter + hrProcess / aiqrsProcess at 500 Hz.
// Does NOT change firmware sources. 50 Hz comb from main.cc is omitted
// (US PhysioNet records are 60 Hz; previous host replay did the same).
//
// stdin: float32 little-endian samples @500 Hz, millivolt-scale.
// mode 1 (strict):  "idx rr_s sqi" on beatDetected
// mode 2 (aiqrs):   "idx" on fired (idx is estimated R sample)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "arduino_compat.h"
#include "filter/filter.h"
#include "heartrate/heartrate.h"
#include "ai_qrs/ai_qrs.h"

int64_t g_host_time_us = 1000000;

int main(int argc, char **argv)
{
    int mode = (argc > 1) ? atoi(argv[1]) : 1;
    AIQRS_Params p;
    aiqrsDefaultParams(&p);
    if (argc > 2) p.refractory_samp = atoi(argv[2]);
    if (argc > 3) p.thr_ratio = (float)atof(argv[3]);
    if (argc > 4) p.thr_floor_frac = (float)atof(argv[4]);
    if (argc > 5) p.tguard_samp = atoi(argv[5]);
    if (argc > 6) p.tguard_frac = (float)atof(argv[6]);
    if (argc > 7) p.r_offset_samp = atoi(argv[7]);

    setvbuf(stdout, NULL, _IOFBF, 1 << 20);
    filterInit();
    hrInit();
    aiqrsInit(&p);

    float x;
    long i = 0;
    bool first = true;
    while (fread(&x, sizeof(float), 1, stdin) == 1) {
        if (first) {
            filterWarmup(x);
            first = false;
        }
        float f = applyFilter(x);
        if (mode == 1) {
            HR_Result hr = hrProcess(f);
            if (hr.beatDetected) {
                printf("%ld %.6f %.4f\n", i, hr.rrInterval, hr.sqi);
            }
        } else {
            AIQRS_Result a = aiqrsProcess(f);
            if (a.fired) {
                long idx = i - (long)a.r_lag;
                if (idx < 0) idx = 0;
                printf("%ld\n", idx);
            }
        }
        i++;
        g_host_time_us += 2000;
    }
    return 0;
}
