// Host harness (mirrors firmware main loop for R-peak triggering):
//   mode 1 (legacy): applyFilter + hrProcess (strict heartrate.cpp v6) -> beatDetected index (500 Hz)
//   mode 2 (permissive AI trigger): applyFilter + aiqrsProcess (ai_qrs.c) -> estimated R index (500 Hz)
//   usage: hr_host <mode> [refr thr_ratio floor tguard tfrac roff]
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "arduino_compat.h"
#include "filter/filter.h"
#include "heartrate/heartrate.h"
#include "ai_qrs/ai_qrs.h"
int64_t g_host_time_us = 1000000;
int main(int argc, char** argv) {
    int mode = (argc > 1) ? atoi(argv[1]) : 1;
    AIQRS_Params p; aiqrsDefaultParams(&p);
    if (argc > 2) p.refractory_samp = atoi(argv[2]);
    if (argc > 3) p.thr_ratio = (float)atof(argv[3]);
    if (argc > 4) p.thr_floor_frac = (float)atof(argv[4]);
    if (argc > 5) p.tguard_samp = atoi(argv[5]);
    if (argc > 6) p.tguard_frac = (float)atof(argv[6]);
    if (argc > 7) p.r_offset_samp = atoi(argv[7]);
    filterInit(); hrInit(); aiqrsInit(&p);
    float x; long i = 0; bool first = true;
    while (fread(&x, sizeof(float), 1, stdin) == 1) {
        if (first) { filterWarmup(x); first = false; }
        float f = applyFilter(x);
        HR_Result hr = hrProcess(f);
        AIQRS_Result a = aiqrsProcess(f);
        if (mode == 1 && hr.beatDetected) printf("%ld\n", i);
        if (mode == 2 && a.fired) printf("%ld\n", i - (long)a.r_lag);
        i++; g_host_time_us += 2000;
    }
    return 0;
}
