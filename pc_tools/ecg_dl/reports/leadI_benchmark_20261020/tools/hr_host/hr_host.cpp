// Host harness: firmware applyFilter (HP0.5/LP40 @500Hz) + hrProcess (Pan-Tompkins v6) on 500 Hz float32 stdin.
// Prints sample index (500 Hz domain) at every hr.beatDetected. Time advances 2000 us per sample (as on board).
#include <stdio.h>
#include <stdint.h>
#include "arduino_compat.h"
#include "filter/filter.h"
#include "heartrate/heartrate.h"
int64_t g_host_time_us = 1000000;
SerialCompat Serial_obj;
int main(int argc, char** argv) {
    int use_fw_filter = (argc > 1) ? atoi(argv[1]) : 1;
    filterInit(); hrInit();
    float x; long i = 0; bool first = true;
    while (fread(&x, sizeof(float), 1, stdin) == 1) {
        if (first && use_fw_filter) { filterWarmup(x); first = false; }
        float f = use_fw_filter ? applyFilter(x) : x;
        HR_Result hr = hrProcess(f);
        if (hr.beatDetected) printf("%ld\n", i);
        i++; g_host_time_us += 2000;
    }
    return 0;
}
