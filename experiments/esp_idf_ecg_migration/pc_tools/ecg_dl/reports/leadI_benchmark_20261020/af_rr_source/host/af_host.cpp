// Host harness for af_detect.cpp. Reads "rr sqi" text lines, prints
// "t_end nRr cv entropy score label reason" when a window is attempted.
#include <stdio.h>
#include <stdint.h>
#include "arduino_compat.h"
#include "heartrate/heartrate.h"
#include "af_detect/af_detect.h"

int64_t g_host_time_us = 0;

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    afInit();
    float rr, sqi;
    double t = 0.0;
    while (scanf("%f %f", &rr, &sqi) == 2) {
        HR_Result hr = {};
        hr.beatDetected = true;
        hr.rrInterval = rr;
        hr.sqi = sqi;
        if (rr > 0.0f && rr <= 3.0f) t += rr;
        AF_Result af = afProcess(&hr);
        if (af.windowReady || af.nRr > 0) {
            printf("%.4f %u %.6f %.6f %.6f %u\n",
                   t, (unsigned)af.nRr, af.cv, af.entropy, af.score,
                   (unsigned)af.label);
        }
    }
    return 0;
}
