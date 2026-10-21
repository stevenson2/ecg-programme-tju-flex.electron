#pragma once
#include <stdint.h>
/* Host clock. fw_replay_host.cpp owns g_host_time_us and steps 2 ms per sample. */
extern int64_t g_host_time_us;
static inline int64_t esp_timer_get_time(void) { return g_host_time_us; }
