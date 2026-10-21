#pragma once
/* Minimal FreeRTOS stand-ins so vf_detect.cpp compiles on the host.
 * Mutex/queue creation returns NULL, so vfEnsureWorker() never starts the
 * worker and vfProcess() takes the synchronous path (the else branch at the
 * bottom of vfProcess). That path is the one the firmware uses when the
 * worker failed to start, and it is the decision logic we score. */
#include <stdint.h>
#include <stddef.h>
typedef int BaseType_t;
typedef unsigned int UBaseType_t;
typedef uint32_t TickType_t;
#ifndef pdTRUE
#define pdTRUE 1
#endif
#ifndef pdFALSE
#define pdFALSE 0
#endif
#ifndef pdPASS
#define pdPASS pdTRUE
#endif
#ifndef pdFAIL
#define pdFAIL pdFALSE
#endif
#ifndef portMAX_DELAY
#define portMAX_DELAY ((TickType_t)0xFFFFFFFFu)
#endif
#ifndef pdMS_TO_TICKS
#define pdMS_TO_TICKS(ms) ((TickType_t)(ms))
#endif
