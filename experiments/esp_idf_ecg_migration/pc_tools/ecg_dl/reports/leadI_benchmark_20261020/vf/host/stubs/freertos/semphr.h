#pragma once
#include "FreeRTOS.h"
typedef void* SemaphoreHandle_t;
static inline SemaphoreHandle_t xSemaphoreCreateMutex(void) { return NULL; }
static inline BaseType_t xSemaphoreTake(SemaphoreHandle_t, TickType_t) { return pdFALSE; }
static inline BaseType_t xSemaphoreGive(SemaphoreHandle_t) { return pdTRUE; }
