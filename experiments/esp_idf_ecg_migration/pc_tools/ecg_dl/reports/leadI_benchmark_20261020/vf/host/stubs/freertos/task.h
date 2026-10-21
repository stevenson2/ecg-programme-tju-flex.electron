#pragma once
#include "FreeRTOS.h"
typedef void* TaskHandle_t;
static inline uint32_t ulTaskNotifyTake(BaseType_t, TickType_t) { return 0; }
static inline void xTaskNotifyGive(TaskHandle_t) {}
static inline BaseType_t xTaskCreatePinnedToCore(
    void (*fn)(void*), const char*, uint32_t, void*, int, TaskHandle_t*, int) {
    (void)fn;
    return pdFAIL;
}
