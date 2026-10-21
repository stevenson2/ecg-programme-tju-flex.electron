#pragma once
#include "FreeRTOS.h"
typedef void* QueueHandle_t;
static inline QueueHandle_t xQueueCreate(UBaseType_t, UBaseType_t) { return NULL; }
static inline BaseType_t xQueueOverwrite(QueueHandle_t, const void*) { return pdFAIL; }
static inline BaseType_t xQueueReceive(QueueHandle_t, void*, TickType_t) { return pdFALSE; }
