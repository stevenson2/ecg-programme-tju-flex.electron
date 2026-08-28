#ifndef ARDUINO_COMPAT_H
#define ARDUINO_COMPAT_H
/* Minimal Arduino-compatible shim for ESP-IDF core algorithm port. */
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "esp_timer.h"

typedef bool boolean;
typedef uint8_t byte;
#define HIGH 1
#define LOW 0

static inline uint32_t millis(void) { return (uint32_t)(esp_timer_get_time() / 1000); }
static inline uint32_t micros(void) { return (uint32_t)esp_timer_get_time(); }

template<typename T>
static inline T constrain(T x, T a, T b) { return x < a ? a : (x > b ? b : x); }

class SerialCompat {
public:
    void begin(long) {}
    void print(const char *s) { fputs(s, stdout); }
    void print(int v) { printf("%d", v); }
    void print(unsigned v) { printf("%u", v); }
    void print(long v) { printf("%ld", v); }
    void print(unsigned long v) { printf("%lu", v); }
    void print(float v) { printf("%f", v); }
    void print(double v) { printf("%f", v); }
    void println() { fputc('\n', stdout); }
    void println(const char *s) { print(s); println(); }
    void println(int v) { print(v); println(); }
    void println(unsigned v) { print(v); println(); }
    void println(long v) { print(v); println(); }
    void println(unsigned long v) { print(v); println(); }
    void println(float v) { print(v); println(); }
    void println(double v) { print(v); println(); }
};
extern SerialCompat Serial;
#define Serial SerialCompat()
#endif
