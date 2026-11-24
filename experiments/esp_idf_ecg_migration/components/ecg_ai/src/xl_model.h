#ifndef XL_MODEL_H
#define XL_MODEL_H
/* Device-only: two int8 XL_small TFLM interpreters (one arena each, PSRAM). */
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

bool   xl_model_init(const unsigned char *const models[2], bool use_psram);
void   xl_model_deinit(void);
/* Returns Invoke time in microseconds (>=0) or -1.  win: 250 int8, feat: 4 int8. */
int    xl_model_infer(int m, const int8_t *win, const int8_t *feat, int8_t out[4]);
size_t xl_model_arena_used(int m);
size_t xl_model_arena_size(int m);

#endif
