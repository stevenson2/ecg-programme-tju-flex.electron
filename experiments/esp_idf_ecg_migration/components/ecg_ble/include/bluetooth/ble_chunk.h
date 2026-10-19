/**
 * @file ble_chunk.h
 * @brief Split a notify payload on ';' or '\\n' so each chunk fits ATT_MTU-3.
 *
 * NimBLE does not fragment one notify. A chunk that would cut a CSV frame
 * walks back to the previous terminator. If a single segment has no
 * terminator inside the budget, a raw fragment is returned; the app
 * reassembles until ';' or newline.
 */
#ifndef BLE_CHUNK_H
#define BLE_CHUNK_H

#include <stddef.h>
#include <stdint.h>

static inline size_t ble_next_chunk(const char *msg, size_t off, size_t len,
                                    size_t max_payload) {
    if (!msg || off >= len || max_payload == 0) return 0;
    size_t remain = len - off;
    if (remain <= max_payload) return remain;
    size_t end = off + max_payload;
    for (size_t i = end; i > off; i--) {
        char c = msg[i - 1];
        if (c == ';' || c == '\n') return i - off;
    }
    return max_payload;
}

#endif /* BLE_CHUNK_H */
