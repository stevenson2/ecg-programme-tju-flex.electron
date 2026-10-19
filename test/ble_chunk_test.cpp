/**
 * Application-level BLE notify chunking on ';' / '\\n' boundaries.
 *
 *   g++ -std=c++17 \
 *     -I experiments/esp_idf_ecg_migration/components/ecg_ble/include \
 *     test/ble_chunk_test.cpp -o /tmp/ble_chunk_test && /tmp/ble_chunk_test
 */
#include "bluetooth/ble_chunk.h"

#include <cstdio>
#include <cstring>
#include <string>

static int g_fail = 0;

static void expect(bool cond, const char *msg) {
    if (!cond) {
        std::printf("FAIL: %s\n", msg);
        g_fail++;
    }
}

static std::string reassemble(const char *msg, size_t maxp) {
    std::string out;
    size_t len = std::strlen(msg);
    size_t off = 0;
    int guard = 0;
    while (off < len && guard++ < 64) {
        size_t n = ble_next_chunk(msg, off, len, maxp);
        if (n == 0) break;
        out.append(msg + off, n);
        off += n;
    }
    return out;
}

int main(void) {
    const char *two =
        "0.125,-0.010,0.100,75,0,0.90,0,0,0.100,20;"
        "0.130,-0.012,0.110,76,0,0.91,0,1,0.200,16;";
    expect(reassemble(two, 20) == two, "chunks reassemble to the original batch");

    size_t n0 = ble_next_chunk(two, 0, std::strlen(two), 50);
    expect(n0 == 42 && two[n0 - 1] == ';', "chunk ends on the first frame when it fits");
    expect(n0 <= 50, "chunk respects max payload");
    size_t raw20 = ble_next_chunk(two, 0, std::strlen(two), 20);
    expect(raw20 == 20 && two[raw20 - 1] != ';',
           "MTU-20 cuts inside the first frame as a raw fragment");

    const char *long_frame =
        "0.125,-0.010,0.100,75,0,0.90,0,0,0.100,20;";
    size_t raw = ble_next_chunk(long_frame, 0, std::strlen(long_frame), 20);
    expect(raw == 20, "a frame with no boundary inside the MTU is a raw fragment");
    expect(reassemble(long_frame, 20) == long_frame, "raw fragments still reassemble");

    const char *hello = "HELLO,2,1.0,wave;\n";
    expect(reassemble(hello, 8) == hello, "HELLO newline chunks reassemble");

    const char *small = "PING\n";
    expect(ble_next_chunk(small, 0, std::strlen(small), 20) == std::strlen(small),
           "short payload is sent whole");

    if (g_fail == 0) {
        std::printf("PASS ble chunk\n");
        return 0;
    }
    std::printf("%d failure(s)\n", g_fail);
    return 1;
}
