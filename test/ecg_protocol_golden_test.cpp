/**
 * @file ecg_protocol_golden_test.cpp
 * @brief P0-1/P0-3 主机端金样测试：BLE 帧字段定义 + ECGR v1/v2 解析。
 *
 * 数据来源: protocol/golden/ble_frames.json / ecgr_*.bin / ecgr_expected.json
 * 与 Web/App 金样测试共享同一 fixture，三端断言必须一致。
 *
 * 编译运行 (WSL/Windows g++):
 *   g++ -std=c++11 -I experiments/esp_idf_ecg_migration/components/ecg_storage/include \
 *       test/ecg_protocol_golden_test.cpp -o /tmp/ecg_protocol_golden && /tmp/ecg_protocol_golden
 */
#include "storage/ecg_protocol_generated.h"
#include "storage/ecg_recorder_format.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define TEST(name) printf("  TEST: %s ... ", name)
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); g_fail++; return; } } while (0)
#define PASS() do { printf("PASS\n"); g_pass++; } while (0)

/* ---------- 金样 ECGR v2 bytes (与 protocol/golden/ecgr_v2.bin 一致) ---------- */
static std::vector<uint8_t> build_v1() {
    std::vector<uint8_t> b(32 + 7 * 2 + 3, 0);
    b[0]='E'; b[1]='C'; b[2]='G'; b[3]='R'; b[4]=1; b[5]=1;
    _ecgr_write32le(&b[6], 250);
    _ecgr_write32le(&b[10], 1700000000);
    _ecgr_write32le(&b[14], 3);
    _ecgr_write32le(&b[18], 7);
    _ecgr_write32le(&b[22], 2);
    int16_t samples[7] = {0, 1000, -2000, 3000, -4000, 32767, -32768};
    for (int i = 0; i < 7; i++) _ecgr_write16le(&b[32 + i*2], (uint16_t)samples[i]);
    b[32 + 14] = 1; b[32 + 15] = 0; b[32 + 16] = 1;
    return b;
}

static std::vector<uint8_t> build_v2() {
    std::vector<uint8_t> b = build_v1();
    b[4] = 2;
    b[26] = 0x01;
    _ecgr_write32le(&b[22], 3);
    b[32 + 14] = 1; b[32 + 15] = 20; b[32 + 16] = 16;
    return b;
}

static void test_contract_asrc_bits(void) {
    TEST("contract asrc bits (generated header)");
    CHECK(ECG_ASRC_AI == 0x01, "AI bit");
    CHECK(ECG_ASRC_RS == 0x02, "RS bit");
    CHECK(ECG_ASRC_FLAT == 0x04, "FLAT bit");
    CHECK(ECG_ASRC_VF == 0x08, "VF bit");
    CHECK(ECG_ASRC_LEADOFF == 0x10, "LEADOFF bit");
    CHECK(ECG_PROTO_VERSION == 2, "proto version");
    PASS();
}

static void test_ecgr_v1_parse(void) {
    TEST("ECGR v1 parse (0/1 bitmap)");
    std::vector<uint8_t> b = build_v1();
    CHECK(ecgrHeaderValidate(b.data(), 250), "v1 header rejected");
    CHECK(ecgrHeaderVersion(b.data()) == 1, "v1 version");
    CHECK(!ecgrHeaderBitmapIsAsrc(b.data()), "v1 must not be asrc");
    CHECK(ecgrHeaderTotalSamples(b.data()) == 7, "v1 totalSamples");
    CHECK(ecgrHeaderAbnormalSec(b.data()) == 2, "v1 abnormalSec");
    PASS();
}

static void test_ecgr_v2_parse(void) {
    TEST("ECGR v2 parse (asrc bitmap)");
    std::vector<uint8_t> b = build_v2();
    CHECK(ecgrHeaderValidate(b.data(), 250), "v2 header rejected");
    CHECK(ecgrHeaderVersion(b.data()) == 2, "v2 version");
    CHECK(ecgrHeaderBitmapIsAsrc(b.data()), "v2 must be asrc");
    CHECK(ecgrHeaderAbnormalSec(b.data()) == 3, "v2 abnormalSec");
    PASS();
}

static void test_ecgr_v2_reserved0(void) {
    TEST("ECGR v2 header init sets reserved0 asrc flag");
    uint8_t hdr[ECGR_HEADER_SIZE];
    ecgrHeaderInit(hdr, 250, 100, 0, 0, 0, 0);
    CHECK(hdr[ECGR_OFF_RESERVED] & ECGR_RESERVED0_ABNORMAL_IS_ASRC,
          "reserved0 must mark asrc for v2");
    PASS();
}

static void test_ecgr_v1_still_buildable(void) {
    TEST("ECGR v1 header semantic (reserved0 = 0)");
    uint8_t hdr[ECGR_HEADER_SIZE];
    /* v1 header: 手动构造, 验证 validate 接受且不当作 asrc */
    memset(hdr, 0, sizeof(hdr));
    hdr[0]='E'; hdr[1]='C'; hdr[2]='G'; hdr[3]='R';
    hdr[ECGR_OFF_VERSION] = ECGR_VERSION_1;
    _ecgr_write32le(&hdr[ECGR_OFF_SAMPLE_RATE], 250);
    CHECK(ecgrHeaderValidate(hdr, 250), "v1 header rejected");
    CHECK(!ecgrHeaderBitmapIsAsrc(hdr), "v1 reserved0 must mean 0/1 bitmap");
    PASS();
}

int main(void) {
    printf("=== Protocol Golden Test Suite (C++) ===\n\n");
    test_contract_asrc_bits();
    test_ecgr_v1_parse();
    test_ecgr_v2_parse();
    test_ecgr_v2_reserved0();
    test_ecgr_v1_still_buildable();
    printf("\nPASS: %d  FAIL: %d\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}