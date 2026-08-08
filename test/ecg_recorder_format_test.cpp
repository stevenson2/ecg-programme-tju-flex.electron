/**
 * @file ecg_recorder_format_test.cpp
 * @brief ECGR 文件格式单元测试 (主机端, 纯 C++, 无 Arduino 依赖)
 *
 * 测试覆盖:
 *   - 头部初始化/验证 (魔数/版本)
 *   - 头部各字段 getter 往返一致性
 *   - ecgrHeaderUpdate 计数更新
 *   - ecgrFileSize / ecgrSamplesFromFileSize 数学一致性
 *   - ecgrIdxLine / ecgrIdxParse 索引行往返
 *   - 边界情况: 零样本, 大整数
 *
 * 编译运行:
 *   g++ -std=c++11 -Iinclude test/ecg_recorder_format_test.cpp -o /tmp/ecgr_test && /tmp/ecgr_test
 */

#include "storage/ecg_recorder_format.h"
#include <cstdio>
#include <cstring>
#include <cassert>

static int g_pass = 0;
static int g_fail = 0;

#define TEST(name)  printf("  TEST: %s ... ", name)
#define PASS()      do { printf("PASS\n"); g_pass++; } while(0)
#define FAIL(msg)   do { printf("FAIL: %s\n", msg); g_fail++; } while(0)
#define CHECK(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

/* ======================== 头部初始化与验证 ======================== */

static void test_header_init_and_validate(void) {
    TEST("header init + validate OK");
    uint8_t hdr[ECGR_HEADER_SIZE];
    memset(hdr, 0xFF, sizeof(hdr)); // 先填充垃圾值
    ecgrHeaderInit(hdr, 250, 1234567890, 0, 0, 0, 0);
    CHECK(ecgrHeaderValidate(hdr, 250), "valid header rejected");
    PASS();
}

static void test_header_magic_wrong(void) {
    TEST("header validate: wrong magic");
    uint8_t hdr[ECGR_HEADER_SIZE];
    ecgrHeaderInit(hdr, 250, 0, 0, 0, 0, 0);
    hdr[0] = 'X'; // 破坏魔数
    CHECK(!ecgrHeaderValidate(hdr, 250), "bad magic accepted");
    PASS();
}

static void test_header_version_wrong(void) {
    TEST("header validate: wrong version");
    uint8_t hdr[ECGR_HEADER_SIZE];
    ecgrHeaderInit(hdr, 250, 0, 0, 0, 0, 0);
    hdr[ECGR_OFF_VERSION] = 99; // 错误版本
    CHECK(!ecgrHeaderValidate(hdr, 250), "bad version accepted");
    PASS();
}

static void test_header_sample_rate_mismatch(void) {
    TEST("header validate: sample rate mismatch");
    uint8_t hdr[ECGR_HEADER_SIZE];
    ecgrHeaderInit(hdr, 250, 0, 0, 0, 0, 0);
    CHECK(!ecgrHeaderValidate(hdr, 500), "wrong sample rate accepted");
    PASS();
}

/* ======================== 头部 Getter 往返 ======================== */

static void test_header_getters_roundtrip(void) {
    TEST("header getters round-trip");
    uint8_t hdr[ECGR_HEADER_SIZE];
    ecgrHeaderInit(hdr, 250, 1600000000, 5000, 20, 3, 0);
    CHECK(ecgrHeaderTotalSamples(hdr) == 5000, "totalSamples mismatch");
    CHECK(ecgrHeaderDurationSec(hdr) == 20, "durationSec mismatch");
    CHECK(ecgrHeaderAbnormalSec(hdr) == 3, "abnormalSec mismatch");
    CHECK(ecgrHeaderStartUnix(hdr) == 1600000000, "startUnix mismatch");
    PASS();
}

static void test_header_getters_max_values(void) {
    TEST("header getters: max uint32 values");
    uint8_t hdr[ECGR_HEADER_SIZE];
    ecgrHeaderInit(hdr, 250, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0);
    CHECK(ecgrHeaderTotalSamples(hdr) == 0xFFFFFFFF, "totalSamples max mismatch");
    CHECK(ecgrHeaderDurationSec(hdr) == 0xFFFFFFFF, "durationSec max mismatch");
    CHECK(ecgrHeaderAbnormalSec(hdr) == 0xFFFFFFFF, "abnormalSec max mismatch");
    CHECK(ecgrHeaderStartUnix(hdr) == 0xFFFFFFFF, "startUnix max mismatch");
    PASS();
}

/* ======================== 头部更新 ======================== */

static void test_header_update(void) {
    TEST("header update changes counts");
    uint8_t hdr[ECGR_HEADER_SIZE];
    ecgrHeaderInit(hdr, 250, 100, 0, 0, 0, 0);
    CHECK(ecgrHeaderTotalSamples(hdr) == 0, "init totalSamples != 0");
    CHECK(ecgrHeaderDurationSec(hdr) == 0, "init duration != 0");

    ecgrHeaderUpdate(hdr, 2500, 10, 5);
    CHECK(ecgrHeaderTotalSamples(hdr) == 2500, "update totalSamples");
    CHECK(ecgrHeaderDurationSec(hdr) == 10, "update duration");
    CHECK(ecgrHeaderAbnormalSec(hdr) == 5, "update abnormalSec");

    // 验证 other fields unchanged
    CHECK(ecgrHeaderStartUnix(hdr) == 100, "startUnix changed by update");
    CHECK(ecgrHeaderValidate(hdr, 250), "update broke validation");
    PASS();
}

static void test_header_update_sets_bitmap_flag(void) {
    TEST("header update: abnormalSec > 0 sets bitmap flag");
    uint8_t hdr[ECGR_HEADER_SIZE];
    ecgrHeaderInit(hdr, 250, 100, 0, 0, 0, 0);
    CHECK((hdr[ECGR_OFF_FLAGS] & ECGR_FLAG_HAS_ABNORMAL_BITMAP) == 0,
          "bitmap flag should be clear");

    ecgrHeaderUpdate(hdr, 2500, 10, 3);
    CHECK((hdr[ECGR_OFF_FLAGS] & ECGR_FLAG_HAS_ABNORMAL_BITMAP) != 0,
          "bitmap flag should be set");
    PASS();
}

static void test_header_update_zero_abnormal(void) {
    TEST("header update: abnormalSec=0 clears bitmap flag");
    uint8_t hdr[ECGR_HEADER_SIZE];
    ecgrHeaderInit(hdr, 250, 100, 0, 0, 0, 0);
    ecgrHeaderUpdate(hdr, 2500, 10, 0);
    CHECK((hdr[ECGR_OFF_FLAGS] & ECGR_FLAG_HAS_ABNORMAL_BITMAP) == 0,
          "bitmap flag should be clear for zero abnormal");
    PASS();
}

/* ======================== 文件大小计算 ======================== */

static void test_file_size_no_bitmap(void) {
    TEST("file size: no bitmap");
    // 32 header + 100*2 samples + 0 bitmap = 32 + 200 = 232
    uint32_t sz = ecgrFileSize(100, 30, false);
    CHECK(sz == 32 + 200, "file size wrong (no bitmap)");
    PASS();
}

static void test_file_size_with_bitmap(void) {
    TEST("file size: with bitmap");
    // 32 header + 100*2 samples + 30 bitmap bytes = 32 + 200 + 30 = 262
    uint32_t sz = ecgrFileSize(100, 30, true);
    CHECK(sz == 32 + 200 + 30, "file size wrong (with bitmap)");
    PASS();
}

static void test_file_size_zero(void) {
    TEST("file size: zero samples, zero duration");
    uint32_t sz = ecgrFileSize(0, 0, false);
    CHECK(sz == 32, "file size for empty should be 32");
    PASS();
}

static void test_samples_from_file_size_roundtrip(void) {
    TEST("samples from file size: round-trip");
    uint32_t origSamples = 12345;
    uint32_t dur = 50;
    uint32_t sz = ecgrFileSize(origSamples, dur, true);
    uint32_t recovered = ecgrSamplesFromFileSize(sz, dur, true);
    CHECK(recovered == origSamples, "samples round-trip mismatch");
    PASS();
}

static void test_samples_from_file_size_no_bitmap(void) {
    TEST("samples from file size: no bitmap");
    uint32_t sz = ecgrFileSize(500, 10, false);
    uint32_t recovered = ecgrSamplesFromFileSize(sz, 10, false);
    CHECK(recovered == 500, "samples inverse mismatch (no bitmap)");
    PASS();
}

static void test_file_size_inverse_consistency(void) {
    TEST("file size: forward+inverse consistency for multiple values");
    uint32_t testCases[][2] = {
        {0, 0}, {1, 1}, {100, 1}, {250, 1}, {1000, 4}, {10000, 40},
        {25000, 100}, {100000, 400}, {250000, 1000}
    };
    for (size_t i = 0; i < sizeof(testCases) / sizeof(testCases[0]); i++) {
        uint32_t samples = testCases[i][0];
        uint32_t dur    = testCases[i][1];
        for (int bmp = 0; bmp <= 1; bmp++) {
            uint32_t sz = ecgrFileSize(samples, dur, (bool)bmp);
            uint32_t back = ecgrSamplesFromFileSize(sz, dur, (bool)bmp);
            if (back != samples) {
                printf("FAIL: fwd/inv mismatch: samples=%u dur=%u bmp=%d "
                       "sz=%u back=%u\n",
                       (unsigned)samples, (unsigned)dur, bmp,
                       (unsigned)sz, (unsigned)back);
                g_fail++; return;
            }
        }
    }
    PASS();
}

/* ======================== 索引行构建与解析 ======================== */

static void test_idx_line_build_parse_roundtrip(void) {
    TEST("idx line: build + parse round-trip");
    char buf[128];
    int len = ecgrIdxLine(buf, sizeof(buf), 1600000000, 30, 7500, 5, 15232);
    CHECK(len > 0, "idx line build failed");

    uint32_t st, dur, samples, abn, sz;
    CHECK(ecgrIdxParse(buf, &st, &dur, &samples, &abn, &sz),
          "idx line parse failed");
    CHECK(st == 1600000000, "startUnix mismatch");
    CHECK(dur == 30, "dur mismatch");
    CHECK(samples == 7500, "samples mismatch");
    CHECK(abn == 5, "abnormalSec mismatch");
    CHECK(sz == 15232, "sizeBytes mismatch");
    PASS();
}

static void test_idx_line_build_parse_zero_values(void) {
    TEST("idx line: zero values round-trip");
    char buf[128];
    int len = ecgrIdxLine(buf, sizeof(buf), 0, 0, 0, 0, 32);
    CHECK(len > 0, "idx line build failed");

    uint32_t st, dur, samples, abn, sz;
    CHECK(ecgrIdxParse(buf, &st, &dur, &samples, &abn, &sz),
          "idx line parse failed");
    CHECK(st == 0 && dur == 0 && samples == 0 && abn == 0 && sz == 32,
          "zero values parsed incorrectly");
    PASS();
}

static void test_idx_line_max_values(void) {
    TEST("idx line: max uint32 values round-trip");
    char buf[256];
    int len = ecgrIdxLine(buf, sizeof(buf),
                          0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                          0xFFFFFFFF, 0xFFFFFFFF);
    CHECK(len > 0, "idx line build failed");

    uint32_t st, dur, samples, abn, sz;
    CHECK(ecgrIdxParse(buf, &st, &dur, &samples, &abn, &sz),
          "idx line max parse failed");
    CHECK(st == 0xFFFFFFFF && dur == 0xFFFFFFFF
          && samples == 0xFFFFFFFF && abn == 0xFFFFFFFF && sz == 0xFFFFFFFF,
          "max values mismatch");
    PASS();
}

static void test_idx_parse_malformed(void) {
    TEST("idx parse: malformed line returns false");

    // 缺少字段
    CHECK(!ecgrIdxParse("123,456,789", nullptr, nullptr, nullptr, nullptr, nullptr),
          "short line should fail");
    // 空行
    CHECK(!ecgrIdxParse("", nullptr, nullptr, nullptr, nullptr, nullptr),
          "empty line should fail");
    // 非数字
    CHECK(!ecgrIdxParse("abc,def,ghi,jkl,mno", nullptr, nullptr, nullptr, nullptr, nullptr),
          "non-numeric should fail");
    // 正确格式但多余字段
    CHECK(!ecgrIdxParse("1,2,3,4,5,6", nullptr, nullptr, nullptr, nullptr, nullptr),
          "extra fields should fail");

    PASS();
}

static void test_idx_line_buffer_too_small(void) {
    TEST("idx line: buffer too small returns -1");
    char tiny[8];
    int ret = ecgrIdxLine(tiny, sizeof(tiny), 1234567890, 999, 250000, 10, 500000);
    CHECK(ret == -1, "should return -1 for tiny buffer");
    PASS();
}

static void test_idx_parse_trailing_newline(void) {
    TEST("idx parse: handles trailing \\n");
    uint32_t st, dur, samples, abn, sz;
    char line[128];
    ecgrIdxLine(line, sizeof(line), 100, 20, 5000, 2, 10064);
    // Append newline (simulate file read)
    strcat(line, "\n");
    CHECK(ecgrIdxParse(line, &st, &dur, &samples, &abn, &sz),
          "parse with \\n failed");
    CHECK(st == 100 && dur == 20 && samples == 5000 && abn == 2 && sz == 10064,
          "values with \\n mismatch");
    PASS();
}

/* ======================== 边界情况 ======================== */

static void test_header_zero_samples(void) {
    TEST("header: zero samples (crash-safe start state)");
    uint8_t hdr[ECGR_HEADER_SIZE];
    ecgrHeaderInit(hdr, 250, 999, 0, 0, 0, 0);
    CHECK(ecgrHeaderValidate(hdr, 250), "zero-sample header rejected");
    CHECK(ecgrHeaderTotalSamples(hdr) == 0, "totalSamples != 0");
    CHECK(ecgrHeaderDurationSec(hdr) == 0, "duration != 0");
    PASS();
}

static void test_header_huge_duration(void) {
    TEST("header: huge duration with bitmap");
    uint8_t hdr[ECGR_HEADER_SIZE];
    ecgrHeaderInit(hdr, 250, 100, 25000, 1000000, 500000, ECGR_FLAG_HAS_ABNORMAL_BITMAP);
    CHECK(ecgrHeaderDurationSec(hdr) == 1000000, "huge duration mismatch");
    CHECK(ecgrHeaderAbnormalSec(hdr) == 500000, "huge abnormal mismatch");
    PASS();
}

static void test_header_reserved_zeros(void) {
    TEST("header: reserved bytes are zero");
    uint8_t hdr[ECGR_HEADER_SIZE];
    ecgrHeaderInit(hdr, 250, 100, 0, 0, 0, 0);
    for (int i = ECGR_OFF_RESERVED; i < ECGR_HEADER_SIZE; i++) {
        if (hdr[i] != 0) {
            FAIL("reserved byte not zero");
            return;
        }
    }
    PASS();
}

static void test_fields_at_correct_offsets(void) {
    TEST("header: field byte offsets correct");
    uint8_t hdr[ECGR_HEADER_SIZE];
    ecgrHeaderInit(hdr, 250, 0x12345678, 0x9ABCDEF0, 0x0A0B0C0D, 0x0E0F1011, 0);

    // 魔数 "ECGR" = 0x45 0x43 0x47 0x52
    CHECK(hdr[0] == 0x45 && hdr[1] == 0x43 && hdr[2] == 0x47 && hdr[3] == 0x52,
          "magic mismatch");
    CHECK(hdr[4] == ECGR_VERSION, "version mismatch");

    // startUnix @ offset 10: 0x12345678 → LE: 78 56 34 12
    CHECK(_ecgr_read32le(&hdr[ECGR_OFF_START_UNIX]) == 0x12345678,
          "startUnix offset/readback wrong");

    // totalSamples @ offset 18: 0x9ABCDEF0 → LE: F0 DE BC 9A
    CHECK(_ecgr_read32le(&hdr[ECGR_OFF_TOTAL_SAMPLES]) == 0x9ABCDEF0,
          "totalSamples offset/readback wrong");

    PASS();
}

/* ======================== 主程序 ======================== */

int main(void) {
    printf("=== ECGR Format Test Suite ===\n\n");

    printf("[1] Header Init & Validate\n");
    test_header_init_and_validate();
    test_header_magic_wrong();
    test_header_version_wrong();
    test_header_sample_rate_mismatch();

    printf("\n[2] Header Getters\n");
    test_header_getters_roundtrip();
    test_header_getters_max_values();

    printf("\n[3] Header Update\n");
    test_header_update();
    test_header_update_sets_bitmap_flag();
    test_header_update_zero_abnormal();

    printf("\n[4] File Size Math\n");
    test_file_size_no_bitmap();
    test_file_size_with_bitmap();
    test_file_size_zero();
    test_samples_from_file_size_roundtrip();
    test_samples_from_file_size_no_bitmap();
    test_file_size_inverse_consistency();

    printf("\n[5] Index Line Build & Parse\n");
    test_idx_line_build_parse_roundtrip();
    test_idx_line_build_parse_zero_values();
    test_idx_line_max_values();
    test_idx_parse_malformed();
    test_idx_line_buffer_too_small();
    test_idx_parse_trailing_newline();

    printf("\n[6] Edge Cases\n");
    test_header_zero_samples();
    test_header_huge_duration();
    test_header_reserved_zeros();
    test_fields_at_correct_offsets();

    printf("\n=============================\n");
    printf("  PASS: %d  FAIL: %d\n", g_pass, g_fail);
    printf("=============================\n");

    return (g_fail > 0) ? 1 : 0;
}
