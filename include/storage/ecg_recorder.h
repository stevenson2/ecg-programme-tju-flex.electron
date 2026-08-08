/**
 * @file ecg_recorder.h
 * @brief ECG 录制模块 — 250Hz int16 录制到 SPIFFS，崩溃安全
 *
 * 录制格式: .ecgr 文件 (32 字节头部 + int16 样本流 + 可选异常位图),
 * 详情见 ecg_recorder_format.h。
 *
 * 崩溃安全策略:
 *   - REC_START: 立即写入 totalSamples=0 的头部, 随后追加样本
 *   - REC_STOP:  刷缓冲, 回到文件头重写最终头部字段
 *   - 挂载扫描:  删除头部与文件大小不一致的损坏 .ecgr
 *
 * 线程模型: 单任务 (loop() 上下文), 无需锁。
 */

#ifndef ECG_RECORDER_H
#define ECG_RECORDER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ======================== 可调参数 (#ifndef 默认值) ======================== */

/** @brief 录制采样率 (Hz) — 调用者已完成 500→250Hz 2:1 抽取 */
#ifndef ECG_REC_SAMPLE_RATE
#define ECG_REC_SAMPLE_RATE  250
#endif

/** @brief 最多保留记录数 */
#ifndef ECG_REC_KEEP_MAX
#define ECG_REC_KEEP_MAX     10
#endif

/** @brief 写缓冲批量刷新阈值 (字节) */
#ifndef ECG_REC_BATCH_BYTES
#define ECG_REC_BATCH_BYTES  8192
#endif

/** @brief SPIFFS 最低空闲字节阈值 — 不足时触发删除最旧记录 */
#ifndef ECG_REC_FREE_MIN_BYTES
#define ECG_REC_FREE_MIN_BYTES  (512 * 1024)
#endif

/** @brief 自动录制: 连续正常秒数达到此值后自动停止 */
#ifndef ECG_REC_AUTO_STOP_ABNORMAL_SECS
#define ECG_REC_AUTO_STOP_ABNORMAL_SECS  5
#endif

/* ======================== 公共 API ======================== */

/**
 * @brief 初始化录制模块 (挂载 SPIFFS + 扫描修复 + 重建索引)
 *
 * 使用 SPIFFS.begin(true, "/spiffs", 8, "ecgdata") 挂载 ecgdata 分区。
 * 挂载后扫描 *.ecgr 文件, 删除损坏文件, 重建 records.idx。
 *
 * @return true 初始化成功
 */
bool ecgRecorderInit(void);

/**
 * @brief 开始录制
 *
 * 创建新 .ecgr 文件 (路径: /ecgdata/ecg_rec_<unixTime>.ecgr),
 * 写入初始头部 (totalSamples=0) 并以追加模式打开。
 * 录制中再次调用本函数返回 false。
 *
 * @return true 成功开始录制, false 已在录制中或文件创建失败
 */
bool ecgRecorderStart(void);

/**
 * @brief 推送一个 250Hz int16 样本 (调用者已完成 500→250Hz 2:1 抽取)
 *
 * 样本写入 8KB 内部缓冲区, 满时批量刷入 SPIFFS。
 * 若未在录制状态则直接返回 (廉价 no-op)。
 *
 * @param sample 16 位有符号样本值
 */
void ecgRecorderPushSample(int16_t sample);

/**
 * @brief 标记当前秒是否有异常 (每秒调用一次, 1Hz)
 *
 * 秒边界写入异常位图字节 (若在录制中)。
 * 自动录制逻辑:
 *   - 自动录制启用且检测到异常上升沿 → 自动开始录制
 *   - 连续 ECG_REC_AUTO_STOP_ABNORMAL_SECS 秒正常 → 自动停止录制
 *
 * @param abnormal true 表示当前秒内检测到异常
 */
void ecgRecorderSetSecondAbnormal(bool abnormal);

/**
 * @brief 停止录制
 *
 * 刷写缓冲区, 回到文件头重写最终头部字段 (seek(0) + 重写),
 * 关闭文件, 追加索引行, 执行保留策略 (最多10条 + 空闲空间检查)。
 *
 * @return true 成功停止
 */
bool ecgRecorderStop(void);

/**
 * @brief 查询是否正在录制
 * @return true 录制中
 */
bool ecgRecorderIsRecording(void);

/**
 * @brief 列出所有录制记录 (渲染 records.idx 内容)
 *
 * 将索引文件内容复制到 outBuf 中 (以 \0 结尾的文本)。
 *
 * @param outBuf  输出缓冲区
 * @param bufLen  缓冲区大小
 * @return 实际写入字节数 (含 \0), 若缓冲区不足返回 -1
 */
int ecgRecorderList(char* outBuf, int bufLen);

/**
 * @brief 获取当前保留的记录数
 * @return 记录数
 */
uint32_t ecgRecorderRecordCount(void);

/**
 * @brief 获取当前录制起始 Unix 时间戳
 * @return 时间戳, 若未在录制中返回 0
 */
uint32_t ecgRecorderCurrentRecordStart(void);

/**
 * @brief 获取当前录制已持续的秒数
 * @return 秒数, 若未在录制中返回 0
 */
uint32_t ecgRecorderCurrentDurationSec(void);

/**
 * @brief 启用或禁用自动录制模式
 * @param enable true 启用自动录制
 */
void ecgRecorderSetAutoRecord(bool enable);

/**
 * @brief 查询自动录制是否启用
 * @return true 启用
 */
bool ecgRecorderAutoRecordEnabled(void);

/**
 * @brief 重置模块状态 (停止录制 + 清内部状态, 不删除文件)
 */
void ecgRecorderReset(void);

#ifdef __cplusplus
}
#endif

#endif /* ECG_RECORDER_H */
