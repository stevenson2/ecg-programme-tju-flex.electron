#ifndef ADS1292R_H
#define ADS1292R_H

/**
 * @file ads1292r.h
 * @brief ADS1292R 双通道 24 位模拟前端 + 呼吸阻抗测量驱动
 *
 * 硬件接口 (ESP32-S3, 按用户提供引脚):
 *   ADS_START  -> IO8
 *   ADS_RST    -> IO9   (PWDN/RESET#，低有效)
 *   ADS_DRDY   -> IO14
 *   SPI_MISO   -> IO13
 *   SPI_SCLK   -> IO12
 *   SPI_MOSI   -> IO11
 *   SPI_CS     -> IO10
 *
 * 数据通道说明 (ADS1292R 呼吸模式):
 *   Channel 1 = 呼吸阻抗解调信号 (RESP)
 *   Channel 2 = ECG 信号
 * 这是由于 ADS1292R 开启呼吸调制/解调后，Channel 1 不再采集 ECG。
 *
 * SPI 模式: CPOL=0, CPHA=1 (SPI Mode 1)
 * 输出数据: 24-bit 状态 + 24-bit CH1 + 24-bit CH2 = 72 bit (9 字节)
 */

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ======================== 数据结果 ======================== */

/**
 * @brief 一次 ADS1292R 采样结果
 */
typedef struct {
    float   ecgVolts;    /**< 通道2 ECG 电压 (V, 输入折算) */
    float   respVolts;   /**< 通道1 呼吸阻抗解调电压 (V) */
    int32_t ecgRaw;      /**< 通道2 24位原始码 */
    int32_t respRaw;     /**< 通道1 24位原始码 */
    uint8_t leadOffMask; /**< 导联脱落状态位: bit0=IN1P, bit1=IN1N, bit2=IN2P, bit3=IN2N, bit4=RLD */
    bool    valid;       /**< 本次读取是否有效 */
} ADS1292R_Data;

/* ======================== 引脚配置 ======================== */
#define ADS1292R_START_PIN   GPIO_NUM_8
#define ADS1292R_RESET_PIN   GPIO_NUM_9
#define ADS1292R_DRDY_PIN    GPIO_NUM_14
#define ADS1292R_SPI_MISO    GPIO_NUM_13
#define ADS1292R_SPI_SCLK    GPIO_NUM_12
#define ADS1292R_SPI_MOSI    GPIO_NUM_11
#define ADS1292R_SPI_CS      GPIO_NUM_10

#define ADS1292R_SPI_FREQ    1000000UL   /* 1 MHz; 兼顾寄存器读写 SCLK<2*fCLK 约束, 远高于 500SPS 所需 36kHz */
#define ADS1292R_ECG_GAIN    6.0f        /* CH2 PGA 增益 */
#define ADS1292R_RESP_GAIN   4.0f        /* CH1 呼吸 PGA 增益 (推荐 3/4) */
#define ADS1292R_VREF        2.42f       /* 内部基准, VREF_4V=0 */
#define ADS1292R_SAMPLE_RATE 500.0f      /* CONFIG1 DR=010 -> 500SPS */

/* ======================== 寄存器地址 ======================== */
#define ADS1292R_REG_ID        0x00
#define ADS1292R_REG_CONFIG1   0x01
#define ADS1292R_REG_CONFIG2   0x02
#define ADS1292R_REG_LOFF      0x03
#define ADS1292R_REG_CH1SET    0x04
#define ADS1292R_REG_CH2SET    0x05
#define ADS1292R_REG_RLD_SENS  0x06
#define ADS1292R_REG_LOFF_SENS 0x07
#define ADS1292R_REG_LOFF_STAT 0x08
#define ADS1292R_REG_RESP1     0x09
#define ADS1292R_REG_RESP2     0x0A
#define ADS1292R_REG_GPIO      0x0B

/* ======================== SPI 命令 ======================== */
#define ADS1292R_CMD_WAKEUP    0x02
#define ADS1292R_CMD_STANDBY   0x04
#define ADS1292R_CMD_RESET     0x06
#define ADS1292R_CMD_START     0x08
#define ADS1292R_CMD_STOP      0x0A
#define ADS1292R_CMD_OFFSETCAL 0x1A
#define ADS1292R_CMD_RDATAC    0x10
#define ADS1292R_CMD_SDATAC    0x11
#define ADS1292R_CMD_RDATA     0x12

/* ======================== API ======================== */

/**
 * @brief 初始化 ADS1292R (GPIO + SPI + 寄存器配置 + 启动转换)
 * @return true 初始化并识别成功
 */
bool ads1292rInit(void);

/**
 * @brief 判断是否已有新的转换数据 (DRDY 低电平)
 */
bool ads1292rIsDataReady(void);

/**
 * @brief 读取一帧数据 (仅在 DRDY 低时有效)
 * @param out 输出数据
 * @return true 读取成功
 */
bool ads1292rRead(ADS1292R_Data *out);

/**
 * @brief 读取芯片 ID 寄存器
 */
uint8_t ads1292rReadRegister(uint8_t reg);

/**
 * @brief 写寄存器
 */
void ads1292rWriteRegister(uint8_t reg, uint8_t val);

/**
 * @brief 发送 SPI 命令
 */
void ads1292rSendCommand(uint8_t cmd);

/**
 * @brief 软件复位 (发送 RESET 命令 + 重新配置)
 */
void ads1292rReset(void);

/**
 * @brief 停止转换
 */
void ads1292rStop(void);

/**
 * @brief 启动连续转换
 */
void ads1292rStart(void);

/**
 * @brief 最近一次导联脱落掩码
 */
uint8_t ads1292rGetLeadOffMask(void);

/**
 * @brief 获取芯片 ID (0 表示未识别)
 */
uint8_t ads1292rGetId(void);

/**
 * @brief 获取成功读取帧数
 */
uint32_t ads1292rGetReadOk(void);

/**
 * @brief 获取未就绪/读取失败次数
 */
uint32_t ads1292rGetReadFail(void);

#ifdef __cplusplus
}
#endif

#endif /* ADS1292R_H */
