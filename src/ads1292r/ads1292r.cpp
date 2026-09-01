#include <Arduino.h>
#include <SPI.h>
#include "ads1292r/ads1292r.h"

/**
 * @file ads1292r.cpp
 * @brief ADS1292R 驱动实现
 *
 * 本驱动面向 ESP32-S3 + Arduino SPI。
 * 默认使用芯片内部时钟 (CLKSEL 引脚硬件接高/内部振荡器) 与内部 2.42V 基准。
 *
 * 寄存器配置要点:
 *   - CONFIG1 = 0x02 : 500 SPS, 连续转换
 *   - CONFIG2 = 0xE0 : 使能内部基准缓冲器 + DC 导联脱落比较器, 2.42V 基准
 *   - CH1SET  = 0x40 : CH1 增益 4, 正常电极输入 (呼吸通道)
 *   - CH2SET  = 0x00 : CH2 增益 6, 正常电极输入 (ECG 通道)
 *   - RLD_SENS= 0x2C : 使能 RLD, 取 CH2 差模作为右腿驱动反馈
 *   - RESP1   = 0xEA : 内部 32kHz 呼吸调制/解调, 112.5° 相位
 *   - RESP2   = 0x02 : 32kHz, 内部 RLDREF
 */

/* ======================== 静态状态 ======================== */
static bool    s_initialized = false;
static uint8_t s_leadOffMask = 0;
static uint8_t s_lastId = 0;
static uint32_t s_readOk = 0;
static uint32_t s_readFail = 0;

/* ======================== 24 位有符号读取 ======================== */
static int32_t readInt24(const uint8_t *b)
{
    int32_t v = ((int32_t)b[0] << 16) | ((int32_t)b[1] << 8) | (int32_t)b[2];
    if (v & 0x800000) {
        v |= ~0xFFFFFF;   /* 符号扩展 */
    }
    return v;
}

/* ======================== 底层 SPI 事务 ======================== */
static void spiBegin(void)
{
    SPI.beginTransaction(SPISettings(ADS1292R_SPI_FREQ, MSBFIRST, SPI_MODE1));
    digitalWrite(ADS1292R_SPI_CS, LOW);
}

static void spiEnd(void)
{
    digitalWrite(ADS1292R_SPI_CS, HIGH);
    SPI.endTransaction();
}

/* ======================== 命令/寄存器 ======================== */

void ads1292rSendCommand(uint8_t cmd)
{
    if (!s_initialized) return;
    spiBegin();
    SPI.transfer(cmd);
    spiEnd();
    delayMicroseconds(5);
}

uint8_t ads1292rReadRegister(uint8_t reg)
{
    if (!s_initialized) return 0;
    uint8_t val = 0;
    spiBegin();
    SPI.transfer(0x20 | (reg & 0x1F));  /* RREG */
    SPI.transfer(0x00);                  /* 读 1 个寄存器 */
    val = SPI.transfer(0x00);
    spiEnd();
    delayMicroseconds(5);
    return val;
}

void ads1292rWriteRegister(uint8_t reg, uint8_t val)
{
    if (!s_initialized) return;
    spiBegin();
    SPI.transfer(0x40 | (reg & 0x1F));  /* WREG */
    SPI.transfer(0x00);                  /* 写 1 个寄存器 */
    SPI.transfer(val);
    spiEnd();
    delayMicroseconds(5);
}

/* ======================== 初始化 ======================== */

static void configureRegisters(void)
{
    /* 必须处于 SDATAC 模式才能写寄存器 (上电默认 RDATAC) */
    ads1292rSendCommand(ADS1292R_CMD_SDATAC);

    /* 内部基准 + 2.42V + 使能 DC 导联脱落比较器 */
    ads1292rWriteRegister(ADS1292R_REG_CONFIG2, 0xE0);
    /* 导联脱落阈值: 95%/5%, DC 模式 (默认 0x10, 显式写出便于调试) */
    ads1292rWriteRegister(ADS1292R_REG_LOFF, 0x10);
    /* 500 SPS 连续转换 */
    ads1292rWriteRegister(ADS1292R_REG_CONFIG1, 0x02);
    /* CH1: 呼吸通道, PGA=4, 正常电极输入 */
    ads1292rWriteRegister(ADS1292R_REG_CH1SET, 0x40);
    /* CH2: ECG 通道, PGA=6, 正常电极输入 */
    ads1292rWriteRegister(ADS1292R_REG_CH2SET, 0x00);
    /* RLD: 使能缓冲, 取 CH2P/CH2N 作为反馈源 */
    ads1292rWriteRegister(ADS1292R_REG_RLD_SENS, 0x2C);
    /* 呼吸: 调制+解调开启, 32kHz 内部时钟, 112.5° 相位 */
    ads1292rWriteRegister(ADS1292R_REG_RESP1, 0xEA);
    /* 呼吸: 32kHz, RLDREF 内部生成 */
    ads1292rWriteRegister(ADS1292R_REG_RESP2, 0x02);

    /* 使能四个电极的 DC 导联脱落检测; 状态字中的 LOFF_STAT 会随数据帧输出 */
    ads1292rWriteRegister(ADS1292R_REG_LOFF_SENS, 0x0F);
}

bool ads1292rInit(void)
{
    /* GPIO 初始化 */
    pinMode(ADS1292R_SPI_CS, OUTPUT);
    digitalWrite(ADS1292R_SPI_CS, HIGH);

    pinMode(ADS1292R_START_PIN, OUTPUT);
    digitalWrite(ADS1292R_START_PIN, LOW);

    pinMode(ADS1292R_RESET_PIN, OUTPUT);
    digitalWrite(ADS1292R_RESET_PIN, HIGH);   /* 先释放复位 */

    pinMode(ADS1292R_DRDY_PIN, INPUT);

    /* SPI: SCLK, MISO, MOSI, CS */
    SPI.begin(ADS1292R_SPI_SCLK, ADS1292R_SPI_MISO, ADS1292R_SPI_MOSI, ADS1292R_SPI_CS);
    SPI.setFrequency(ADS1292R_SPI_FREQ);

    s_initialized = true;

    /* 硬件复位脉冲 (PWDN/RESET 低有效) */
    digitalWrite(ADS1292R_RESET_PIN, LOW);
    delayMicroseconds(20);
    digitalWrite(ADS1292R_RESET_PIN, HIGH);
    delay(10);   /* 等待内部振荡器/上电稳定 */

    /* 软件复位命令 (双保险) */
    ads1292rSendCommand(ADS1292R_CMD_RESET);
    delay(10);

    /* 上电/复位后默认处于 RDATAC 模式, 必须 SDATAC 后才能读写寄存器 */
    ads1292rSendCommand(ADS1292R_CMD_SDATAC);
    delay(5);

    /* 读 ID 验证芯片在位 */
    s_lastId = ads1292rReadRegister(ADS1292R_REG_ID);
    if ((s_lastId & 0x0F) != 0x0B && (s_lastId & 0x0F) != 0x03) {
        /* ADS1292R ID: 高3位 011, 低2位 11 => 常见 0x7B 或 0x7F? 这里宽松检查 */
        Serial.printf("[ADS1292R] 警告: ID=0x%02X 与预期 ADS1292R 不一致, 继续尝试配置\n",
                      s_lastId);
        Serial0.printf("[ADS1292R] 警告: ID=0x%02X 与预期 ADS1292R 不一致, 继续尝试配置\n",
                       s_lastId);
    } else {
        Serial.printf("[ADS1292R] ID=0x%02X (ADS1292R 识别成功)\n", s_lastId);
        Serial0.printf("[ADS1292R] ID=0x%02X (ADS1292R 识别成功)\n", s_lastId);
    }

    configureRegisters();

    /* 寄存器回读校验: 确认 SPI 写读正常 */
    uint8_t rdCfg1 = ads1292rReadRegister(ADS1292R_REG_CONFIG1);
    uint8_t rdCh2  = ads1292rReadRegister(ADS1292R_REG_CH2SET);
    uint8_t rdResp = ads1292rReadRegister(ADS1292R_REG_RESP1);
    Serial.printf("[ADS1292R] 回读 CONFIG1=0x%02X CH2SET=0x%02X RESP1=0x%02X\n",
                  rdCfg1, rdCh2, rdResp);
    Serial0.printf("[ADS1292R] 回读 CONFIG1=0x%02X CH2SET=0x%02X RESP1=0x%02X\n",
                   rdCfg1, rdCh2, rdResp);
    if (rdCfg1 == 0x00 && rdCh2 == 0x00 && rdResp == 0x00) {
        Serial.println("[ADS1292R] 回读全 0：请检查 SPI 接线、供电、CS/MISO/MOSI/SCLK");
        Serial0.println("[ADS1292R] 回读全 0：请检查 SPI 接线、供电、CS/MISO/MOSI/SCLK");
    }

    /* 进入连续读数据模式 */
    ads1292rSendCommand(ADS1292R_CMD_RDATAC);

    /* 启动转换 */
    digitalWrite(ADS1292R_START_PIN, HIGH);

    /* 等待首个 DRDY，验证 ADC/时钟是否真正工作 */
    unsigned long drdyWaitStart = millis();
    bool firstDrdy = false;
    while ((millis() - drdyWaitStart) < 100) {
        if (ads1292rIsDataReady()) {
            firstDrdy = true;
            break;
        }
        delay(1);
    }

    if (firstDrdy) {
        Serial.println("[ADS1292R] DRDY 已确认，ADC 转换正常");
        Serial0.println("[ADS1292R] DRDY 已确认，ADC 转换正常");
    } else {
        Serial.println("[ADS1292R] 警告: 100ms 内未见到 DRDY！");
        Serial0.println("[ADS1292R] 警告: 100ms 内未见到 DRDY！");
        Serial.println("[ADS1292R] 请检查: CLKSEL 是否为内部振荡器(高)/外部时钟是否接入、供电、SPI 接线、START/RESET");
        Serial0.println("[ADS1292R] 请检查: CLKSEL 是否为内部振荡器(高)/外部时钟是否接入、供电、SPI 接线、START/RESET");
    }

    Serial.println("[ADS1292R] 初始化完成: CH1=呼吸阻抗, CH2=ECG, 500SPS");
    Serial0.println("[ADS1292R] 初始化完成: CH1=呼吸阻抗, CH2=ECG, 500SPS");
    return true;
}

void ads1292rReset(void)
{
    if (!s_initialized) return;
    digitalWrite(ADS1292R_START_PIN, LOW);
    ads1292rSendCommand(ADS1292R_CMD_STOP);
    ads1292rSendCommand(ADS1292R_CMD_RESET);
    delay(10);
    configureRegisters();
    ads1292rSendCommand(ADS1292R_CMD_RDATAC);
    digitalWrite(ADS1292R_START_PIN, HIGH);
}

void ads1292rStop(void)
{
    if (!s_initialized) return;
    digitalWrite(ADS1292R_START_PIN, LOW);
    ads1292rSendCommand(ADS1292R_CMD_STOP);
}

void ads1292rStart(void)
{
    if (!s_initialized) return;
    ads1292rSendCommand(ADS1292R_CMD_START);
    digitalWrite(ADS1292R_START_PIN, HIGH);
}

bool ads1292rIsDataReady(void)
{
    return digitalRead(ADS1292R_DRDY_PIN) == LOW;
}

bool ads1292rRead(ADS1292R_Data *out)
{
    if (!s_initialized || out == NULL) return false;

    if (!ads1292rIsDataReady()) {
        s_readFail++;
        out->valid = false;
        return false;
    }

    s_readOk++;
    uint8_t buf[9];
    spiBegin();
    for (int i = 0; i < 9; i++) {
        buf[i] = SPI.transfer(0x00);
    }
    spiEnd();

    /* 状态字: 1100 + LOFF_STAT[4:0] + GPIO[1:0] + 13*0 */
    /* 实际 LOFF 位在 buf[0] bit4..bit0? 这里按 24 位状态中 LOFF_STAT 位置解析:
     * status[23:20]=1100, status[19:15]=LOFF_STAT[4:0] */
    uint32_t status = ((uint32_t)buf[0] << 16) | ((uint32_t)buf[1] << 8) | buf[2];
    uint8_t loffStat = (uint8_t)((status >> 15) & 0x1F);
    s_leadOffMask = loffStat & 0x1F;

    int32_t ch1 = readInt24(buf + 3);  /* CH1 = RESP */
    int32_t ch2 = readInt24(buf + 6);  /* CH2 = ECG */

    out->respRaw = ch1;
    out->ecgRaw  = ch2;
    out->respVolts = ((float)ch1 / 8388607.0f) * (ADS1292R_VREF / ADS1292R_RESP_GAIN);
    out->ecgVolts  = ((float)ch2 / 8388607.0f) * (ADS1292R_VREF / ADS1292R_ECG_GAIN);
    out->leadOffMask = s_leadOffMask;
    out->valid = true;
    return true;
}

uint8_t ads1292rGetLeadOffMask(void)
{
    return s_leadOffMask;
}

uint8_t ads1292rGetId(void)
{
    return s_lastId;
}

uint32_t ads1292rGetReadOk(void)
{
    return s_readOk;
}

uint32_t ads1292rGetReadFail(void)
{
    return s_readFail;
}
