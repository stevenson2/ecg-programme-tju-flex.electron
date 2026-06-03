/**
 * GPIO4 ADC 输入快速检测
 * 编译: pio run --target upload --upload-port COM4 -e adc_test
 * 或者在 platformio.ini 中替换 src/main.cpp
 */

#include <Arduino.h>

void setup() {
    Serial.begin(115200);
    delay(100);
    
    Serial.println("\n===== GPIO4 ADC 测试 =====");
    Serial.println("接好信号后观察电压变化");
    Serial.println("格式: raw_code,voltage_V");
    Serial.println("==========================\n");
    
    analogReadResolution(12);
    pinMode(GPIO_NUM_4, INPUT);
    analogSetPinAttenuation(GPIO_NUM_4, ADC_11db);
}

void loop() {
    uint32_t sum = 0;
    for (int i = 0; i < 8; i++) {
        sum += analogRead(GPIO_NUM_4);
    }
    uint16_t raw = sum / 8;
    float volt = raw * 3.3f / 4095.0f;
    
    Serial.print(raw);
    Serial.print(",");
    Serial.println(volt, 4);
    
    delay(10);  // 100Hz 采样
}
