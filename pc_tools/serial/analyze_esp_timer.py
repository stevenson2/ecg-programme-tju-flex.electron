"""分析 serial_monitor.py 捕获的 esp_timer 验证日志.

用法:
    python analyze_esp_timer.py <capture.txt> [capture_duration_seconds]

输出:
    - CSV 数据行数 -> 推算串口输出率与主循环采样率
    - [SAMPLE] tick backlog 行 (应为 0/偶发)
    - [BLE] conn params 行
    - 心率状态行数
"""
import sys
import re

path = sys.argv[1] if len(sys.argv) > 1 else "esp_timer_check.txt"
duration = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0

csv_re = re.compile(r"^-?\d+\.\d{4},-?\d+\.\d{4},-?\d+\.\d{4},")
sample_re = re.compile(r"\[SAMPLE\]")
ble_conn_re = re.compile(r"\[BLE\] conn params evt")
hr_re = re.compile(r"\[心率\]")

csv_count = 0
sample_lines = []
ble_lines = []
hr_count = 0
line_count = 0

with open(path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line_count += 1
        line = line.strip()
        if csv_re.match(line):
            csv_count += 1
        elif sample_re.search(line):
            sample_lines.append(line)
        elif ble_conn_re.search(line):
            ble_lines.append(line)
        elif hr_re.search(line):
            hr_count += 1

serial_hz = csv_count / duration if duration > 0 else 0.0
sample_hz = serial_hz * 5.0  # 固件每 5 帧输出一行 CSV

print(f"总行数: {line_count}")
print(f"CSV 数据行数: {csv_count}")
print(f"串口输出率: {serial_hz:.2f} Hz (期望 100 Hz)")
print(f"推算主循环采样率: {sample_hz:.2f} Hz (期望 500 Hz)")
print(f"[心率] 状态行数: {hr_count}")
print(f"[SAMPLE] tick backlog 行数: {len(sample_lines)}")
for line in sample_lines[:50]:
    print("  " + line)
print(f"[BLE] conn params 行数: {len(ble_lines)}")
for line in ble_lines[:20]:
    print("  " + line)

if abs(sample_hz - 500.0) < 5.0 and not sample_lines:
    print("结论: 采样率正常, 无 tick 积压丢弃.")
elif sample_hz < 450.0:
    print("警告: 主循环可能未达到 500Hz, 需继续排查阻塞点.")
