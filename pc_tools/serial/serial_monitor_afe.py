"""真实 AFE 模式下的 esp_timer 验证: 发送 m,m 切到 AFE 并持续捕获 60s.

用法:
    python serial_monitor_afe.py [port] [seconds] [out_file]

与 serial_monitor.py 的区别: 在同一会话内先发送两次 'm' (SIM→REPLAY→AFE),
再持续读取。不要用 serial_cmd.py 分两次发 —— 串口关闭会复位设备导致模式丢失。
"""
import serial
import sys
import time

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
out_file = sys.argv[3] if len(sys.argv) > 3 else "esp_timer_check_afe.txt"

s = serial.Serial(port, 460800, timeout=0.2)
# 传播式 DTR/RTS 复位序列 (与 serial_cmd.py 一致)
s.rts = False
s.dtr = True
s.rts = False
time.sleep(0.2)
s.dtr = False
s.rts = False
time.sleep(3.0)  # 等待 boot 完成
s.reset_input_buffer()

# SIM → REPLAY → AFE
s.write(b"m\nm\n")
s.flush()

end = time.time() + dur
buf = b""
while time.time() < end:
    d = s.read(4096)
    if d:
        buf += d
s.close()

text = buf.decode("utf-8", "replace")
with open(out_file, "w", encoding="utf-8") as f:
    f.write(text)

keys = ("[SAMPLE]", "[BLE] conn params evt", "真实AFE", "模拟", "回放",
        "[心率]", "[REC]", "[BLE]", "GAP")
seen = set()
for line in text.splitlines():
    line = line.strip()
    if any(k in line for k in keys) and line not in seen:
        seen.add(line)
        print(line)
print("total bytes:", len(buf))
print("output:", out_file)
