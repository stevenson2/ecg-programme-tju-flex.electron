"""监控 2 分钟: boot 后持续读串口, 捕获固件侧 [ECGR]/[REC]/BLE 事件 (App 的 BLE 命令会触发固件打印).
用法: python serial_monitor.py [port] [seconds] [out_file]
"""
import serial, sys, time

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
out_file = sys.argv[3] if len(sys.argv) > 3 else "serial_monitor_out.txt"

s = serial.Serial(port, 460800, timeout=0.2)
s.rts = False
s.dtr = True
s.rts = False
time.sleep(0.2)
s.dtr = False
s.rts = False

end = time.time() + dur
buf = b""
while time.time() < end:
    d = s.read(4096)
    if d:
        buf += d
s.close()
text = buf.decode("utf-8", "replace")
open(out_file, "w", encoding="utf-8").write(text)
# 打印关键事件行
keys = ("[ECGR]", "[REC]", "BLE", "GAP", "connected", "disconnect", "REC_", "连接", "recording")
seen = set()
for l in text.splitlines():
    l = l.strip()
    if any(k in l for k in keys) and l not in seen:
        seen.add(l)
        print(l)
print("total bytes:", len(buf))
