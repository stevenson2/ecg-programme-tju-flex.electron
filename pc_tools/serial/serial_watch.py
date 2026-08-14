"""卡死观察: DTR 复位 → 读 90s, 每 3s 记录字节数 + 最后一行, 定位卡死时间点.
用法: python serial_watch.py [port]"""
import serial, sys, time

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"
total = 90

s = serial.Serial(port, 460800, timeout=0.2)
# DTR/RTS 复位
s.dtr = True; s.rts = False; time.sleep(0.15)
s.dtr = False; s.rts = True; time.sleep(0.15)
s.dtr = False; s.rts = False; time.sleep(0.1)

buf = b""
start = time.time()
last_report = 0
print("t(s)  bytes  last_line")
while time.time() - start < total:
    d = s.read(4096)
    if d:
        buf += d
    now = time.time() - start
    if now - last_report >= 3:
        lines = buf.decode("utf-8", "replace").splitlines()
        last = lines[-1] if lines else "(none)"
        print(f"{int(now):4d}  {len(buf):6d}  {last[:70]}")
        last_report = now
s.close()
print(f"总字节 {len(buf)}")
