"""ESP32-S3 USB-Serial-JTAG DTR/RTS 复位 + 读 boot 输出.
用法: python serial_reset.py [port]"""
import serial, sys, time

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"

s = serial.Serial(port, 460800, timeout=0.2)
print("[1] DTR/RTS 复位序列 (EN 拉低再释放)...")
# ESP32-S3 USB-Serial-JTAG: DTR/RTS 组合控制复位/下载
s.dtr = True
s.rts = False
time.sleep(0.15)
s.dtr = False
s.rts = True
time.sleep(0.15)
s.dtr = False
s.rts = False
time.sleep(0.1)

print("[2] 读 10s boot 输出...")
end = time.time() + 10
buf = b""
while time.time() < end:
    d = s.read(4096)
    if d:
        buf += d
s.close()
print(f"bytes: {len(buf)}")
print(buf.decode("utf-8", "replace")[:2000] if buf else "(still no output)")
