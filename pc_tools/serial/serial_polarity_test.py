"""测试 USB-Serial-JTAG 复位极性: DTR=True/False 各试一次 RTS 脉冲, 观察 boot 模式.
"""
import serial, sys, time

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"

def attempt(dtr_val, tag):
    s = serial.Serial(port, 460800, timeout=0.2)
    s.dtr = dtr_val
    s.rts = True
    time.sleep(0.2)
    s.rts = False
    time.sleep(0.1)
    buf = b""
    end = time.time() + 5.0
    while time.time() < end:
        d = s.read(4096)
        if d:
            buf += d
    s.close()
    text = buf.decode("utf-8", "replace")
    print("=== attempt", tag, "(dtr=%s) bytes=%d ===" % (dtr_val, len(buf)))
    for l in text.splitlines():
        if any(k in l for k in ("boot:", "ESP-ROM", "ESP32-ECG", "[系统]", "waiting")):
            print("   ", l.strip())

attempt(True, "DTR=HIGH")
time.sleep(0.5)
attempt(False, "DTR=LOW")
