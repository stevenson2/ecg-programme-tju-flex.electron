"""串口命令发送器 v3: 复用 boot_read 的传播式复位序列 (每次 dtr 赋值后跟 rts 赋值).
用法: python serial_cmd.py <port> <cmd> [read_seconds] [prewait_seconds]
"""
import serial, sys, time

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"
cmd = sys.argv[2] if len(sys.argv) > 2 else ""
dur = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
prewait = float(sys.argv[4]) if len(sys.argv) > 4 else 11.0

s = serial.Serial(port, 460800, timeout=0.2)
s.rts = False
s.dtr = True
s.rts = False
time.sleep(0.2)
s.dtr = False
s.rts = False
if prewait > 0:
    time.sleep(prewait)
    s.reset_input_buffer()
if cmd:
    s.write((cmd + "\n").encode())
    s.flush()
end = time.time() + dur
buf = b""
while time.time() < end:
    d = s.read(4096)
    if d:
        buf += d
s.close()
text = buf.decode("utf-8", "replace")
print(text if text else "(no output)")
print("bytes:", len(buf))
