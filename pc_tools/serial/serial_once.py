"""一次性串口命令发送器 v2: 可选 boot 等待(prewait), 发送单条命令后读取 N 秒.
用法: python serial_once.py <port> <cmd> [read_seconds] [prewait_seconds]
USB-Serial-JTAG: 打开串口可能触发 DTR/RTS 复位, 故默认等待 6s 再发命令."""
import serial, sys, time

port = sys.argv[1]
cmd = sys.argv[2]
dur = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
prewait = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0

s = serial.Serial(port, 460800, timeout=0.2)
if prewait > 0:
    time.sleep(prewait)      # 等待 boot + USB 枚举
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
