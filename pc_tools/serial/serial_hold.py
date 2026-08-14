"""串口保持打开读取器 v2: 打开端口 -> 可选发送命令 -> 持续读取 N 秒 (端口不关闭).
用于"串口保持打开"状态下执行命令 + netsh 扫描 (避免关闭串口触发设备复位).
用法: python serial_hold.py <port> <seconds> [cmd] [prefix_filter]"""
import serial, sys, time

port = sys.argv[1]
dur = float(sys.argv[2])
cmd  = sys.argv[3] if len(sys.argv) > 3 else ""
filt = sys.argv[4] if len(sys.argv) > 4 else ""

s = serial.Serial(port, 460800, timeout=0.2)
time.sleep(1.5)          # 打开后稳定
s.reset_input_buffer()
if cmd:
    s.write((cmd + "\n").encode())
    s.flush()
print("[HOLD] port open%s, reading %.0fs..." % (" cmd=" + cmd if cmd else "", dur))
end = time.time() + dur
while time.time() < end:
    d = s.read(4096)
    if d:
        text = d.decode("utf-8", "replace")
        if filt:
            for line in text.splitlines():
                if filt in line:
                    print(line)
        else:
            print(text, end="")
s.close()
print("[HOLD] closed")
