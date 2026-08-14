"""设备状态诊断: 打开串口 -> 读 boot/运行日志 (BLE连接/REC命令/WiFi自动启动) -> REC_STATUS -> REC_LIST
用法: python serial_diag.py <port> <observe_seconds>"""
import serial, sys, time

port = sys.argv[1]
observe = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0

s = serial.Serial(port, 460800, timeout=0.2)
time.sleep(1.5)
s.reset_input_buffer()
print("[DIAG] port open (设备可能已复位重启), 观察 %.0fs ..." % observe)

def read_until(deadline):
    buf = b""
    while time.time() < deadline:
        d = s.read(4096)
        if d:
            buf += d
    return buf.decode("utf-8", "replace")

# 观察阶段: 过滤关键日志
end = time.time() + observe
text = read_until(end)
for line in text.splitlines():
    if any(k in line for k in ("WiFi", "AP diag", "AP started", "REC", "ECGR", "BLE", "AI init", "系统")):
        print(line)

def send(cmd, read_sec):
    s.reset_input_buffer()
    s.write((cmd + "\n").encode())
    s.flush()
    end = time.time() + read_sec
    text = read_until(end)
    for line in text.splitlines():
        if any(k in line for k in ("REC", "ECGR", "WiFi", "WIFI", "DIAG", "ok", "fail")):
            print(line)

print("[DIAG] REC_STATUS")
send("REC_STATUS", 3)
print("[DIAG] REC_LIST")
send("REC_LIST", 4)
s.close()
print("[DIAG] done")
