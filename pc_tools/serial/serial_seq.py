"""串口命令序列: REC_START -> 等待 25s (录制) -> REC_STOP -> REC_LIST
保持串口打开全程, 避免打开/关闭触发设备复位中断录制。
用法: python serial_seq.py <port>"""
import serial, sys, time

port = sys.argv[1]
s = serial.Serial(port, 460800, timeout=0.2)
time.sleep(1.5)
s.reset_input_buffer()

def send(cmd, read_sec):
    s.write((cmd + "\n").encode())
    s.flush()
    end = time.time() + read_sec
    buf = b""
    while time.time() < end:
        d = s.read(4096)
        if d:
            buf += d
    text = buf.decode("utf-8", "replace")
    for line in text.splitlines():
        if any(k in line for k in ("REC", "ECGR", "WiFi", "WIFI", "DIAG")):
            print(line)

print("[SEQ] REC_START")
send("REC_START", 3)
print("[SEQ] 录制 25s ...")
time.sleep(25)
print("[SEQ] REC_STOP")
send("REC_STOP", 3)
print("[SEQ] REC_LIST")
send("REC_LIST", 4)
s.close()
print("[SEQ] done")
