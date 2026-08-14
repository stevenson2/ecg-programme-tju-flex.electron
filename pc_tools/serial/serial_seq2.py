"""录制链路完整观察: REC_START -> 25s -> REC_STOP -> REC_LIST
打印所有非 CSV 行 (含删除日志/索引内容), 验证 deleteOldestRecord 行为。
用法: python serial_seq2.py <port>"""
import serial, sys, time

port = sys.argv[1]
s = serial.Serial(port, 460800, timeout=0.2)
time.sleep(1.5)
s.reset_input_buffer()
time.sleep(5)

def send(cmd, read_sec):
    s.reset_input_buffer()
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
        # 排除 CSV 数据行 (以数字开头含逗号)
        if line and line[0].isdigit() and "," in line:
            continue
        if line.strip():
            print(line)

print("[SEQ2] REC_STATUS")
send("REC_STATUS", 3)
print("[SEQ2] REC_START")
send("REC_START", 3)
print("[SEQ2] 录制 25s ...")
time.sleep(25)
print("[SEQ2] REC_STOP")
send("REC_STOP", 4)
print("[SEQ2] REC_LIST (完整)")
send("REC_LIST", 4)
print("[SEQ2] REC_STATUS")
send("REC_STATUS", 3)
s.close()
print("[SEQ2] done")
