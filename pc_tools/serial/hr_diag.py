"""hr_diag.py — 心率双倍计数诊断: 切 AFE 后抓 90s [心率] 行 + CSV bpm 统计"""
import serial, sys, time, re

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"

s = serial.Serial(port, 460800, timeout=0.2)
s.rts = False
s.dtr = True
s.rts = False
time.sleep(0.2)
s.dtr = False
s.rts = False

def read_for(sec):
    end = time.time() + sec
    buf = b""
    while time.time() < end:
        d = s.read(4096)
        if d:
            buf += d
    return buf.decode("utf-8", "replace")

def send_read(cmd, sec):
    s.reset_input_buffer()
    s.write((cmd + "\n").encode())
    s.flush()
    return read_for(sec)

read_for(11.0)
# 状态感知切 AFE
for i in range(6):
    t = send_read("m", 2.5)
    if "真实AFE" in t:
        print("MODE: AFE (第%d次 m)" % (i + 1))
        break
# 观察 90s
obs = read_for(90.0)
hr_lines = [l.strip() for l in obs.splitlines() if "心率" in l or "运动" in l]
print("=== [心率]/[运动] 状态行 (最多 30 条) ===")
for l in hr_lines[-30:]:
    print(l)
# CSV bpm 统计
bpms = []
for l in obs.splitlines():
    l = l.strip()
    if re.match(r"^-?\d+\.\d+,", l):
        p = l.split(",")
        if len(p) >= 9:
            try:
                b = int(p[3])
                if b > 0:
                    bpms.append(b)
            except ValueError:
                pass
if bpms:
    print("=== CSV bpm 列: n=%d min=%d median=%d max=%d ===" % (
        len(bpms), min(bpms), sorted(bpms)[len(bpms)//2], max(bpms)))
s.close()
