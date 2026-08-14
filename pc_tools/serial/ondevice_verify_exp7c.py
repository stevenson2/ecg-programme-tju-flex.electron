"""ondevice_verify_exp7c.py — 真机 AI 置信度验证: boot -> 切 AFE -> 观察 -> 'a' 统计"""
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

boot = read_for(12.0)
# 切到真实 AFE (状态感知)
for i in range(6):
    t = send_read("m", 2.5)
    if "真实AFE" in t:
        print(f"MODE: AFE (第{i+1}次 m)")
        break
    if i == 5:
        print("MODE: 未能切到 AFE")
        s.close(); sys.exit(1)
# 观察 25s 收集 AI 置信度 (CSV 第9列)
obs = read_for(25.0)
confs = []
alarm_rows = 0
tot = 0
for l in obs.splitlines():
    l = l.strip()
    if re.match(r"^-?\d+\.\d+,", l):
        p = l.split(",")
        if len(p) >= 9:
            tot += 1
            try:
                c = float(p[8])
                if c > 0:
                    confs.append(c)
                if p[7] == "1":
                    alarm_rows += 1
            except ValueError:
                pass
if confs:
    print(f"CSV rows={tot} | nonzero-conf={len(confs)} | alarm_rows={alarm_rows}")
    print(f"conf mean={sum(confs)/len(confs):.4f} median={sorted(confs)[len(confs)//2]:.4f} "
          f"min={min(confs):.4f} max={max(confs):.4f}")
    print(f"frac>0.5={sum(1 for c in confs if c>0.5)/len(confs):.4f} "
          f"frac>0.8={sum(1 for c in confs if c>0.8)/len(confs):.4f}")
t = send_read("a", 3.0)
for l in t.splitlines():
    if "[AI]" in l:
        print(l.strip())
s.close()
