"""真实ECG监测 v2: 确认boot完成 -> 切换真实AFE并验证 -> 观察 raw ADC/削顶/报警
用法: python serial_ecg_diag2.py <port> [observe_seconds]"""
import serial, sys, time, re

port = sys.argv[1]
observe = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

s = serial.Serial(port, 460800, timeout=0.2)
time.sleep(1.5)
s.reset_input_buffer()
time.sleep(8)  # boot (固件含 AP 自动启动, boot 变慢)

def send_and_read(cmd, sec):
    s.reset_input_buffer()
    s.write((cmd + "\n").encode())
    s.flush()
    end = time.time() + sec
    buf = b""
    while time.time() < end:
        d = s.read(4096)
        if d:
            buf += d
    return buf.decode("utf-8", "replace")

# 确认设备在线
t = send_and_read("DIAG", 2)
print("[DIAG] 设备回复:", "DIAG" if "DIAG" in t else "NO-REPLY")

# 切换到真实 AFE 模式并验证 (最多 4 次 'm')
for i in range(4):
    t = send_and_read("m", 2)
    if "真实AFE" in t:
        print(f"[DIAG] 已到达真实AFE模式 (第{i+1}次 m)")
        break
else:
    print("[DIAG] WARN: 未能切到真实AFE, 继续观察当前模式")

# 观察
end = time.time() + observe
buf = b""
while time.time() < end:
    d = s.read(4096)
    if d:
        buf += d
text = buf.decode("utf-8", "replace")
lines = text.splitlines()

# raw ADC 范围 (clean + 1.65)
raw_vals = []
fvals = []
alarm_lines = []
conf_vals = []
clip_count = 0
for l in lines:
    if l and l[0].isdigit() and "," in l:
        parts = l.split(",")
        if len(parts) >= 9:
            try:
                clean = float(parts[0]); f = float(parts[2])
                abn = int(parts[7]); conf = float(parts[8])
                raw_vals.append(clean + 1.65)
                fvals.append(f)
                if abn == 1:
                    alarm_lines.append(l)
                    conf_vals.append(round(conf, 3))
            except ValueError:
                pass
    elif "削顶" in l:
        clip_count += 1

if raw_vals:
    print(f"[DIAG] raw ADC: min={min(raw_vals):.3f} max={max(raw_vals):.3f} (0~3.3V 量程)")
    if max(raw_vals) - min(raw_vals) < 0.05:
        print("[DIAG] !! raw ADC 几乎恒定 -> 信号未接入/悬空!")
if fvals:
    pp = max(fvals) - min(fvals)
    print(f"[DIAG] filtered: min={min(fvals):.4f} max={max(fvals):.4f} 峰峰值={pp:.4f}V")
print(f"[DIAG] 观察 {observe:.0f}s: 行数={len(lines)}, 报警行={len(alarm_lines)}, 削顶警告={clip_count}")
if conf_vals:
    print(f"[DIAG] 报警置信度: {sorted(set(conf_vals))}")

# AI 统计
t = send_and_read("a", 3)
for l in t.splitlines():
    if "[AI]" in l:
        print("[AI-STAT] " + l.strip())
s.close()
print("[DIAG] done")
