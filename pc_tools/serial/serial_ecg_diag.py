"""真实ECG AI 误报诊断: 打开串口 -> boot -> 切真实AFE模式 -> 观察报警/置信度/幅度 -> AI统计
用法: python serial_ecg_diag.py <port> [observe_seconds]"""
import serial, sys, time, re

port = sys.argv[1]
observe = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

s = serial.Serial(port, 460800, timeout=0.2)
time.sleep(1.5)
s.reset_input_buffer()
time.sleep(6)  # 等待 boot

# 切到真实 AFE 模式 (模拟器 -> 回放 -> 真实AFE, 每次 'm' 切一次)
for cmd in ("m", "m"):
    s.write((cmd + "\n").encode())
    s.flush()
    time.sleep(1.0)
print("[DIAG] 已发送两次 'm' (目标: 真实AFE模式)")

# 观察 CSV 数据
end = time.time() + observe
buf = b""
while time.time() < end:
    d = s.read(4096)
    if d:
        buf += d
text = buf.decode("utf-8", "replace")
lines = text.splitlines()

# 报警行 (abnormal=1, 第8列)
alarm_lines = [l for l in lines if re.match(r'^-?[\d.]+,-?[\d.]+,-?[\d.]+,\d+,\d+,[\d.]+,\d+,1,', l)]
conf_vals = []
for l in alarm_lines:
    parts = l.split(",")
    if len(parts) >= 9:
        try:
            conf_vals.append(round(float(parts[8]), 3))
        except ValueError:
            pass
print(f"[DIAG] 观察 {observe:.0f}s: 总行数={len(lines)}, 报警行={len(alarm_lines)}")
if conf_vals:
    print(f"[DIAG] 报警置信度分布: {sorted(set(conf_vals))}")
else:
    print("[DIAG] 无报警行 (观察窗口内 abnormal 均为 0)")

# filtered 幅度统计 (第3列)
fvals = []
for l in lines:
    parts = l.split(",")
    if len(parts) >= 3:
        try:
            fvals.append(float(parts[2]))
        except ValueError:
            pass
if fvals:
    pp = max(fvals) - min(fvals)
    print(f"[DIAG] filtered 幅度: min={min(fvals):.4f} max={max(fvals):.4f} 峰峰值={pp:.4f}V (n={len(fvals)})")
    print(f"[DIAG] 20mV 阈值对照: {'峰峰值 < 20mV → flatline 会触发!' if pp < 0.02 else '峰峰值充足, flatline 不应触发'}")

# 关键日志
for l in lines:
    if any(k in l for k in ("[AI]", "[SAFETY]", "削顶", "当前输入", "切换到")):
        print("[LOG] " + l.strip())

# AI 统计
s.reset_input_buffer()
s.write(b"a\n")
s.flush()
time.sleep(2)
d = s.read(4096)
if d:
    t = d.decode("utf-8", "replace")
    for l in t.splitlines():
        if "[AI]" in l:
            print("[AI-STAT] " + l.strip())
s.close()
print("[DIAG] done")
