"""切真实 AFE 模式 + 抓 AI 置信度: DTR复位 → m×2 切模式 → 确认模式 → l/a 命令.
用法: python serial_afe_ai.py [port]"""
import serial, sys, time

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"

s = serial.Serial(port, 460800, timeout=0.2)
# DTR 复位 (确保干净状态)
s.dtr = True; s.rts = False; time.sleep(0.15)
s.dtr = False; s.rts = True; time.sleep(0.15)
s.dtr = False; s.rts = False; time.sleep(0.1)
time.sleep(2.5)  # boot + USB 枚举
s.reset_input_buffer()

def read_sec(sec):
    buf = b""
    end = time.time() + sec
    while time.time() < end:
        d = s.read(4096)
        if d: buf += d
    return buf.decode("utf-8", "replace")

def send(cmd, wait=1.2):
    s.write((cmd + "\n").encode())
    s.flush()
    time.sleep(wait)

print("[1] 发 m×2 切到真实 AFE (模拟器→回放→真实AFE)...")
send("m"); send("m")

print("[2] 确认模式 + 抓真实 ECG 波形 (filtered 列)...")
txt = read_sec(4)
csv = [l for l in txt.splitlines() if l and l[0].isdigit() and "," in l]
if csv:
    p = csv[-1].split(",")
    mode = "真实AFE" if p[4] == "0" else f"模拟器/回放(true_bpm={p[4]})"
    print(f"  模式: {mode}, bpm={p[3]}, sqi={p[5]}, abnormal={p[7]}, conf={p[8]}")
    for l in csv[-10:]:
        print("   ", l[:72])
else:
    print("  (无 CSV 输出)")

print("[3] 发 'l' 看 LOD...")
s.reset_input_buffer(); send("l")
txt = read_sec(2)
lod = [l for l in txt.splitlines() if "LOD" in l]
print(f"  {lod if lod else '(无回复)'}")

print("[4] 发 'a' 看 AI 置信度...")
s.reset_input_buffer(); send("a")
txt = read_sec(2)
ai = [l for l in txt.splitlines() if "[AI]" in l]
print(f"  {ai if ai else '(无回复)'}")

s.close()
print("[5] 完成")
