"""串口命令检测: 读启动日志 + CSV, 发 l/a 命令看回复.
用法: python serial_check.py [port]"""
import serial, sys, time

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"

def read_sec(s, sec):
    buf = b""
    end = time.time() + sec
    while time.time() < end:
        d = s.read(4096)
        if d:
            buf += d
    return buf.decode("utf-8", "replace")

s = serial.Serial(port, 460800, timeout=0.2)
print("[1] 打开串口, prewait 3s (防 USB 复位)...")
time.sleep(3)
s.reset_input_buffer()

print("[2] 读 3s 原始输出 (看 CSV + 启动日志):")
txt = read_sec(s, 3)
lines = txt.splitlines()
csv = [l for l in lines if l and l[0].isdigit() and "," in l]
noncsv = [l for l in lines if l.strip() and not (l[0].isdigit() and "," in l)]
print(f"  总行 {len(lines)}, CSV 行 {len(csv)}, 非 CSV 行 {len(noncsv)}")
for l in noncsv[:15]:
    print("   |", l)
if csv:
    print("   CSV 样例:", csv[0][:80])
    print("   CSV 末:", csv[-1][:80])

print("[3] 发 'l' 命令, 读 3s:")
s.reset_input_buffer()
s.write(b"l\n")
s.flush()
txt = read_sec(s, 3)
lod = [l for l in txt.splitlines() if "LOD" in l]
print(f"  [LOD] 回复: {lod if lod else '(无回复)'}")

print("[4] 发 'a' 命令, 读 3s:")
s.reset_input_buffer()
s.write(b"a\n")
s.flush()
txt = read_sec(s, 3)
ai = [l for l in txt.splitlines() if "[AI]" in l]
print(f"  [AI] 回复: {ai if ai else '(无回复)'}")

print("[5] 发 'DIAG AI 1' 命令, 读 3s:")
s.reset_input_buffer()
s.write(b"DIAG AI 1\n")
s.flush()
txt = read_sec(s, 3)
diag = [l for l in txt.splitlines() if "DIAG" in l or "AI" in l]
print(f"  [DIAG] 回复: {diag if diag else '(无回复)'}")

s.close()
print("[6] 检测完成")
