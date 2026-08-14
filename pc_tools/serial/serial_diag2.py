"""串口全列诊断: 读串口 CSV, 统计 clean/noisy/filtered + sqi/bpm/motion/abnormal.
用法: python serial_diag2.py [port] [seconds]"""
import serial, sys, time

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

s = serial.Serial(port, 460800, timeout=0.2)
time.sleep(1.5)
s.reset_input_buffer()

end = time.time() + dur
rows = []
while time.time() < end:
    d = s.read(4096)
    if not d:
        continue
    for line in d.decode("utf-8", "replace").splitlines():
        p = line.split(",")
        if len(p) >= 9:
            try:
                rows.append([float(p[0]), float(p[1]), float(p[2]),
                             int(p[3]), int(p[4]), float(p[5]),
                             int(p[6]), int(p[7]), float(p[8])])
            except ValueError:
                pass
s.close()

n = len(rows)
if n == 0:
    print("(no CSV data)")
    sys.exit(1)

def stats(idx, name, unit=1000.0):
    v = [r[idx] for r in rows]
    m = sum(v) / n
    return f"{name}: mean={m*unit:+.2f}  min={min(v)*unit:+.2f}  max={max(v)*unit:+.2f}  pp={(max(v)-min(v))*unit:.2f}"

print(f"样本 {n}, 时长 {dur:.0f}s")
print(stats(0, "clean", 1000.0))
print(stats(1, "noisy", 1000.0))
print(stats(2, "filtered", 1000.0))

sqi = [r[5] for r in rows]
bpm = [r[3] for r in rows if 30 <= r[3] <= 200]
motion = sum(1 for r in rows if r[6] == 1)
abn = sum(1 for r in rows if r[7] == 1)
print(f"sqi: mean={sum(sqi)/n:.3f}  min={min(sqi):.3f}  max={max(sqi):.3f}")
print(f"bpm: 有效值 {len(bpm)}/{n}  ({len(bpm)*100//n}%), 范围 {min(bpm) if bpm else 'N/A'}~{max(bpm) if bpm else 'N/A'}")
print(f"motion=1: {motion}/{n} ({motion*100//n}%)")
print(f"abnormal=1: {abn}/{n} ({abn*100//n}%)")

# 前 10 个样本原始值 (看波形形态)
print("前 10 样本 (clean, filtered, bpm, sqi):")
for r in rows[:10]:
    print(f"  clean={r[0]*1000:+.1f}mV  filt={r[2]*1000:+.1f}mV  bpm={r[3]}  sqi={r[5]:.3f}  abn={r[7]}")
