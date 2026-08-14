"""串口基线漂移分析: 读串口 CSV, 统计 filtered 列(第3列)的每秒均值/峰峰值, 观察基线漂移.
用法: python serial_baseline.py [port] [seconds]"""
import serial, sys, time

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

s = serial.Serial(port, 460800, timeout=0.2)
time.sleep(1.5)          # USB-Serial-JTAG 打开稳定 + 防复位
s.reset_input_buffer()

end = time.time() + dur
t0 = None
samples = []
while time.time() < end:
    d = s.read(4096)
    if not d:
        continue
    for line in d.decode("utf-8", "replace").splitlines():
        p = line.split(",")
        if len(p) >= 9:
            try:
                v = float(p[2])          # filtered 列
                if t0 is None:
                    t0 = time.time()
                samples.append((time.time() - t0, v))
            except ValueError:
                pass
s.close()

n = len(samples)
if n == 0:
    print("(no CSV data)")
    sys.exit(1)

vals = [v for _, v in samples]
mean = sum(vals) / n
mn = min(vals)
mx = max(vals)

print(f"样本数 {n}, 时长 {samples[-1][0]:.1f}s")
print(f"filtered 总体: mean={mean*1000:+.2f}mV  min={mn*1000:+.2f}mV  max={mx*1000:+.2f}mV  pp={(mx-mn)*1000:.2f}mV")

# 每秒桶均值 (基线漂移观察)
buckets = {}
for t, v in samples:
    b = int(t)
    buckets.setdefault(b, []).append(v)

print("每秒 filtered 均值 (基线漂移):")
for b in sorted(buckets):
    bv = buckets[b]
    bm = sum(bv) / len(bv)
    bpp = max(bv) - min(bv)
    print(f"  t={b:2d}s  n={len(bv):4d}  mean={bm*1000:+7.2f}mV  pp={bpp*1000:7.2f}mV")
