"""持续观察真实 ECG 下 AI 置信度分布: 读 N 秒 CSV, 统计 confidence 列分布.
用法: python serial_ai_dist.py [port] [seconds]"""
import serial, sys, time

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0

s = serial.Serial(port, 460800, timeout=0.2)
time.sleep(1.5)
s.reset_input_buffer()

end = time.time() + dur
rows = []
while time.time() < end:
    d = s.read(4096)
    if not d: continue
    for line in d.decode("utf-8", "replace").splitlines():
        p = line.split(",")
        if len(p) >= 9:
            try:
                rows.append([float(p[2]), int(p[3]), float(p[5]), int(p[7]), float(p[8])])
            except ValueError:
                pass
s.close()

n = len(rows)
if n == 0:
    print("(no CSV)"); sys.exit(1)

# confidence 分类
confs = [r[4] for r in rows]
nonzero = [c for c in confs if c > 0.001]
abn = sum(1 for r in rows if r[3] == 1)
sqi = [r[2] for r in rows]
bpm = [r[1] for r in rows if 30 <= r[1] <= 200]

print(f"样本 {n}, 时长 {dur:.0f}s")
print(f"bpm: {min(bpm) if bpm else 'N/A'}~{max(bpm) if bpm else 'N/A'} (有效 {len(bpm)}/{n})")
print(f"sqi: mean={sum(sqi)/n:.3f}  min={min(sqi):.3f}  max={max(sqi):.3f}")
print(f"abnormal=1: {abn}/{n} ({abn*100//n}%)")

# confidence 分布 (排除 0.000 无结果)
if nonzero:
    print(f"AI 置信度 (非零样本 {len(nonzero)}):")
    print(f"  min={min(nonzero):.3f}  max={max(nonzero):.3f}  mean={sum(nonzero)/len(nonzero):.3f}")
    # 直方图 (0.1 桶)
    import collections
    hist = collections.Counter(int(c*10) for c in nonzero)
    for k in sorted(hist):
        print(f"  0.{k}~0.{k+1}: {'#' * hist[k]} ({hist[k]})")
else:
    print("AI 置信度: 全部 0.000 (AI 始终判正常)")
