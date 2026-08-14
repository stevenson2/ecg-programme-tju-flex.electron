"""串口诊断 v3: 过滤解析异常值, 区分 abnormal 来源 (AI 置信度 vs LOD/flatline 0.99).
用法: python serial_diag3.py [port] [seconds]"""
import serial, sys, time

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

s = serial.Serial(port, 460800, timeout=0.2)
time.sleep(1.5)
s.reset_input_buffer()

end = time.time() + dur
raw = 0
rows = []
while time.time() < end:
    d = s.read(4096)
    if not d:
        continue
    for line in d.decode("utf-8", "replace").splitlines():
        p = line.split(",")
        if len(p) < 9:
            continue
        try:
            r = [float(p[0]), float(p[1]), float(p[2]), int(p[3]),
                 int(p[4]), float(p[5]), int(p[6]), int(p[7]), float(p[8])]
        except ValueError:
            continue
        raw += 1
        # 过滤明显解析错误 (正常 ECG/AFE 信号不会超过 ±5V)
        if abs(r[0]) > 5.0 or abs(r[2]) > 5.0:
            continue
        rows.append(r)
s.close()

n = len(rows)
print(f"样本 {n} (丢弃 {raw-n} 异常行)")

def stats(idx, name):
    v = [r[idx] for r in rows]
    m = sum(v) / n
    return f"{name}: mean={m*1000:+.2f}mV  pp={(max(v)-min(v))*1000:.2f}mV"

print(stats(0, "clean"))
print(stats(2, "filtered"))

sqi = [r[5] for r in rows]
bpm = [r[3] for r in rows if 30 <= r[3] <= 200]
abn = [r for r in rows if r[7] == 1]
motion = sum(1 for r in rows if r[6] == 1)
print(f"sqi: mean={sum(sqi)/n:.3f}  min={min(sqi):.3f}  max={max(sqi):.3f}")
print(f"bpm 有效: {len(bpm)}/{n} ({len(bpm)*100//n}%), {min(bpm) if bpm else 'N/A'}~{max(bpm) if bpm else 'N/A'}")
print(f"motion=1: {motion}/{n}")
print(f"abnormal=1: {len(abn)}/{n} ({len(abn)*100//n}%)")

# abnormal 来源: confidence=0.99 => LOD/flatline 强制; 否则 AI 概率
lod_like = sum(1 for r in abn if r[8] > 0.98)
ai_like = len(abn) - lod_like
print(f"  abnormal 来源: conf≈0.99(LOD/flatline 强制)={lod_like}, AI 概率={ai_like}")
if ai_like > 0:
    aiconf = [r[8] for r in abn if r[8] <= 0.98]
    print(f"  AI 置信度分布: min={min(aiconf):.3f} max={max(aiconf):.3f} mean={sum(aiconf)/len(aiconf):.3f}")

# 正常样本的 confidence (abnormal=0)
norm = [r for r in rows if r[7] == 0]
if norm:
    nc = [r[8] for r in norm]
    print(f"  正常样本 AI 置信度: mean={sum(nc)/len(nc):.3f} max={max(nc):.3f}")
