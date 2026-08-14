"""真实 ECG 全自动采集 v3: boot -> 状态感知切 AFE -> 录前 REC_LIST 快照 -> 录制 -> 录后 diff 定位新记录.
用法: python rec_collect.py [port] [record_seconds] [out_file]
"""
import serial, sys, time, re

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"
rec_secs = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0
out_file = sys.argv[3] if len(sys.argv) > 3 else "rec_collect_out.txt"

log_lines = []
def log(tag, text):
    line = "[%s] %s" % (tag, text.strip() if text else "")
    log_lines.append(line)
    print(line, flush=True)

s = serial.Serial(port, 460800, timeout=0.2)
s.rts = False
s.dtr = True
s.rts = False
time.sleep(0.2)
s.dtr = False
s.rts = False
log("BOOT", "reset released (run mode)")

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

def parse_idx_lines(text):
    """返回 {(id,dur,samples,abnSec,size): line}"""
    out = {}
    for l in text.splitlines():
        l = l.strip()
        m = re.match(r"^(\d+),(\d+),(\d+),(\d+),(\d+)$", l)
        if m:
            t = tuple(int(x) for x in m.groups())
            out[t] = l
    return out

def csv_truebpm_75_count(text):
    n = 0
    for l in text.splitlines():
        l = l.strip()
        if re.match(r"^-?\d+\.\d+,", l):
            p = l.split(",")
            if len(p) >= 9 and p[4].strip() == "75":
                n += 1
    return n

# 1) boot
boot = read_for(12.0)
for l in boot.splitlines():
    if any(k in l for k in ("AP started", "SSID", "[系统] 当前输入")):
        log("BOOT", l)

# 2) 状态感知切真实 AFE
mode = ""
for i in range(6):
    t = send_read("m", 2.5)
    probe = [l.strip() for l in t.splitlines() if any(k in l for k in ("切换至", "当前输入模式", "回放", "真实AFE", "模拟发生器"))]
    if probe:
        log("MODEPROBE", "第%d次 m 后: %s" % (i + 1, " | ".join(probe[:2])))
    if "真实AFE" in t:
        mode = "AFE_REAL"
        log("MODE", "已确认真实AFE (第%d次 m)" % (i + 1))
        break
if not mode:
    t = read_for(3.0)
    sim = csv_truebpm_75_count(t)
    log("MODEPROBE", "兜底: true_bpm=75 行数=%d (0=非模拟)" % sim)
    if sim == 0:
        mode = "MAYBE_AFE"
        t = send_read("n", 2.5)
        if "回放" in t:
            log("ABORT", "实际是回放模式 — 再按 'm' 一次")
            mode = "AFE_REAL" if "真实AFE" in send_read("m", 2.5) else ""
        elif "当前非回放模式" in t:
            mode = "AFE_REAL"
            log("MODE", "'n' 探针确认: 真实AFE")
if not mode:
    log("ABORT", "未能切到真实AFE")
    s.close()
    open(out_file, "w", encoding="utf-8").write("\n".join(log_lines))
    sys.exit(3)

# 3) 信号自检 (LOD 浮空随机 0.99 报警, 只看峰峰值)
sig = read_for(8.0)
vals = []
alarm_rows = 0
tot_rows = 0
sim_rows = 0
for l in sig.splitlines():
    l = l.strip()
    if re.match(r"^-?\d+\.\d+,", l):
        parts = l.split(",")
        if len(parts) >= 9:
            tot_rows += 1
            try:
                vals.append(float(parts[0]))
                if parts[7] == "1":
                    alarm_rows += 1
                if parts[4].strip() == "75":
                    sim_rows += 1
            except ValueError:
                pass
if sim_rows > 0:
    log("ABORT", "CSV true_bpm=75 行=%d — 仍在模拟模式" % sim_rows)
    s.close()
    open(out_file, "w", encoding="utf-8").write("\n".join(log_lines))
    sys.exit(3)
if vals:
    pp = max(vals) - min(vals)
    log("SIGCHECK", "rows=%d pp=%.4fV min=%.4f max=%.4f alarm_rows=%d (浮空LOD随机报警,忽略)" % (
        tot_rows, pp, min(vals), max(vals), alarm_rows))
    if pp < 0.05:
        log("ABORT", "无信号 (pp<0.05V)")
        s.close()
        open(out_file, "w", encoding="utf-8").write("\n".join(log_lines))
        sys.exit(4)
else:
    log("SIGCHECK", "无 CSV 数据")

# 3.5) 关闭自动录制 + 关闭定时录制 + 停掉一切进行中的录制
t = send_read("REC_AUTO 0", 2.0)
for l in t.splitlines():
    if "REC_AUTO" in l:
        log("AUTOOFF", l)
t = send_read("REC_SCHEDULE OFF", 2.0)
for l in t.splitlines():
    if "REC_SCHEDULE" in l:
        log("SCHEDOFF", l)
t = send_read("REC_STOP", 2.0)
for l in t.splitlines():
    if "REC_STOP" in l:
        log("PRESTOP", l)

# 4) 录前 REC_LIST 快照
pre = parse_idx_lines(send_read("REC_LIST", 3.0))
log("SNAP", "pre-list: %d records" % len(pre))

# 5) 开始录制 (最多 3 次重试: 若 BLE App 抢先开始, 先停掉再开始)
started = False
for attempt in range(3):
    t = send_read("REC_START", 3.0)
    for l in t.splitlines():
        if "REC_START" in l:
            log("RECSTART", l)
    if "REC_START ok" in t:
        started = True
        break
    log("RECSTART", "fail, retry %d" % (attempt + 1))
    send_read("REC_STOP", 2.0)
if not started:
    log("ABORT", "REC_START 三次失败")
    s.close()
    open(out_file, "w", encoding="utf-8").write("\n".join(log_lines))
    sys.exit(5)

# 6) 等待录制
log("REC", "recording %ds ..." % rec_secs)
end = time.time() + rec_secs
drained = 0
while time.time() < end:
    d = s.read(4096)
    if d:
        drained += len(d)
log("REC", "drained %d bytes" % drained)

# 7) 停止
t = send_read("REC_STOP", 3.0)
for l in t.splitlines():
    if "REC_STOP" in l:
        log("RECSTOP", l)

# 8) 录后 REC_LIST, diff 出新增记录
post = parse_idx_lines(send_read("REC_LIST", 3.0))
new = [k for k in post if k not in pre]
log("SNAP", "post-list: %d records, 新增: %d" % (len(post), len(new)))
for k in new:
    log("NEWREC", "id=%d dur=%ds samples=%d abnSec=%d size=%d" % k)

s.close()
open(out_file, "w", encoding="utf-8").write("\n".join(log_lines))
log("DONE", "output saved to %s" % out_file)
