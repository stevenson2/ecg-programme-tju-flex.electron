# serial_wave_capture.py — serial-only WAVE/CSV capture to real_captures/inbox
# Usage: python serial_wave_capture.py COM15 120
import serial, sys, time, os, json, re
from datetime import datetime

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM15"
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
BAUD = 460800
INBOX = r"C:\Users\cai\OneDrive\Desktop\Fe programme 25261\real_captures\inbox"
os.makedirs(INBOX, exist_ok=True)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
session_id = f"serial_{stamp}_wearable_single_lead_AD8232"
csv_path = os.path.join(INBOX, f"{session_id}.csv")
meta_path = os.path.join(INBOX, f"{session_id}_meta.json")
log_path = os.path.join(INBOX, f"{session_id}_log.txt")

log_lines = []
def log(tag, msg):
    line = f"[{tag}] {msg}"
    log_lines.append(line)
    print(line, flush=True)

# Open WITHOUT DTR reset (DTR toggle boots ESP32 into SIMULATOR often)
ser = serial.Serial()
ser.port = PORT
ser.baudrate = BAUD
ser.timeout = 0.25
ser.write_timeout = 2
ser.dsrdtr = False
ser.rtscts = False
ser.dtr = False
ser.rts = False
ser.open()
time.sleep(0.3)
ser.reset_input_buffer()
log("OPEN", f"{PORT} @ {BAUD} pid={os.getpid()} dtr=False")

def read_for(sec):
    end = time.time() + sec
    buf = b""
    while time.time() < end:
        d = ser.read(8192)
        if d:
            buf += d
    return buf.decode("utf-8", "replace")

def send(cmd, wait=1.0):
    ser.write((cmd + "\n").encode("ascii"))
    ser.flush()
    return read_for(wait)

# Enable WAVE stream
t = send("WAVE 1", 1.5)
log("WAVE", "sent WAVE 1; reply_snip=" + repr(t[:200].replace("\n"," | ")))

# Prefer explicit MODE afe (also try lowercase 'a' / 'm' if needed)
confirmed = False
afe_evidence = []
for attempt, cmd in enumerate(["MODE afe", "afe", "a", "m", "m", "m"], 1):
    t = send(cmd, 2.0)
    probes = [l.strip() for l in t.splitlines() if any(k in l for k in ("切换至", "当前输入", "真实AFE", "模拟", "回放", "AFE", "MODE"))]
    if probes:
        log("MODE", f"cmd={cmd!r} probe: {' | '.join(probes[:3])}")
    # Parse CSV rows for true_bpm (col index 4): 0 => AFE/real, 75 => simulator typically
    rows = []
    for l in t.splitlines():
        l = l.strip()
        if re.match(r"^-?\d+\.?\d*,", l):
            p = l.split(",")
            if len(p) >= 5:
                rows.append(p)
    # Also drain extra 3s of live stream
    extra = read_for(3.0)
    for l in extra.splitlines():
        l = l.strip()
        if re.match(r"^-?\d+\.?\d*,", l):
            p = l.split(",")
            if len(p) >= 5:
                rows.append(p)
    if not rows:
        log("MODE", f"cmd={cmd!r} no CSV yet")
        continue
    tb = {}
    for p in rows:
        k = p[4].strip()
        tb[k] = tb.get(k, 0) + 1
    log("MODE", f"cmd={cmd!r} true_bpm hist={tb} n={len(rows)}")
    # AFE: true_bpm mostly 0; SIM: often 75
    zero_n = tb.get("0", 0)
    sim_n = tb.get("75", 0)
    if zero_n >= max(10, int(0.7 * len(rows))) and sim_n == 0:
        confirmed = True
        afe_evidence.append(f"true_bpm=0 rows={zero_n}/{len(rows)} after {cmd}")
        log("AFE_OK", afe_evidence[-1])
        break
    if "真实AFE" in t:
        confirmed = True
        afe_evidence.append(f"text 真实AFE after {cmd}")
        log("AFE_OK", afe_evidence[-1])
        break

if not confirmed:
    # Final 4s sample
    final = read_for(4.0)
    rows = []
    for l in final.splitlines():
        l = l.strip()
        if re.match(r"^-?\d+\.?\d*,", l):
            p = l.split(",")
            if len(p) >= 5:
                rows.append(p)
    tb = {}
    for p in rows:
        k = p[4].strip()
        tb[k] = tb.get(k, 0) + 1
    log("AFE_FAIL", f"could not confirm AFE; hist={tb} n={len(rows)}")
    open(log_path, "w", encoding="utf-8").write("\n".join(log_lines))
    meta = {
        "session_id": session_id,
        "status": "aborted_not_afe",
        "port": PORT,
        "true_bpm_hist": tb,
        "afe_confirmed": False,
        "note": "serial capture aborted — src not confirmed AFE",
    }
    open(meta_path, "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False, indent=2))
    ser.close()
    sys.exit(3)

# Capture header: keep raw CSV columns as board emits them
header = "clean,noisy,filtered,bpm,true_bpm,sqi,motion,abnormal,conf,asrc\n"
# Board may send 7–10 cols; we store raw line + host timestamp
with open(csv_path, "w", encoding="utf-8", newline="\n") as f:
    f.write("host_ts_s,raw_line\n")

log("CAPTURE", f"start {DURATION}s -> {csv_path}")
t0 = time.time()
n_lines = 0
n_csv = 0
tb_hist = {}
last_prog = t0
ser.reset_input_buffer()
send("WAVE 1", 0.3)

with open(csv_path, "a", encoding="utf-8", newline="\n") as f:
    while time.time() - t0 < DURATION:
        raw = ser.read(8192)
        if not raw:
            continue
        text = raw.decode("utf-8", "replace")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            elapsed = time.time() - t0
            if re.match(r"^-?\d+\.?\d*,", line):
                n_csv += 1
                parts = line.split(",")
                if len(parts) >= 5:
                    k = parts[4].strip()
                    tb_hist[k] = tb_hist.get(k, 0) + 1
                # escape commas already in line; store as quoted raw
                safe = line.replace('"', '""')
                f.write(f'{elapsed:.3f},"{safe}"\n')
        if time.time() - last_prog >= 10:
            log("PROG", f"t={time.time()-t0:.0f}s csv={n_csv} hist={tb_hist}")
            last_prog = time.time()
            f.flush()

elapsed = time.time() - t0
send("WAVE 0", 0.5)
ser.close()

sim_leak = tb_hist.get("75", 0)
afe_ok = (tb_hist.get("0", 0) > 0) and (sim_leak == 0 or sim_leak < 0.05 * max(1, n_csv))
meta = {
    "session_id": session_id,
    "capture_method": "serial_wave_csv",
    "port": PORT,
    "baud": BAUD,
    "afe": "AD8232",
    "lead": "wearable_single_lead (user 2026-10-20 16:08; placement may differ from earlier sessions)",
    "afe_confirmed": bool(afe_ok and confirmed),
    "afe_evidence": afe_evidence,
    "true_bpm_hist": tb_hist,
    "duration_s_requested": DURATION,
    "duration_s_actual": round(elapsed, 2),
    "csv_rows": n_csv,
    "total_lines_seen": n_lines,
    "csv_path": csv_path,
    "pid": os.getpid(),
    "firmware_note": "true_bpm==0 treated as AFE/real; 75 as simulator",
    "clipping_note": "user reported baseline drift clipping — review filtered/noisy cols",
    "label_status": "unlabeled_raw_capture",
    "do_not_use_onboard_ai_as_gold": True,
    "created_at_local": datetime.now().isoformat(timespec="seconds"),
    "timezone": "Asia/Shanghai",
}
open(meta_path, "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False, indent=2))
open(log_path, "w", encoding="utf-8").write("\n".join(log_lines + [f"[DONE] csv={n_csv} elapsed={elapsed:.1f}s afe_ok={afe_ok}"]))
log("DONE", f"csv_rows={n_csv} elapsed={elapsed:.1f}s afe_ok={afe_ok} path={csv_path}")
print("META", meta_path, flush=True)
sys.exit(0 if (n_csv > 100 and afe_ok) else 4)
