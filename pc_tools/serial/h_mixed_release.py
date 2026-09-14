"""h_mixed_release.py — Round-H §3.4 混合段擎住/解除/再触发时序取证 (TH §114)

时间线 (单次连续捕获, 中途切模式):
  阶段1  0-75s   MODE replay_normal + REPLAY 40 (lo_mixed_cycle 60s 循环):
                10 正常 / 15 工频 / 10 正常 / 20 弹出+噪声 / 5 正常
                预期: t~13s LATCH 含 0x10, 持续擎住
  阶段2  75-190s REPLAY 0 (切干净 MIT-100, 不复位 —— MODE 会 alarmReset,
                那是立即复位不是定时解除): 预期: 定时解除 CLEAR
                (30s 擎住 + SQI 回线 + 有拍 + 密度回落)
  阶段3  之后   REPLAY 37 (lo_pop_noise 再拔线):
                预期: 再次 LATCH 含 0x10

断言三事件齐全 => 擎住/解除/再触发时序闭环。
用法: python h_mixed_release.py [--port COM4]
"""
import argparse
import json
import re
import time

import serial

LATCH_RE = re.compile(r"\[ALARM\] (LATCH|EXTEND|CLEAR)")
SRC_RE = re.compile(r"src=0x([0-9a-fA-F]+)")
TICK_RE = re.compile(r"TICK,\d+,src=\S+?,bpm=\d+,sqi=[0-9.]+,.*alarm=(\d),asrc=0x([0-9a-fA-F]+)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--out", default="captures/h_mixed_release")
    args = ap.parse_args()

    s = serial.Serial(args.port, 460800, timeout=0.2)
    s.rts = False
    s.dtr = True
    s.rts = False
    time.sleep(0.2)
    s.dtr = False
    s.rts = False
    s.reset_input_buffer()

    lines = []

    def capture(seconds, phase):
        t0 = time.time()
        while time.time() - t0 < seconds:
            d = s.read(4096)
            if not d:
                continue
            now = time.time() - t0
            for raw in d.split(b"\n"):
                text = raw.decode("utf-8", "replace").strip()
                if text:
                    lines.append((now, phase, text))

    def send(cmd):
        s.write((cmd + "\n").encode("ascii"))
        s.flush()
        time.sleep(1.5)

    send("MODE replay_normal")
    send("REPLAY 40")
    s.reset_input_buffer()
    capture(75.0, "p1_mixed")
    send("REPLAY 0")   # 不复位: 定时解除语义 (MODE 会 alarmReset)
    capture(115.0, "p2_recover")
    send("REPLAY 37")
    capture(38.0, "p3_repull")
    s.close()

    with open(args.out + "_raw.txt", "w", encoding="utf-8") as f:
        for ts, ph, text in lines:
            f.write("%8.3f [%s] %s\n" % (ts, ph, text))

    events = []
    alarm_p1 = alarm_p3 = clear_p2 = 0
    for ts, ph, text in lines:
        m = LATCH_RE.search(text)
        if m:
            ms = SRC_RE.search(text)
            events.append({"t": round(ts, 2), "phase": ph, "kind": m.group(1),
                           "src": int(ms.group(1), 16) if ms else None})
        if ph == "p1_mixed" and TICK_RE.search(text):
            if int(TICK_RE.search(text).group(1)):
                alarm_p1 += 1
        if ph == "p2_recover" and "CLEAR" in text:
            clear_p2 += 1
        if ph == "p3_repull" and TICK_RE.search(text):
            if int(TICK_RE.search(text).group(1)):
                alarm_p3 += 1

    def phase_srcs(ph):
        out = set()
        for _, p, text in lines:
            if p == ph:
                m = TICK_RE.search(text)
                if m:
                    out.add(int(m.group(2), 16))
        return sorted(out)

    summary = {
        "tool": "h_mixed_release.py",
        "port": args.port,
        "capture_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "events": events,
        "p1_alarm_ticks": alarm_p1,
        "p1_asrc_seen": [hex(a) for a in phase_srcs("p1_mixed")],
        "p2_clear_events": clear_p2,
        "p3_alarm_ticks": alarm_p3,
        "p3_asrc_seen": [hex(a) for a in phase_srcs("p3_repull")],
    }
    ok = (alarm_p1 >= 30 and any(0x10 & a for a in phase_srcs("p1_mixed"))
          and clear_p2 >= 1 and alarm_p3 >= 5
          and any(0x10 & a for a in phase_srcs("p3_repull")))
    summary["verdict"] = "PASS" if ok else "FAIL"
    with open(args.out + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("verdict:", summary["verdict"])


if __name__ == "__main__":
    main()
